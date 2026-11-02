"""Checks for B2 (U-Net) and B3 (U-Net + FiLM + NLL), on fake data.

Two kinds of check, in one run:

**Unit** -- the properties that are easy to get quietly wrong and impossible to
eyeball later:

  * the parameter counts B2 and B3 are *supposed* to have, since "capacity
    matched" is a claim the physics comparison rests on;
  * B3 at initialisation computing exactly what B2 computes, so any benefit
    from conditioning is learned rather than an initialisation artefact;
  * the NLL being minimised at the true error scale -- the one analytic
    property of a heteroscedastic model, checked against a closed form;
  * masked pixels contributing nothing, which is the rule the 09-10 meeting
    set and the sort of thing that silently poisons a loss;
  * the calibration metric reporting 1.0 on data that is calibrated by
    construction, and >1 when sigma is deliberately too small. A calibration
    metric nobody has pointed at a known answer is not evidence.

**End to end** -- prepare, train both models for a couple of epochs, benchmark,
and check the artefacts exist and contain what the report will quote.

    python tests/test_b2_b3.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from tests.make_fake_data import make_all_splits

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "_b2b3_run"


def check(label: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def build_sandbox_config() -> Path:
    """A config pointing at the sandbox, sized for a 2-minute run."""
    cfg_dir = SANDBOX / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    base["paths"] = {
        "raw_dir": "_b2b3_run/data/raw",
        "processed_dir": "_b2b3_run/data/processed",
        "splits_dir": "_b2b3_run/data/splits",
        "checkpoints_dir": "_b2b3_run/results/checkpoints",
        "results_dir": "_b2b3_run/results/metrics",
        "figures_dir": "_b2b3_run/results/figures",
    }
    base["dataset"]["expected_counts"] = {}
    base["dataloader"]["num_workers"] = 0
    base["dataloader"]["pin_memory"] = False
    base["dataloader"]["batch_size"] = 16
    base["seeds"] = [42, 123]
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False),
                                         encoding="utf-8")

    for name in ("b0_baseline.yaml", "b1_baseline.yaml"):
        shutil.copy(ROOT / "config" / name, cfg_dir / name)

    # A narrow net trains in seconds on CPU. The parameter count then differs
    # from the prescribed one, so the assertion is dropped here deliberately --
    # the full-width count is checked as a unit test instead.
    b2 = yaml.safe_load((ROOT / "config" / "b2_unet.yaml").read_text(encoding="utf-8"))
    b2["model"]["base_channels"] = 8
    b2["model"].pop("expected_parameters", None)
    b2["training"]["epochs"] = 3
    b2["training"]["early_stopping_patience"] = None
    (cfg_dir / "b2_unet.yaml").write_text(yaml.safe_dump(b2, sort_keys=False),
                                          encoding="utf-8")

    b3 = yaml.safe_load(
        (ROOT / "config" / "b3_unet_film_nll.yaml").read_text(encoding="utf-8")
    )
    b3["model"]["base_channels"] = 8
    b3["model"].pop("expected_parameters", None)
    b3["training"]["epochs"] = 4
    b3["training"]["early_stopping_patience"] = None
    b3["uncertainty"]["nll_warmup_epochs"] = 2
    (cfg_dir / "b3_unet_film_nll.yaml").write_text(
        yaml.safe_dump(b3, sort_keys=False), encoding="utf-8"
    )
    return cfg_dir


def run_script(name: str, argv: list[str]) -> int:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"script_{name[:2]}", path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [str(path)] + argv
    try:
        spec.loader.exec_module(mod)
        return mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------- unit checks


def unit_checks(failures: list[str]) -> None:
    from src.config import load_config
    from src.evaluation.uncertainty import (
        anomaly_candidates,
        calibration,
        gaussian_nll,
        per_galaxy_nll,
        sigma_to_physical,
        verdict,
    )
    from src.models.b2_unet import UNet, build_b2, check_parameter_count
    from src.models.b3_unet_film_nll import (
        UNetFiLMNLL,
        build_b3,
        make_b3_loss,
        masked_gaussian_nll,
        masked_l1_on_mean,
    )

    cfg = load_config()

    print("\n--- architecture ---")
    b2 = build_b2(cfg.b2["model"])
    n2 = b2.count_parameters()
    expected2 = int(cfg.b2["model"]["expected_parameters"])
    check("B2 parameter count matches its config", n2 == expected2,
          f"{n2:,} vs {expected2:,}", failures)

    b3 = build_b3(cfg.b3["model"], cfg.b3["film"], cfg.b3["uncertainty"])
    n3 = b3.count_parameters()
    expected3 = int(cfg.b3["model"]["expected_parameters"])
    check("B3 parameter count matches its config", n3 == expected3,
          f"{n3:,} vs {expected3:,}", failures)

    gap = abs(n3 - n2) / n2
    check("B2 and B3 are capacity-matched within 5%", gap < 0.05,
          f"{gap:.2%} apart ({n2:,} vs {n3:,})", failures)

    bd = b3.parameter_breakdown()
    check("B3's breakdown adds up", bd["backbone"] + bd["film_generator"] == n3,
          f"{bd['backbone']:,} + {bd['film_generator']:,} = {n3:,}", failures)

    x = torch.randn(2, 4, 64, 64)
    out2 = b2(x)
    check("B2 output shape", tuple(out2.shape) == (2, 1, 64, 64),
          str(tuple(out2.shape)), failures)
    check("B2 output is finite", bool(torch.isfinite(out2).all()),
          "no NaN or inf", failures)

    mu, logvar = b3(x, torch.tensor([0.1, 0.3]))
    check("B3 returns two maps of the right shape",
          tuple(mu.shape) == (2, 1, 64, 64) == tuple(logvar.shape),
          f"mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}", failures)

    # The drift guard must actually fire, or it is decoration.
    try:
        check_parameter_count(123, {"expected_parameters": 1_000_000}, "T")
        check("parameter drift is rejected", False, "no error raised", failures)
    except ValueError as exc:
        check("parameter drift is rejected", "capacity match" in str(exc),
              "ValueError naming the capacity match", failures)

    # An image size the depth cannot halve cleanly must fail loudly, not
    # silently misalign the skip connections.
    try:
        UNet(depth=3)(torch.randn(1, 4, 60, 60))
        check("odd image size is rejected", False, "no error raised", failures)
    except ValueError as exc:
        check("odd image size is rejected", "divisible" in str(exc),
              "ValueError mentioning divisibility", failures)

    print("\n--- FiLM ---")
    # At init, gamma=1 and beta=0, so conditioning must change nothing.
    mu_cond, _ = b3(x, torch.tensor([0.05, 0.35]))
    mu_plain, _ = b3(x, None)
    delta = float((mu_cond - mu_plain).abs().max())
    check("B3 at init equals B2's computation", delta == 0.0,
          f"max |difference| = {delta:.2e}", failures)

    lv0 = float(logvar.abs().max())
    check("B3 starts at sigma = 1 in standardised units", lv0 == 0.0,
          f"max |logvar| = {lv0:.2e}", failures)

    # With non-zero FiLM weights the redshift must move the output, otherwise
    # the conditioning is wired up wrong and no amount of training would help.
    b3b = UNetFiLMNLL(base_channels=8, identity_init=False)
    with torch.no_grad():
        a, _ = b3b(x, torch.full((2,), 0.05))
        b, _ = b3b(x, torch.full((2,), 0.39))
    moved = float((a - b).abs().max())
    check("redshift changes the prediction when FiLM is active", moved > 1e-6,
          f"max |difference| = {moved:.3e}", failures)

    print("\n--- the NLL ---")
    # Closed form: for a fixed residual r, 0.5*(logvar + r^2/exp(logvar)) is
    # minimised at sigma = |r|. Checking a loss against its analytic optimum is
    # the only way to know the sign and the factor of two are right.
    target = torch.zeros(1, 1, 8, 8)
    pred = torch.full((1, 1, 8, 8), 2.0)
    mask = torch.ones_like(target)
    losses = {}
    for sigma in (0.5, 1.0, 2.0, 4.0):
        lv = torch.full_like(target, float(np.log(sigma**2)))
        losses[sigma] = float(masked_gaussian_nll(pred, lv, target, mask))
    best = min(losses, key=losses.get)
    check("NLL is minimised at sigma = |residual|", best == 2.0,
          f"residual 2.0, best sigma {best} from {sorted(losses)}", failures)

    # Masked pixels must not contribute. Corrupting them must change nothing.
    mask2 = torch.ones_like(target)
    mask2[..., :4, :] = 0.0
    pred2 = pred.clone()
    pred2[..., :4, :] = 1e6
    lv = torch.zeros_like(target)
    a_loss = float(masked_gaussian_nll(pred, lv, target, mask2))
    b_loss = float(masked_gaussian_nll(pred2, lv, target, mask2))
    check("masked pixels contribute nothing to the NLL", a_loss == b_loss,
          f"{a_loss:.6f} with and without a 1e6 corruption behind the mask",
          failures)

    a_l1 = float(masked_l1_on_mean(pred, target, mask2))
    b_l1 = float(masked_l1_on_mean(pred2, target, mask2))
    check("masked pixels contribute nothing to the warmup L1", a_l1 == b_l1,
          f"{a_l1:.6f} either way", failures)

    # The warmup must actually switch objective on the stated epoch.
    loss_fn = make_b3_loss(warmup_epochs=2)
    lv_big = torch.full_like(target, 6.0)
    during = float(loss_fn((pred, lv_big), target, mask, epoch=2))
    after = float(loss_fn((pred, lv_big), target, mask, epoch=3))
    check("warmup uses L1 then switches to NLL",
          during == float(masked_l1_on_mean(pred, target, mask))
          and after == float(masked_gaussian_nll(pred, lv_big, target, mask)),
          f"epoch 2 -> {during:.4f} (L1), epoch 3 -> {after:.4f} (NLL)",
          failures)
    check("the switching loss advertises its warmup",
          getattr(loss_fn, "warmup_epochs", None) == 2,
          "warmup_epochs = 2, which the Trainer reads to reset best-val",
          failures)

    print("\n--- sigma in physical units ---")
    stats = {"denom": np.array([1.0, 2.0, 3.0, 4.0, 7.0]),
             "mean": np.array([0.0, 0.0, 0.0, 0.0, 100.0])}
    phys = sigma_to_physical(np.array([2.0]), stats)
    check("sigma scales by the band divisor and ignores the mean",
          float(phys[0]) == 14.0, f"2.0 * 7.0 = {float(phys[0])}", failures)

    print("\n--- calibration, against a known answer ---")
    rng = np.random.default_rng(0)
    truth = rng.normal(0.0, 1.0, size=(200, 1, 32, 32))
    sigma_true = np.full_like(truth, 1.0)
    cal_ok = calibration(truth, np.zeros_like(truth), sigma_true)
    check("calibrated data reports a ratio near 1",
          abs(cal_ok["ratio_std"] - 1.0) < 0.02,
          f"ratio {cal_ok['ratio_std']:.4f}", failures)
    cov68 = cal_ok["coverage"]["0.68"]["empirical"]
    check("68% interval covers ~68% of calibrated pixels",
          abs(cov68 - 0.68) < 0.02, f"{cov68 * 100:.1f}%", failures)

    cal_bad = calibration(truth, np.zeros_like(truth), sigma_true * 0.5)
    check("halved sigma is reported as over-confident",
          cal_bad["ratio_std"] > 1.8,
          f"ratio {cal_bad['ratio_std']:.3f} (expected ~2)", failures)
    v = verdict({"calibration": cal_bad,
                 "sharpness": {"spearman_sigma_vs_abs_error": 0.0}})
    check("the verdict names over-confidence",
          v["calibration_state"] == "over-confident",
          v["calibration_state"], failures)
    v_ok = verdict({"calibration": cal_ok,
                    "sharpness": {"spearman_sigma_vs_abs_error": 0.5}})
    check("the verdict accepts calibrated data",
          v_ok["calibration_state"] == "calibrated"
          and v_ok["knows_where_it_is_uncertain"],
          f"{v_ok['calibration_state']}, resolution ok", failures)

    nll_ok = gaussian_nll(truth, np.zeros_like(truth), sigma_true)
    # For unit-variance residuals scored with sigma=1 the expected NLL is
    # 0.5*log(2*pi) + 0.5 = 1.4189.
    check("NLL matches its analytic value on calibrated data",
          abs(nll_ok["nll"] - 1.4189) < 0.02,
          f"{nll_ok['nll']:.4f} vs 1.4189", failures)

    print("\n--- the anomaly score ---")
    # One galaxy is wrong and confident; another is wrong but honest about it.
    y = rng.normal(0.0, 1.0, size=(20, 1, 16, 16))
    pred_a = np.zeros_like(y)
    sig = np.ones_like(y)
    y[7] += 8.0            # confidently wrong -> should rank first
    y[11] += 8.0
    sig[11] = 12.0         # equally wrong, but said so -> should not
    scores = per_galaxy_nll(y, pred_a, sig)
    ranked = np.argsort(-scores)
    check("the confidently-wrong galaxy ranks first", int(ranked[0]) == 7,
          f"top 3 = {[int(v) for v in ranked[:3]]} (7 is wrong+confident, "
          f"11 is wrong+uncertain)", failures)
    cand = anomaly_candidates(y, pred_a, sig, np.arange(20) * 3, top_k=5)
    check("the shortlist carries object ids", cand["object_id"][0] == 21,
          f"object_id {cand['object_id'][0]} for index 7", failures)
    check("the shortlist is honest about what it is",
          "not a detection" in cand["note"], "caveat present", failures)

    print("\n--- the Trainer's objective switch ---")
    from src.training.trainer import Trainer

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, a, cond=None):
            mu = a[:, 3:4] * self.w
            return mu, torch.zeros_like(mu)

    batch = {
        "input": torch.randn(4, 4, 8, 8),
        "target": torch.randn(4, 1, 8, 8),
        "mask": torch.ones(4, 1, 8, 8),
        "redshift": torch.rand(4),
    }
    loader = [batch]
    tr = Trainer(Tiny(), loader, loader, epochs=4, device="cpu",
                 loss_fn=make_b3_loss(2), conditioning_key="redshift")
    check("the Trainer knows when the objective changes",
          tr.objective_changes_at_epoch == 3,
          f"epoch {tr.objective_changes_at_epoch} (warmup 2)", failures)
    fit = tr.fit()
    check("the best epoch is chosen after the switch, not during the warmup",
          fit["best_epoch"] >= 3,
          f"best epoch {fit['best_epoch']} of {fit['epochs_run']}", failures)

    # A plain f(pred, target, mask) loss must still work untouched -- B1 and B2
    # depend on it.
    tr2 = Trainer(torch.nn.Conv2d(4, 1, 1), loader, loader, epochs=1,
                  device="cpu")
    check("the default masked-L1 path still works",
          np.isfinite(tr2.fit()["best_val_loss"]), "B1/B2 path unchanged",
          failures)


# ------------------------------------------------------------ end-to-end run


def end_to_end(failures: list[str]) -> None:
    from src.config import load_config

    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    cfg_dir = build_sandbox_config()
    make_all_splits(SANDBOX / "data" / "raw", n_train=160, n_val=48, n_test=48)
    cfg = load_config(cfg_dir)

    print("\n--- 01_prepare_data ---")
    status = run_script("01_prepare_data.py", ["--config", str(cfg_dir)])
    # Exit 1 means the audit reported problems, and on fake data it always
    # does: the files hold a few hundred galaxies rather than the official
    # 204,573. The filtering itself still has to succeed.
    check("01_prepare_data ran", status in (0, 1), f"exit code {status}",
          failures)

    print("\n--- 02_run_b0 (for the ladder table) ---")
    status = run_script("02_run_b0.py", ["--config", str(cfg_dir)])
    check("02_run_b0 ran", status == 0, f"exit code {status}", failures)

    print("\n--- 08_train_b2 ---")
    status = run_script("08_train_b2.py", ["--config", str(cfg_dir), "--seed", "42"])
    check("08_train_b2 ran", status == 0, f"exit code {status}", failures)

    res = cfg.path("results_dir")
    import json

    b2_path = res / "b2_metrics.json"
    check("b2_metrics.json written", b2_path.exists(), str(b2_path), failures)
    if b2_path.exists():
        b2 = json.loads(b2_path.read_text(encoding="utf-8"))
        mae = b2["metrics_over_seeds"]["pixel"]["mae"]["mean"]
        check("B2's MAE is finite", np.isfinite(mae), f"MAE {mae:.5g}", failures)
        hist = b2["per_seed"][0]["history"]
        improved = hist[-1]["train_loss"] < hist[0]["train_loss"]
        check("B2's training loss went down", improved,
              f"{hist[0]['train_loss']:.5f} -> {hist[-1]['train_loss']:.5f}",
              failures)
        ckpt = Path(b2["per_seed"][0]["checkpoint"])
        check("B2's checkpoint exists", ckpt.exists(),
              f"{ckpt.stat().st_size / 1e6:.2f} MB" if ckpt.exists() else "missing",
              failures)

    print("\n--- 09_train_b3 ---")
    status = run_script("09_train_b3.py", ["--config", str(cfg_dir), "--seed", "42"])
    check("09_train_b3 ran", status == 0, f"exit code {status}", failures)

    b3_path = res / "b3_metrics.json"
    check("b3_metrics.json written", b3_path.exists(), str(b3_path), failures)
    if b3_path.exists():
        b3 = json.loads(b3_path.read_text(encoding="utf-8"))
        m = b3["metrics_over_seeds"]
        check("B3 reports an uncertainty block", "uncertainty" in m,
              ", ".join(sorted(m.get("uncertainty", {}))), failures)
        nll = m["uncertainty"]["nll"]["nll"]["mean"]
        check("B3's NLL is finite", np.isfinite(nll), f"NLL {nll:.5g}", failures)
        ratio = m["uncertainty"]["calibration"]["ratio_std"]["mean"]
        check("B3 reports a calibration ratio", np.isfinite(ratio) and ratio > 0,
              f"(y-mu)/sigma spread {ratio:.3f}", failures)
        anom = b3["anomaly_candidates_seed0"]
        check("B3 wrote an anomaly shortlist", len(anom["object_id"]) > 0,
              f"{anom['top_k']} candidates", failures)
        resp = b3["film_response_per_seed"][0]
        check("the FiLM response is measured and reported",
              resp is not None and "relative_response" in resp,
              f"relative response {resp['relative_response']:.4g}", failures)
        switched = b3["per_seed"][0]["objective_changed_at_epoch"]
        check("the objective switch is recorded", switched == 3,
              f"epoch {switched} (warmup 2)", failures)
        arr = res / "arrays_b3_seed42_test.npz"
        if arr.exists():
            with np.load(arr) as z:
                keys = sorted(z.files)
                has_sigma = "sigma_median_per_galaxy" in keys
                check("per-galaxy sigma is saved for later re-ranking",
                      has_sigma, ", ".join(keys), failures)

    print("\n--- 10_benchmark_all ---")
    status = run_script("10_benchmark_all.py", ["--config", str(cfg_dir)])
    check("10_benchmark_all ran", status == 0, f"exit code {status}", failures)

    table = res / "ladder_table.md"
    check("ladder_table.md written", table.exists(), str(table), failures)
    if table.exists():
        text = table.read_text(encoding="utf-8")
        for needle in ("B0 (Copy-z)", "B2 (U-Net)", "What each rung bought",
                       "calibration"):
            check(f"the table covers {needle!r}", needle in text,
                  "present" if needle in text else "missing", failures)
    for name in ("b3_calibration.png", "ladder_mae_vs_params.png"):
        p = cfg.path("figures_dir") / name
        check(f"figure {name}", p.exists() and p.stat().st_size > 5000,
              f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "missing",
              failures)
    csv_path = res / "b3_anomaly_candidates.csv"
    check("the anomaly CSV is written", csv_path.exists(),
          str(csv_path), failures)

    # The table must survive a model not having been run -- that is the normal
    # state of this repo while the ladder is being climbed.
    print("\n--- 10_benchmark_all with B3 excluded ---")
    status = run_script("10_benchmark_all.py",
                        ["--config", str(cfg_dir), "--models", "b0", "b2"])
    check("the ladder table tolerates missing models", status == 0,
          f"exit code {status}", failures)


def main() -> int:
    failures: list[str] = []
    print("=" * 74)
    print("B2 / B3 CHECKS")
    print("=" * 74)
    unit_checks(failures)
    end_to_end(failures)

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  [!] {f}")
        print("=" * 74)
        return 1
    print("PASSED -- B2 and B3 train, evaluate and benchmark on fake data,")
    print("and the uncertainty metrics agree with their analytic values.")
    print(f"Outputs are in {SANDBOX}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
