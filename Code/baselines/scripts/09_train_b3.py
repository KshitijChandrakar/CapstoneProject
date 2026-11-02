"""Step 9 -- train Baseline B3 (U-Net + FiLM(redshift) + Gaussian NLL).

    python scripts/09_train_b3.py               # all seeds from config.yaml
    python scripts/09_train_b3.py --seed 42     # one seed
    python scripts/09_train_b3.py --no-film     # ablation: B3 without FiLM

B3 is the capacity-matched control for the physics models P1-P3. It gets the
same size, the same redshift conditioning and the same per-pixel error bars --
and no physics. If a physics model cannot beat this, it has not demonstrated
that physics helped.

Two things this script reports that no earlier script could:

* **calibration** -- whether the predicted sigma is the right size, which is
  the only thing that makes the NLL meaningful, and
* **anomaly candidates** -- the galaxies B3 got confidently wrong, which is
  the project's science application rather than a leaderboard number.

The run trains the mean on masked L1 for ``uncertainty.nll_warmup_epochs``
epochs before switching to the NLL. Without that, the variance head learns to
call everything noise and the mean never fits. The switch is visible in the
history CSV as a jump, and the best-validation record is reset there, because
the two losses are not on the same scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import GalaxyDataset, make_dataloader
from src.data.preprocess import load_global_stats
from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_with_uncertainty,
    summarise_over_seeds,
)
from src.models.b3_unet_film_nll import build_b3, make_b3_loss
from src.training.logger import ExperimentLogger
from src.training.trainer import Trainer, set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline B3 (U-Net + FiLM + NLL)."
    )
    p.add_argument("--seed", type=int, default=None,
                   help="train a single seed instead of all of them")
    p.add_argument("--config", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None,
                   help="epochs of L1 on the mean before the NLL starts")
    p.add_argument("--no-film", action="store_true",
                   help="ablation: train without redshift conditioning, so the "
                        "only difference from B2 is the NLL")
    p.add_argument("--tag", default="b3", help="output filename stem")
    p.add_argument("--label", default=None,
                   help="human-readable name for the report")
    return p.parse_args()


def apply_overrides(cfg, args) -> str:
    """Apply CLI overrides to the loaded B3 config; return the run label."""
    b3 = cfg.model_config("b3")
    if args.lr is not None:
        b3["training"]["lr"] = args.lr
    if args.epochs is not None:
        b3["training"]["epochs"] = args.epochs
    if args.warmup is not None:
        b3["uncertainty"]["nll_warmup_epochs"] = args.warmup
    if args.no_film:
        b3["film"]["enabled"] = False
        # The FiLM generator still exists but is never called, so its
        # parameters sit unused. Report the real trained size, not the built
        # one, and stop enforcing the prescribed count.
        b3["model"].pop("expected_parameters", None)
    if args.label:
        return args.label
    if args.no_film:
        return "B3 ablation (no FiLM)"
    return "B3 (U-Net+FiLM+NLL)" if args.tag == "b3" else args.tag


def train_one_seed(cfg, stats, seed: int, datasets: dict, tag: str = "b3") -> dict:
    """Train B3 with one seed and evaluate it on the test split."""
    set_seed(seed)
    device = cfg.device()
    b3 = cfg.model_config("b3")
    mcfg, tcfg = b3["model"], b3["training"]
    fcfg, ucfg = b3["film"], b3["uncertainty"]
    use_film = bool(fcfg.get("enabled", True))

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    if train_ds.augmenter:
        train_ds.augmenter.__init__(seed=seed)

    bs = int(cfg.loader["batch_size"])
    nw = int(cfg.loader["num_workers"])
    pin = bool(cfg.loader["pin_memory"]) and device != "cpu"

    train_loader = make_dataloader(train_ds, bs, True, nw, pin, seed=seed)
    val_loader = make_dataloader(val_ds, bs, False, nw, pin)
    test_loader = make_dataloader(test_ds, bs, False, nw, pin)

    model = build_b3(mcfg, fcfg, ucfg, seed=seed)
    breakdown = model.parameter_breakdown()
    warmup = int(ucfg["nll_warmup_epochs"])
    cond_key = "redshift" if use_film else None

    logger = ExperimentLogger(f"{tag}_seed{seed}", cfg.path("results_dir"))
    steps = len(train_loader)
    print(f"\n--- B3, seed {seed} ---")
    print(f"  {breakdown['total']:,} learnable parameters   device: {device}")
    print(f"    backbone {breakdown['backbone']:,}  +  FiLM generator "
          f"{breakdown['film_generator']:,}"
          f"{'  (built but UNUSED: --no-film)' if not use_film else ''}")
    print(f"  conditioning: {fcfg['conditioning'] if use_film else 'none'}   "
          f"fourier features {fcfg['fourier_features']}")
    print(f"  train {len(train_ds):,} / val {len(val_ds):,} galaxies")
    print(f"  lr {tcfg['lr']:g}, up to {tcfg['epochs']} epochs x {steps:,} steps")
    print(f"  loss: masked L1 on the mean for epochs 1-{warmup}, "
          f"then Gaussian NLL")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        epochs=int(tcfg["epochs"]),
        device=device,
        logger=logger,
        early_stopping_patience=tcfg.get("early_stopping_patience"),
        loss_fn=make_b3_loss(warmup),
        conditioning_key=cond_key,
        grad_clip_norm=tcfg.get("grad_clip_norm"),
    )
    fit = trainer.fit()
    print(f"  best val NLL {fit['best_val_loss']:.5f} at epoch "
          f"{fit['best_epoch']}/{fit['epochs_run']}"
          f"{'  (early stop)' if fit['stopped_early'] else ''}")

    ckpt = trainer.save_checkpoint(
        cfg.path("checkpoints_dir") / f"{tag}_seed{seed}.pt",
        extra={"seed": seed, "config": b3, "n_parameters": breakdown["total"]},
    )

    preds = collect_predictions(
        model, test_loader, stats, torch.device(device), conditioning_key=cond_key
    )
    if "y_sigma" not in preds:
        raise RuntimeError(
            "B3 returned no sigma. The model must return (mu, logvar)."
        )
    res = evaluate_with_uncertainty(
        cfg,
        preds["y_true"], preds["y_pred"], preds["z_true"], preds["y_sigma"],
        preds["object_id"],
        top_k=int(b3["evaluation"].get("anomaly_top_k", 50)),
        levels=tuple(ucfg.get("coverage_levels", (0.5, 0.68, 0.95))),
        return_arrays=True,
    )
    arrays = res.pop("arrays")
    np.savez(
        cfg.path("results_dir") / f"arrays_{tag}_seed{seed}_test.npz",
        **arrays,
        object_id=preds["object_id"],
        redshift=preds["redshift"],
        # Per-galaxy median sigma: enough to re-rank anomalies or plot
        # calibration later without keeping every pixel's sigma.
        sigma_median_per_galaxy=np.median(
            preds["y_sigma"].reshape(preds["y_sigma"].shape[0], -1), axis=1
        ),
    )

    # FiLM is only worth its parameters if it actually uses the redshift. This
    # measures that directly: re-predict a fixed batch at several redshifts and
    # see whether the output moves.
    film_response = measure_film_response(
        model, test_loader, torch.device(device)
    ) if use_film else None

    logger.close()

    return {
        "model": "B3_unet_film_nll",
        "seed": seed,
        "n_parameters": breakdown["total"],
        "parameter_breakdown": breakdown,
        "film_enabled": use_film,
        "architecture": {
            k: mcfg[k] for k in ("base_channels", "depth", "norm", "groups",
                                 "activation", "dropout", "residual_on_z")
            if k in mcfg
        },
        "film": dict(fcfg),
        "uncertainty_settings": dict(ucfg),
        "hyperparameters": {
            **{k: tcfg[k] for k in ("lr", "weight_decay", "epochs", "loss",
                                    "optimizer", "scheduler")},
            "batch_size": bs,
            "grad_clip_norm": tcfg.get("grad_clip_norm"),
            "early_stopping_patience": tcfg.get("early_stopping_patience"),
            "nll_warmup_epochs": warmup,
            "optimiser_steps_run": int(fit["epochs_run"]) * steps,
        },
        "best_val_loss": fit["best_val_loss"],
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "stopped_early": fit["stopped_early"],
        "objective_changed_at_epoch": fit["objective_changed_at_epoch"],
        "film_response": film_response,
        "checkpoint": str(ckpt),
        "history": fit["history"],
        "metrics": res,
    }


def measure_film_response(
    model, loader, device: torch.device, probes: tuple[float, ...] = (0.05, 0.2, 0.39)
) -> dict:
    """How much does the prediction change when only the redshift changes?

    One batch, predicted at several redshifts with the images held fixed. If
    this comes back at zero, FiLM learned nothing and B3 is B2 plus an
    uncertainty head -- which would be a real finding, not a bug, but it has to
    be stated rather than assumed either way.
    """
    model.eval()
    batch = next(iter(loader))
    x = batch["input"].to(device)
    outs = []
    with torch.no_grad():
        for z in probes:
            cond = torch.full((x.shape[0],), float(z), device=device)
            mu, _ = model(x, cond)
            outs.append(mu.cpu().numpy())
    stack = np.stack(outs, axis=0)
    spread = float(np.mean(np.std(stack, axis=0)))
    pixel_scale = float(np.mean(np.std(stack[0], axis=(1, 2, 3))))
    return {
        "probe_redshifts": list(probes),
        "mean_pixel_std_across_redshifts": spread,
        "mean_pixel_std_within_one_prediction": pixel_scale,
        "relative_response": float(spread / pixel_scale) if pixel_scale else float("nan"),
        "film_is_used": bool(spread > 1e-6),
        "note": (
            "Same images, different redshift. A relative response near zero "
            "means the conditioning is inert and B3's advantage over B2, if "
            "any, comes from the NLL rather than from redshift."
        ),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    label = apply_overrides(cfg, args)

    missing = cfg.missing_processed_files()
    if missing:
        print("Processed files are missing:")
        for p in missing:
            print(f"  {p}")
        print("\nRun: python scripts/01_prepare_data.py")
        return 2

    stats = load_global_stats(cfg.stats_file())
    seeds = [args.seed] if args.seed is not None else cfg.seeds

    print("=" * 74)
    print(f"{label} -- seeds {seeds}")
    print("=" * 74)

    datasets = {
        "train": GalaxyDataset(
            cfg.processed_file("train"), stats, is_train=True, augment=True,
            in_memory=cfg.use_in_memory(cfg.processed_file("train")),
        ),
        "val": GalaxyDataset(
            cfg.processed_file("val"), stats,
            in_memory=cfg.use_in_memory(cfg.processed_file("val")),
        ),
        "test": GalaxyDataset(
            cfg.processed_file("test"), stats,
            in_memory=cfg.use_in_memory(cfg.processed_file("test")),
        ),
    }
    print("images held in RAM: " + ", ".join(
        f"{k}={d.in_memory}" for k, d in datasets.items()
    ))

    per_seed = [train_one_seed(cfg, stats, s, datasets, args.tag) for s in seeds]
    for ds in datasets.values():
        ds.close()

    summary = summarise_over_seeds([r["metrics"] for r in per_seed])
    out = {
        "model": "B3_unet_film_nll",
        "label": label,
        "tag": args.tag,
        "n_parameters": per_seed[0]["n_parameters"],
        "parameter_breakdown": per_seed[0]["parameter_breakdown"],
        "film_enabled": per_seed[0]["film_enabled"],
        "seeds": seeds,
        "architecture": per_seed[0]["architecture"],
        "film": per_seed[0]["film"],
        "uncertainty_settings": per_seed[0]["uncertainty_settings"],
        "hyperparameters": per_seed[0]["hyperparameters"],
        "film_response_per_seed": [r["film_response"] for r in per_seed],
        "metrics_over_seeds": summary,
        # The anomaly shortlist is per seed by construction; the first seed's is
        # the reference one, and script 10 reports how much they overlap.
        "anomaly_candidates_seed0": per_seed[0]["metrics"]["anomaly_candidates"],
        "per_seed": per_seed,
    }
    path = cfg.path("results_dir") / f"{args.tag}_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(out)
    print(f"\nWrote {path}")
    print("Next:  python scripts/10_benchmark_all.py")
    return 0


def print_summary(out: dict) -> None:
    s = out["metrics_over_seeds"]
    pix, pho = s["pixel"], s["photometry"]
    print("\n" + "-" * 74)
    print("B3 TEST METRICS, mean +/- std over seeds (physical flux units)")
    print("-" * 74)
    bd = out["parameter_breakdown"]
    print(f"  parameters        : {bd['total']:,}  "
          f"(backbone {bd['backbone']:,} + FiLM {bd['film_generator']:,})")
    print(f"  MAE               : {pix['mae']['mean']:.6g} +/- "
          f"{pix['mae']['std']:.3g}")
    print(f"  PSNR (dB)         : {pix['psnr_mean']['mean']:.3f} +/- "
          f"{pix['psnr_mean']['std']:.3f}")
    print(f"  SSIM              : {pix['ssim_mean']['mean']:.4f} +/- "
          f"{pix['ssim_mean']['std']:.4f}")
    dm = pho["delta_mag"]
    print(f"  delta mag         : median {dm['median']['mean']:+.4f}   "
          f"NMAD {dm['nmad']['mean']:.4f}")
    print(f"  outliers |dm|>0.5 : {dm['outlier_percent']['mean']:.2f} %")

    unc = s.get("uncertainty", {})
    if unc:
        nll, cal, sharp = unc["nll"], unc["calibration"], unc["sharpness"]
        print("\n  UNCERTAINTY (the part only B3 can be asked)")
        print(f"    NLL per pixel   : {nll['nll']['mean']:.5f} +/- "
              f"{nll['nll']['std']:.3g}")
        print(f"      = log-sigma {nll['log_sigma_term']['mean']:+.4f} "
              f"+ quadratic {nll['quadratic_term']['mean']:.4f} + const")
        print(f"    calibration     : (y-mu)/sigma spread "
              f"{cal['ratio_std']['mean']:.3f}  (1.0 = honest)")
        for level in ("0.50", "0.68", "0.95"):
            node = cal.get("coverage", {}).get(level)
            if node:
                print(f"      {level} interval covers "
                      f"{node['empirical']['mean'] * 100:.1f} % of pixels")
        print(f"    sharpness       : median sigma "
              f"{sharp['sharpness_median_sigma']['mean']:.4g}")
        print(f"    sigma vs |error|: Spearman "
              f"{sharp['spearman_sigma_vs_abs_error']['mean']:+.3f}  "
              f"(0 = sigma says nothing about where the error is)")

    v = out["per_seed"][0]["metrics"].get("uncertainty_verdict", {})
    if v:
        print(f"\n    verdict (seed {out['seeds'][0]}): {v['calibration_state']}; "
              f"knows where it is uncertain: "
              f"{v['knows_where_it_is_uncertain']}")

    responses = [r for r in out.get("film_response_per_seed", []) if r]
    if responses:
        rel = float(np.mean([r["relative_response"] for r in responses]))
        used = all(r["film_is_used"] for r in responses)
        print(f"\n  FiLM response     : relative {rel:.4f}   used: {used}")
        if rel < 1e-3:
            print("    [!] The conditioning barely moves the prediction. Any "
                  "gain over B2 is from the NLL, not from redshift.")

    anom = out.get("anomaly_candidates_seed0", {})
    if anom:
        print(f"\n  anomaly candidates: top {anom['top_k']} by NLL, median "
              f"galaxy NLL {anom['median_nll_all_galaxies']:.4f}")
        print(f"    worst object_id {anom['object_id'][0]} "
              f"(NLL {anom['nll'][0]:.3f})")

    print("\n  Full ladder table: python scripts/10_benchmark_all.py")


if __name__ == "__main__":
    sys.exit(main())
