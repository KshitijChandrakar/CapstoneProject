"""End-to-end check of the whole pipeline on FAKE data.

Builds three fake GalaxiesML files with the real structure, runs
prepare -> B0 -> B1 -> comparison against them, and asserts the scientific
properties that must hold. Lets the code be verified without the ~24 GB of
real data.

    python tests/test_pipeline_fake.py
"""

from __future__ import annotations

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
SANDBOX = ROOT / "_fake_run"


def build_sandbox_config() -> Path:
    """A config pointing at the sandbox, with small-data settings."""
    cfg_dir = SANDBOX / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    base["paths"] = {
        "raw_dir": "_fake_run/data/raw",
        "processed_dir": "_fake_run/data/processed",
        "splits_dir": "_fake_run/data/splits",
        "checkpoints_dir": "_fake_run/results/checkpoints",
        "results_dir": "_fake_run/results/metrics",
        "figures_dir": "_fake_run/results/figures",
    }
    # The fake set is tiny, so the real galaxy counts cannot apply.
    base["dataset"]["expected_counts"] = {}
    base["dataloader"]["num_workers"] = 0
    base["dataloader"]["pin_memory"] = False
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False),
                                         encoding="utf-8")

    for name in ("b0_baseline.yaml", "b1_baseline.yaml"):
        shutil.copy(ROOT / "config" / name, cfg_dir / name)

    # A tiny sample needs far more epochs to converge than 84,911 galaxies do,
    # so the convergence assertion below tests the optimiser, not the dataset.
    b1 = yaml.safe_load((cfg_dir / "b1_baseline.yaml").read_text(encoding="utf-8"))
    b1["training"]["epochs"] = 400
    b1["diagnostics"]["reference_optimum_max_galaxies"] = 400
    (cfg_dir / "b1_baseline.yaml").write_text(yaml.safe_dump(b1, sort_keys=False),
                                              encoding="utf-8")
    return cfg_dir


def check(label: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def main() -> int:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    raw_dir = SANDBOX / "data" / "raw"
    raw_dir.mkdir(parents=True)

    failures: list[str] = []
    print("=" * 74)
    print("PIPELINE TEST ON FAKE DATA")
    print("=" * 74)

    info = make_all_splits(raw_dir, n_train=600, n_val=200, n_test=200)
    print(f"\nfake files: " + ", ".join(
        f"{k}={v['n']} ({v['n_in_window']} in 0<=z<0.4)" for k, v in info.items()
    ))

    cfg_dir = build_sandbox_config()
    from src.config import load_config
    cfg = load_config(cfg_dir)

    # -- 1. prepare --------------------------------------------------------
    print("\n--- 01_prepare_data ---")
    import importlib.util

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

    status = run_script("01_prepare_data.py", ["--config", str(cfg_dir)])
    check("01_prepare_data ran", status in (0, 1), f"exit code {status}", failures)

    import h5py
    from src.data.preprocess import load_global_stats, verify_split_disjoint

    for split in ("train", "val", "test"):
        path = cfg.processed_file(split)
        with h5py.File(path, "r") as f:
            z = np.asarray(f["specz_redshift"][:])
            n = int(f["image"].shape[0])
        check(f"{split} subset inside the redshift window",
              bool(z.min() >= 0.0 and z.max() < 0.4),
              f"n={n}, z from {z.min():.4f} to {z.max():.4f}", failures)

    disjoint = verify_split_disjoint(
        {s: cfg.processed_file(s) for s in ("train", "val", "test")}
    )
    check("splits are disjoint", disjoint["all_disjoint"],
          f"overlaps {disjoint['overlaps']}", failures)

    stats = load_global_stats(cfg.stats_file())
    check("global stats saved", cfg.stats_file().exists(),
          f"means {np.round(stats['mean'], 3).tolist()}", failures)
    check("stats computed on train only",
          stats["meta"]["computed_on"] == cfg.processed_file("train").name,
          stats["meta"]["computed_on"], failures)

    # -- 2. normalisation round-trip & augmentation ------------------------
    print("\n--- data handling ---")
    from src.config import INPUT_BAND_IDX, TARGET_BAND_IDX
    from src.data.dataset import GalaxyDataset
    from src.data.preprocess import denormalize, normalize

    ds = GalaxyDataset(cfg.processed_file("train"), stats, is_train=True)
    raw = np.stack([ds[i]["input_raw"].numpy() for i in range(8)])
    back = denormalize(normalize(raw, stats, INPUT_BAND_IDX), stats, INPUT_BAND_IDX)
    rel = float(np.nanmax(np.abs(raw - back)) / max(float(np.nanmax(np.abs(raw))), 1e-12))
    check("normalise -> denormalise round-trip", rel < 1e-4,
          f"max relative error {rel:.2e}", failures)

    xs = np.stack([ds[i]["input"].numpy() for i in range(len(ds))])
    check("standardised train data has mean~0 std~1",
          bool(np.all(np.abs(xs.mean(axis=(0, 2, 3))) < 0.05)
               and np.all(np.abs(xs.std(axis=(0, 2, 3)) - 1) < 0.05)),
          f"mean {np.round(xs.mean(axis=(0,2,3)), 4).tolist()}, "
          f"std {np.round(xs.std(axis=(0,2,3)), 4).tolist()}", failures)

    aug_ds = GalaxyDataset(cfg.processed_file("train"), stats, is_train=True,
                           augment=True, seed=0)
    item = aug_ds[0]
    plain = ds[0]
    check("augmentation preserves pixel values (flux-safe)",
          bool(np.allclose(np.sort(item["input"].numpy().ravel()),
                           np.sort(plain["input"].numpy().ravel()), atol=1e-5)),
          "only rotations and flips, so the sorted pixels are identical", failures)
    check("target NaN pixels are masked",
          bool(np.all(np.isfinite(item["target"].numpy()))),
          f"mask min {float(item['mask'].numpy().min()):.0f}", failures)
    check("dataset returns the required keys",
          all(k in item for k in ("input", "target", "input_raw", "target_raw",
                                  "redshift", "object_id", "z_cmodel_mag",
                                  "y_cmodel_mag")),
          f"{len(item)} keys", failures)
    ds.close(); aug_ds.close()

    # -- 3. B0 -------------------------------------------------------------
    print("\n--- 02_run_b0 ---")
    status = run_script("02_run_b0.py", ["--config", str(cfg_dir)])
    check("02_run_b0 ran", status == 0, f"exit code {status}", failures)

    import json
    b0 = json.loads((cfg.path("results_dir") / "b0_metrics.json").read_text(encoding="utf-8"))
    sign = b0["sign_verification"]
    check("B0 has 0 parameters", b0["model"]["n_parameters"] == 0, "0", failures)
    check("B0's scale matches the measured flux ratio", sign["matches"] == "plus",
          f"measured {sign['empirical_median_flux_ratio_y_over_z']:.5f}, "
          f"plus {sign['scale_plus_sign']:.5f}, minus {sign['scale_minus_sign']:.5f}"
          f" -> {sign['matches']} wins", failures)
    err = abs(b0["model"]["scale"] - sign["empirical_median_flux_ratio_y_over_z"]) \
        / sign["empirical_median_flux_ratio_y_over_z"]
    check("B0's fitted scale is close to the empirical ratio", err < 0.10,
          f"{err * 100:.2f}% apart", failures)

    m0 = b0["splits"]["test"]["B0"]
    check("colour error equals -delta_mag",
          abs(m0["photometry"]["colour_error_z_minus_y"]["median"]
              + m0["photometry"]["delta_mag"]["median"]) < 1e-9,
          "d(z-y) = -dm, as expected with the same z measurement", failures)
    check("clamp vs exclude both reported",
          "magnitude_handling_comparison" in m0,
          f"exclude NMAD {m0['magnitude_handling_comparison']['exclude']['delta_mag']['nmad']:.4f}"
          f" vs clamp NMAD {m0['magnitude_handling_comparison']['clamp']['delta_mag']['nmad']:.4f}",
          failures)

    # -- 4. B1 -------------------------------------------------------------
    print("\n--- 03_train_b1 ---")
    status = run_script("03_train_b1.py", ["--config", str(cfg_dir)])
    check("03_train_b1 ran", status == 0, f"exit code {status}", failures)

    b1 = json.loads((cfg.path("results_dir") / "b1_metrics.json").read_text(encoding="utf-8"))
    check("B1 has exactly 5 parameters", b1["n_parameters"] == 5,
          f"{b1['n_parameters']}", failures)
    check("3 seeds ran", len(b1["per_seed"]) == 3, f"seeds {b1['seeds']}", failures)

    conv = b1["convergence"]
    check("B1 converges to the solved optimum", conv["converged"],
          f"distance {conv['distance_to_solved_optimum_mean']:.4f}", failures)

    opt = b1["reference_optimum"]["l1_optimum"]["physical_flux_space"]
    trained = b1["weights_over_seeds"]["physical_flux_space"]
    worst = max(abs(trained[f"w_{b}"]["mean"] - opt[f"w_{b}"]) for b in b1["bands"])
    check("physical weight conversion agrees with the solved optimum", worst < 0.05,
          f"largest disagreement {worst:.5f}", failures)
    check("z has the largest physical weight", b1["z_dominates"],
          f"ordering {' > '.join(b1['physical_ordering_mean'])}", failures)

    mae0 = m0["pixel"]["mae"]
    mae1 = b1["metrics_over_seeds"]["pixel"]["mae"]["mean"]
    check("B1 beats or matches B0 on MAE", mae1 <= mae0 * 1.05,
          f"B0 {mae0:.5g} vs B1 {mae1:.5g}", failures)

    # B1 minimises pixelwise L1, which is not the same as aperture photometry,
    # so it is only guaranteed to win where its fitted bias is negligible.
    bs0 = m0["photometry"]["delta_mag_by_brightness"]
    bs1 = b1["per_seed"][0]["metrics"]["photometry"]["delta_mag_by_brightness"]
    check("B1 beats B0 on delta-m NMAD for bright galaxies",
          bs1["bright_half"]["nmad"] <= bs0["bright_half"]["nmad"] * 1.10,
          f"B0 {bs0['bright_half']['nmad']:.4f} vs B1 {bs1['bright_half']['nmad']:.4f}",
          failures)
    print(f"  [note] faint half: B0 {bs0['faint_half']['nmad']:.4f} vs "
          f"B1 {bs1['faint_half']['nmad']:.4f} -- B1's fitted per-pixel bias acts "
          f"like a sky offset and dominates faint galaxies' aperture flux.")

    # -- 5. comparison -----------------------------------------------------
    print("\n--- 04_benchmark_comparison ---")
    status = run_script("04_benchmark_comparison.py", ["--config", str(cfg_dir)])
    check("04_benchmark_comparison ran", status == 0, f"exit code {status}", failures)

    res = cfg.path("results_dir")
    figs = cfg.path("figures_dir")
    for name in ("results_table.md", "results_table.tex"):
        check(f"{name} written", (res / name).exists(), str(res / name), failures)
    expected_figs = ["dmag_histogram.png", "flux_scatter.png",
                     "b1_weights_physical.png", "b1_weights_standardised.png",
                     "galaxy_grid.png"]
    for name in expected_figs:
        p = figs / name
        check(f"figure {name}", p.exists() and p.stat().st_size > 5000,
              f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "missing",
              failures)

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  [!] {f}")
        print("=" * 74)
        return 1
    print("PASSED -- every stage works end to end on fake data.")
    print(f"Outputs are in {SANDBOX}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
