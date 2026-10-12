"""Step 2 -- run Baseline B0 (copy z, scaled by the median train z-y colour).

    python scripts/02_run_b0.py

Fits the single constant on the train split, settles the sign of the formula
against the images themselves, and evaluates on val and test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.data.dataset import GalaxyDataset, make_dataloader
from src.data.preprocess import denormalize, load_global_stats, normalize
from src.evaluation.evaluator import collect_raw_images, evaluate_arrays
from src.evaluation.photometry import circular_aperture_mask
from src.models.b0_copy_z import CopyZBaseline, verify_scale_sign

#: Make +/- and arrows printable on a cp1252 Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline B0.")
    p.add_argument("--config", default=None)
    p.add_argument("--sign-check-galaxies", type=int, default=4000,
                   help="train galaxies read for the empirical sign check")
    return p.parse_args()


def check_roundtrip(dataset: GalaxyDataset, stats: dict) -> dict:
    """normalise -> denormalise must return the original fluxes."""
    from src.config import INPUT_BAND_IDX, TARGET_BAND_IDX

    raw = np.stack([dataset[i]["input_raw"].numpy() for i in range(min(8, len(dataset)))])
    back = denormalize(normalize(raw, stats, INPUT_BAND_IDX), stats, INPUT_BAND_IDX)
    scale = max(float(np.nanmax(np.abs(raw))), 1e-12)
    err = float(np.nanmax(np.abs(raw - back)) / scale)
    if err >= 1e-4:
        raise AssertionError(
            f"normalise/denormalise round-trip failed (max relative error {err:.2e}). "
            "Fix this before trusting any metric."
        )
    return {"max_relative_error": err, "passed": True}


def empirical_sign_check(cfg, train_ds, model, n_galaxies: int) -> dict:
    """Measure the true median y/z aperture flux ratio on train galaxies."""
    n = min(n_galaxies, len(train_ds))
    idx = np.linspace(0, len(train_ds) - 1, n).astype(int)

    z_imgs = np.empty((n, *train_ds.image_shape[1:]), dtype=np.float32)
    y_imgs = np.empty_like(z_imgs)
    for j, i in enumerate(idx):
        item = train_ds[int(i)]
        z_imgs[j] = item["input_raw"].numpy()[3]
        y_imgs[j] = item["target_raw"].numpy()[0]

    size = train_ds.image_shape[-1]
    ap = circular_aperture_mask(
        (size, size),
        cfg.aperture_center(size),
        float(cfg.dataset["aperture_radius"]),
    )
    return verify_scale_sign(z_imgs, y_imgs, model.median_colour, ap)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    missing = cfg.missing_processed_files()
    if missing:
        print("Processed files are missing:")
        for p in missing:
            print(f"  {p}")
        print("\nRun: python scripts/01_prepare_data.py")
        return 2

    stats = load_global_stats(cfg.stats_file())
    sign = int(cfg.b0["scaling"]["sign"])

    print("=" * 74)
    print("BASELINE B0 -- copy z, scaled by the median train z-y colour")
    print("=" * 74)

    train_ds = GalaxyDataset(cfg.processed_file("train"), stats, is_train=True)
    print(f"train galaxies: {len(train_ds):,}")

    rt = check_roundtrip(train_ds, stats)
    print(f"normalise->denormalise round-trip: OK "
          f"(max relative error {rt['max_relative_error']:.2e})")

    # -- fit the single constant, train split only -------------------------
    model = CopyZBaseline.fit(train_ds, sign=sign)
    print(f"\nmedian(m_z - m_y) on train : {model.median_colour:+.4f} mag "
          f"({model.n_used:,} galaxies, {model.n_dropped} dropped for bad mags)")
    print(f"scale = 10**({sign:+d} * 0.4 * median) = {model.scale:.6f}")

    # -- which sign does the data support? ---------------------------------
    print(f"\n--- WHICH SIGN IS RIGHT? (train images, aperture fluxes) ---")
    check = empirical_sign_check(cfg, train_ds, model, args.sign_check_galaxies)
    print(f"  measured median(F_y / F_z)     = "
          f"{check['empirical_median_flux_ratio_y_over_z']:.6f}")
    print(f"  10**(+0.4 * median(m_z - m_y)) = {check['scale_plus_sign']:.6f}"
          f"   |error| = {check['abs_error_plus']:.6f}")
    print(f"  10**(-0.4 * median(m_z - m_y)) = {check['scale_minus_sign']:.6f}"
          f"   |error| = {check['abs_error_minus']:.6f}")
    print(f"  --> the data matches the {check['matches'].upper()} sign "
          f"({check['n_galaxies_used']:,} galaxies)")
    if check["matches"] == "plus" and sign == -1:
        print("  [!] The config is using the MINUS sign but the data says PLUS.")

    # -- evaluate ----------------------------------------------------------
    results: dict = {
        "model": model.to_dict(),
        "sign_verification": check,
        "roundtrip_check": rt,
        "splits": {},
    }
    variants = [("B0", model)]
    if bool(cfg.b0["scaling"].get("evaluate_both_signs", False)):
        other = CopyZBaseline.fit(train_ds, sign=-sign)
        variants.append((f"B0_sign{-sign:+d}", other))

    for split in cfg.b0["evaluation"]["splits"]:
        ds = GalaxyDataset(cfg.processed_file(split), stats, is_train=False)
        loader = make_dataloader(
            ds, batch_size=int(cfg.loader["batch_size"]), shuffle=False,
            num_workers=0, pin_memory=False,
        )
        print(f"\n--- evaluating on {split} ({len(ds):,} galaxies) ---")
        raw = collect_raw_images(loader)

        for name, m in variants:
            y_pred = raw["z_true"] * m.scale
            res = evaluate_arrays(
                cfg, raw["y_true"], y_pred, raw["z_true"],
                return_arrays=(name == "B0"),
            )
            arrays = res.pop("arrays", None)
            results["splits"].setdefault(split, {})[name] = res

            if name == "B0":
                print_metrics(res)
                if arrays is not None and split == "test":
                    np.savez(
                        cfg.path("results_dir") / "arrays_b0_test.npz",
                        **arrays, object_id=raw["object_id"],
                    )
            else:
                dm = res["photometry"]["delta_mag"]
                print(f"  [{name}, the spec's sign] delta mag median "
                      f"{dm['median']:+.4f}, NMAD {dm['nmad']:.4f} "
                      f"-- for comparison")
        ds.close()

    out = cfg.path("results_dir") / "b0_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    train_ds.close()

    print(f"\nWrote {out}")
    print("Next:  python scripts/03_train_b1.py")
    return 0


def print_metrics(res: dict) -> None:
    p, ph = res["pixel"], res["photometry"]
    print(f"  MAE               : {p['mae']:.6g}")
    print(f"  RMSE              : {p['rmse']:.6g}")
    print(f"  PSNR (dB)         : {p['psnr_mean']:.3f} +/- {p['psnr_std']:.3f}")
    print(f"  SSIM              : {p['ssim_mean']:.4f} +/- {p['ssim_std']:.4f}")
    rf = ph["relative_flux_error"]
    print(f"  rel. flux error   : median {rf['median']:+.4f}   NMAD {rf['nmad']:.4f}")
    dm = ph["delta_mag"]
    print(f"  delta mag         : median {dm['median']:+.4f}   NMAD {dm['nmad']:.4f}"
          f"   (n={dm['n']:,}, excluded {dm['n_excluded_total']})")
    print(f"  outliers |dm|>0.5 : {dm['outlier_percent']:.2f} %")
    ce = ph["colour_error_z_minus_y"]
    print(f"  colour err d(z-y) : median {ce['median']:+.4f}   NMAD {ce['nmad']:.4f}")
    bs = ph.get("delta_mag_by_brightness") or {}
    if bs.get("faint_half"):
        print(f"  delta mag by flux : faint half NMAD {bs['faint_half']['nmad']:.4f}"
              f"   bright half NMAD {bs['bright_half']['nmad']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
