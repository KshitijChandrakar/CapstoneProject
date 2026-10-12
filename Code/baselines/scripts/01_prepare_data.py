"""Step 1 -- audit the raw files, cut them to the redshift window, fit the stats.

    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --overwrite     # rebuild the subsets

Checks the three official GalaxiesML files are present, filters each to
0.0 <= z < 0.4 with the homogeneous flag, verifies the counts against the
spec, asserts the splits are disjoint, and computes the global normalisation
statistics from the TRAIN split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import (
    BANDS,
    EDA_TRAIN_MEAN_MAG,
    KNOWN_RAW_SPLIT_SIZES,
    load_config,
)
from src.data.preprocess import (
    MAG_KEYS,
    compute_and_save_global_stats,
    filter_redshift_subset,
    mag_is_valid,
    verify_split_disjoint,
)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the GalaxiesML subsets.")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild the filtered subsets even if they exist")
    p.add_argument("--config", default=None, help="path to the config directory")
    return p.parse_args()


def audit_raw_files(cfg) -> list[str]:
    """Report what is in each raw file before touching it."""
    problems: list[str] = []
    print("=" * 74)
    print("RAW FILE AUDIT")
    print("=" * 74)

    for split in SPLITS:
        path = cfg.raw_file(split)
        print(f"\n--- {split}: {path.name} ---")
        if not path.exists():
            print(f"  MISSING: {path}")
            problems.append(f"{split}: file not found at {path}")
            continue

        print(f"  size: {path.stat().st_size / 1e9:.2f} GB")
        with h5py.File(path, "r") as f:
            if "image" not in f:
                problems.append(f"{split}: no 'image' dataset")
                continue
            img = f["image"]
            n = int(img.shape[0])
            print(f"  image      : {img.shape}  {img.dtype}")
            print(f"  datasets   : {len(f.keys())}")

            expected = KNOWN_RAW_SPLIT_SIZES.get(split)
            if expected is not None and n != expected:
                print(f"  [!] expected {expected:,} galaxies, found {n:,}")
                problems.append(
                    f"{split}: has {n:,} galaxies, the official split has {expected:,}"
                )
            else:
                print(f"  galaxies   : {n:,}  (matches the official split)")

            if img.shape[1] != len(BANDS):
                problems.append(f"{split}: expected 5 bands, found {img.shape[1]}")

            for key in ("specz_redshift",) + MAG_KEYS:
                if key not in f:
                    problems.append(f"{split}: missing '{key}'")

            if "specz_redshift" in f:
                z = np.asarray(f["specz_redshift"][:], dtype=np.float64)
                print(f"  redshift   : {z.min():.4f} to {z.max():.4f} "
                      f"(median {np.median(z):.4f})")

            if "specz_flag_homogeneous" in f:
                flag = np.asarray(f["specz_flag_homogeneous"][:]).astype(bool)
                print(f"  secure flag: {int(flag.sum()):,} of {n:,} True")

    return problems


def check_subset(cfg, split: str) -> list[str]:
    """Validate a filtered subset: counts, redshifts, pixels, magnitudes."""
    problems: list[str] = []
    path = cfg.processed_file(split)
    zmin, zmax = cfg.dataset["z_min"], cfg.dataset["z_max"]

    with h5py.File(path, "r") as f:
        n = int(f["image"].shape[0])
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)

        expected = cfg.dataset["expected_counts"].get(split)
        tol = float(cfg.dataset["count_tolerance"])
        line = f"  {split:<6}: {n:,} galaxies"
        if expected is not None:
            delta = abs(n - expected) / max(expected, 1)
            line += f"   (spec says {expected:,}"
            line += "  MATCH)" if n == expected else f", differs by {n - expected:+,})"
            if delta > tol:
                problems.append(
                    f"{split}: kept {n:,} galaxies but the spec expects {expected:,}"
                )
        print(line)

        if z.min() < zmin or z.max() >= zmax:
            problems.append(
                f"{split}: redshifts run {z.min():.4f}-{z.max():.4f}, "
                f"outside [{zmin}, {zmax})"
            )

        ids = np.asarray(f["object_id"][:])
        if np.unique(ids).size != n:
            problems.append(f"{split}: object_id is not unique")

        # Pixel sanity on a sample -- a full scan of 7 GB is not worth it here.
        sample = f["image"][: min(500, n)]
        n_nan = int(np.count_nonzero(np.isnan(sample)))
        n_inf = int(np.count_nonzero(np.isinf(sample)))
        if n_nan or n_inf:
            problems.append(
                f"{split}: {n_nan} NaN / {n_inf} inf pixels in the first 500 "
                "galaxies -- they will be masked out of the loss and metrics"
            )

        for band in BANDS:
            key = f"{band}_cmodel_mag"
            if key in f:
                m = np.asarray(f[key][:], dtype=np.float64)
                bad = int((~mag_is_valid(m)).sum())
                if bad:
                    problems.append(f"{split}: {bad} invalid {key} values")

    return problems


def report_colour(cfg) -> list[str]:
    """Print the median z-y colour, which fixes the sign of B0's scale."""
    problems: list[str] = []
    with h5py.File(cfg.processed_file("train"), "r") as f:
        zm = np.asarray(f["z_cmodel_mag"][:], dtype=np.float64)
        ym = np.asarray(f["y_cmodel_mag"][:], dtype=np.float64)
    ok = mag_is_valid(zm) & mag_is_valid(ym)
    median = float(np.median(zm[ok] - ym[ok]))
    eda = EDA_TRAIN_MEAN_MAG["z"] - EDA_TRAIN_MEAN_MAG["y"]

    print(f"\n  median(m_z - m_y) on the train subset : {median:+.4f} mag")
    print(f"  the team's EDA (full train split)     : {eda:+.4f} mag (mean)")
    print(f"  -> B0 scale = 10**(+0.4 * {median:.4f}) = {10 ** (0.4 * median):.4f}")
    print(f"     the spec's minus sign would give     {10 ** (-0.4 * median):.4f}")

    if median <= 0:
        problems.append(
            f"median(z - y) = {median:+.4f} is not positive, but the EDA gives "
            f"{eda:+.4f} (y brighter than z). Check the image band order."
        )
    return problems


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    zmin, zmax = cfg.dataset["z_min"], cfg.dataset["z_max"]

    problems = audit_raw_files(cfg)
    missing = cfg.missing_raw_files()
    if missing:
        print("\n" + "=" * 74)
        print("CANNOT CONTINUE -- raw files are missing:")
        for p in missing:
            print(f"  {p}")
        print("\nDownload them from https://zenodo.org/records/13878122 and put "
              f"them in {cfg.path('raw_dir')}")
        print("(Or run tests/test_pipeline_fake.py to exercise the code without data.)")
        print("=" * 74)
        return 2

    # -- filter ------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"FILTERING TO {zmin} <= z < {zmax}"
          f"{' AND specz_flag_homogeneous == True' if cfg.dataset['require_homogeneous_flag'] else ''}")
    print("=" * 74)

    summary: dict[str, dict] = {}
    for split in SPLITS:
        print(f"\n--- {split} ---")
        summary[split] = filter_redshift_subset(
            input_h5=cfg.raw_file(split),
            output_h5=cfg.processed_file(split),
            z_min=zmin,
            z_max=zmax,
            require_flag=bool(cfg.dataset["require_homogeneous_flag"]),
            chunk_size=int(cfg.norm["chunk_size"]),
            overwrite=args.overwrite,
        )

    # -- verify ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERIFICATION")
    print("=" * 74)
    for split in SPLITS:
        problems += check_subset(cfg, split)

    disjoint = verify_split_disjoint({s: cfg.processed_file(s) for s in SPLITS})
    print(f"\n  split overlap: {disjoint['overlaps']}  (all must be 0)")

    problems += report_colour(cfg)

    # -- statistics --------------------------------------------------------
    print("\n" + "=" * 74)
    print("GLOBAL NORMALISATION STATISTICS (train split only)")
    print("=" * 74)
    stats = compute_and_save_global_stats(
        train_h5_path=cfg.processed_file("train"),
        output_json_path=cfg.stats_file(),
        chunk_size=int(cfg.norm["chunk_size"]),
        eps=cfg.eps,
    )

    report = {
        "redshift_window": [zmin, zmax],
        "require_homogeneous_flag": cfg.dataset["require_homogeneous_flag"],
        "filtering": summary,
        "split_disjoint": disjoint,
        "global_stats": stats,
        "problems": problems,
    }
    out = cfg.path("splits_dir") / "prepare_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 74)
    if problems:
        print(f"FINISHED WITH {len(problems)} THING(S) TO CHECK:")
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("ALL CHECKS PASSED. Counts, redshifts, splits and magnitudes "
              "are as expected.")
    print(f"Report -> {out}")
    print("Next:  python scripts/02_run_b0.py")
    print("=" * 74)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
