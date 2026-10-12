"""Step 1 (pilot) -- build train/val/test from ONE HDF5 file.

Use this when the three official GalaxiesML files are not available yet and all
you have is a single extracted sample. It applies the configured redshift
window, splits the galaxies 70/15/15 with a fixed seed, and writes the same
processed files and global_stats.json that 01_prepare_data.py produces -- so
scripts 02, 03 and 04 then run unchanged.

    python scripts/01b_prepare_pilot.py ../../../sample_z0.2_2000.h5

**This is a pilot, not the deliverable.** The official splits are the real
target: they are GalaxiesML's own train/val/test partition, and a re-split of
one file is not the same thing. Every output is tagged accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import BANDS, load_config
from src.data.preprocess import (
    CARRY_KEYS,
    compute_and_save_global_stats,
    mag_is_valid,
    verify_split_disjoint,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a pilot split from one file.")
    p.add_argument("source", help="the single HDF5 file to split")
    p.add_argument("--config", default=None)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--ignore-redshift-window", action="store_true",
                   help="keep every galaxy regardless of the configured window")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    src = Path(args.source)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    if not src.exists():
        print(f"ERROR: no such file: {src}")
        return 2

    fr = np.asarray(args.fractions, dtype=float)
    if abs(fr.sum() - 1.0) > 1e-9:
        print(f"ERROR: fractions must sum to 1.0, got {fr.sum():.4f}")
        return 2

    zmin, zmax = float(cfg.dataset["z_min"]), float(cfg.dataset["z_max"])

    print("=" * 74)
    print("PILOT SPLIT FROM A SINGLE FILE")
    print("=" * 74)
    print(f"source: {src}  ({src.stat().st_size / 1e6:.0f} MB)")

    with h5py.File(src, "r") as f:
        if "image" not in f:
            print(f"ERROR: {src.name} has no 'image' dataset.")
            return 2
        n_total = int(f["image"].shape[0])
        img_shape = tuple(int(v) for v in f["image"].shape[1:])
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)

        if args.ignore_redshift_window:
            keep = np.arange(n_total)
            print(f"redshift window ignored; keeping all {n_total:,} galaxies")
        else:
            keep = np.flatnonzero((z >= zmin) & (z < zmax))
            print(f"redshift window {zmin} <= z < {zmax}: "
                  f"{keep.size:,} of {n_total:,} galaxies kept")
        if keep.size == 0:
            print("ERROR: nothing survives the redshift window.")
            return 2

        n = keep.size
        rng = np.random.RandomState(args.split_seed)
        order = keep[rng.permutation(n)]
        n_tr = int(round(fr[0] * n))
        n_va = int(round(fr[1] * n))
        parts = {
            "train": np.sort(order[:n_tr]),
            "val": np.sort(order[n_tr:n_tr + n_va]),
            "test": np.sort(order[n_tr + n_va:]),
        }

        print(f"\nsplit (seed {args.split_seed}): " + "  ".join(
            f"{k} {v.size:,}" for k, v in parts.items()
        ))

        for split, idx in parts.items():
            if idx.size == 0:
                print(f"ERROR: the {split} split came out empty.")
                return 2
            out = cfg.processed_file(split)
            out.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out, "w") as g:
                dst = g.create_dataset(
                    "image", shape=(idx.size,) + img_shape, dtype=np.float32,
                    chunks=(min(32, idx.size),) + img_shape,
                )
                for start in range(0, idx.size, 200):
                    stop = min(start + 200, idx.size)
                    dst[start:stop] = f["image"][idx[start:stop]]
                for key in CARRY_KEYS:
                    if key in f:
                        g.create_dataset(key, data=np.asarray(f[key][:])[idx])
                g.attrs["source_file"] = src.name
                g.attrs["pilot_resplit"] = True
                g.attrs["split_seed"] = args.split_seed
                g.attrs["z_min"] = zmin
                g.attrs["z_max"] = zmax
                g.attrs["n_kept"] = int(idx.size)
            print(f"  wrote {out.name}  ({idx.size:,} galaxies, "
                  f"{out.stat().st_size / 1e6:.0f} MB)")

    # -- verify ------------------------------------------------------------
    print("\n--- verification ---")
    disjoint = verify_split_disjoint({s: cfg.processed_file(s) for s in parts})
    print(f"  split overlap: {disjoint['overlaps']}  (all must be 0)")

    with h5py.File(cfg.processed_file("train"), "r") as f:
        zm = np.asarray(f["z_cmodel_mag"][:], dtype=np.float64)
        ym = np.asarray(f["y_cmodel_mag"][:], dtype=np.float64)
    ok = mag_is_valid(zm) & mag_is_valid(ym)
    median = float(np.median(zm[ok] - ym[ok]))
    print(f"  median(m_z - m_y) on the pilot train split: {median:+.5f} mag")
    print(f"  -> B0 scale = 10**(+0.4 * {median:.5f}) = {10 ** (0.4 * median):.5f}")
    print(f"     the spec's minus sign would give          {10 ** (-0.4 * median):.5f}")

    # -- statistics --------------------------------------------------------
    print("\n--- global normalisation statistics (train split only) ---")
    stats = compute_and_save_global_stats(
        train_h5_path=cfg.processed_file("train"),
        output_json_path=cfg.stats_file(),
        chunk_size=int(cfg.norm["chunk_size"]),
        eps=cfg.eps,
    )

    report = {
        "mode": "pilot_resplit_from_single_file",
        "source_file": str(src),
        "warning": (
            "These splits are a 70/15/15 re-split of ONE file, not GalaxiesML's "
            "official train/val/test partition. Results are a pilot."
        ),
        "split_seed": args.split_seed,
        "fractions": fr.tolist(),
        "redshift_window": [zmin, zmax],
        "counts": {k: int(v.size) for k, v in parts.items()},
        "split_disjoint": disjoint,
        "median_z_minus_y_train": median,
        "global_stats": stats,
    }
    out = cfg.path("splits_dir") / "pilot_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 74)
    print("PILOT DATA READY.  NOTE: this is a re-split of one file, not the")
    print("official GalaxiesML train/val/test partition.")
    print(f"Report -> {out}")
    print("Next:  python scripts/02_run_b0.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
