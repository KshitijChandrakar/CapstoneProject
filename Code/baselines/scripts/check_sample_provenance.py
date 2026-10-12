"""Work out which official GalaxiesML file(s) a sample was drawn from.

Run this on the machine that holds the three official files. It matches the
sample's object_ids against each split and reports where each galaxy came from,
which settles whether a sample mixes GalaxiesML's train/val/test partition.

    python scripts/check_sample_provenance.py sample_z0.2_2000.h5 \
        --raw-dir path/to/folder/with/the/three/files

Only numpy and h5py are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_FILES = {
    "train": "5x64x64_training_with_morphology.hdf5",
    "val": "5x64x64_validation_with_morphology.hdf5",
    "test": "5x64x64_testing_with_morphology.hdf5",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find which official split(s) a sample's galaxies came from."
    )
    p.add_argument("sample", help="the extracted sample .h5")
    p.add_argument("--raw-dir", default=".",
                   help="folder holding the three official HDF5 files")
    p.add_argument("--train", default=DEFAULT_FILES["train"])
    p.add_argument("--val", default=DEFAULT_FILES["val"])
    p.add_argument("--test", default=DEFAULT_FILES["test"])
    p.add_argument("--out", default="sample_provenance.json")
    return p.parse_args()


def read_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "object_id" not in f:
            raise KeyError(f"{path.name} has no 'object_id' dataset.")
        return np.asarray(f["object_id"][:]).astype(np.int64, copy=False)


def main() -> int:
    args = parse_args()
    sample_path = Path(args.sample)
    raw_dir = Path(args.raw_dir)

    if not sample_path.exists():
        print(f"ERROR: no such sample file: {sample_path}")
        return 2

    sample_ids = read_ids(sample_path)
    n = sample_ids.size
    print(f"sample: {sample_path.name}  ({n:,} galaxies, "
          f"{np.unique(sample_ids).size:,} unique)")

    splits = {"train": args.train, "val": args.val, "test": args.test}
    found: dict[str, np.ndarray] = {}
    for split, name in splits.items():
        path = raw_dir / name
        if not path.exists():
            print(f"  [!] missing: {path}")
            continue
        ids = read_ids(path)
        found[split] = ids
        print(f"  {split:<6}: {path.name}  ({ids.size:,} galaxies)")

    if not found:
        print("\nERROR: none of the three official files were found. "
              "Point --raw-dir at the folder that holds them.")
        return 2

    print(f"\n{'split':<8}{'matched':>10}{'% of sample':>14}")
    print("-" * 32)
    membership: dict[str, np.ndarray] = {}
    report: dict = {"sample": str(sample_path), "n_sample": int(n), "splits": {}}

    for split, ids in found.items():
        hit = np.isin(sample_ids, ids)
        membership[split] = hit
        pct = 100.0 * hit.sum() / n
        print(f"{split:<8}{int(hit.sum()):>10,}{pct:>13.1f}%")
        report["splits"][split] = {
            "n_matched": int(hit.sum()), "percent": float(pct),
            "n_in_split": int(ids.size),
        }

    stacked = np.vstack([membership[s] for s in membership])
    n_matched_any = int(np.count_nonzero(stacked.any(axis=0)))
    n_unmatched = n - n_matched_any
    n_multi = int(np.count_nonzero(stacked.sum(axis=0) > 1))

    print(f"\nmatched in at least one split : {n_matched_any:,} of {n:,}")
    print(f"matched in more than one      : {n_multi:,}  (should be 0)")
    print(f"matched in none               : {n_unmatched:,}")

    report.update({
        "n_matched_any": n_matched_any,
        "n_matched_multiple": n_multi,
        "n_unmatched": n_unmatched,
    })

    # -- verdict -----------------------------------------------------------
    used = [s for s in membership if membership[s].sum() > 0]
    print("\n" + "=" * 60)
    if n_unmatched:
        print(f"[!] {n_unmatched:,} galaxies matched NO official split. The sample "
              "may come from a different data release.")
    if len(used) == 1:
        verdict = f"drawn entirely from the {used[0].upper()} split"
    elif len(used) > 1:
        verdict = "MIXES the official splits: " + ", ".join(
            f"{s} {int(membership[s].sum()):,}" for s in used
        )
    else:
        verdict = "no galaxy matched any official split"
    print(f"VERDICT: the sample is {verdict}.")
    if len(used) > 1:
        print("\nThat matters: GalaxiesML's train/val/test partition is not "
              "preserved in this sample, so results from it are a self-contained "
              "pilot and cannot be compared against numbers computed on the "
              "official splits.")
    print("=" * 60)

    report["verdict"] = verdict
    report["splits_used"] = used
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out} -- send this back to the team.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
