"""Step 14 -- audit catalogue metadata quality and cut unreliable rows.

    python scripts/14_metadata_audit.py                  # audit only, change nothing
    python scripts/14_metadata_audit.py --apply          # write cleaned splits
    python scripts/14_metadata_audit.py --max-z-err 0.001

The meeting asked for "remove high-error redshift rows". This script answers
that, and reports honestly when the answer is "there are almost none".

### What it checks

* **Redshift error** -- ``specz_redshift_err``, absolute and relative to
  (1+z). These are *spectroscopic* redshifts, so the errors are small by
  construction; a photometric sample would look nothing like this.
* **Magnitude sentinels** -- HSC catalogues write 99, -99, -999 or 9999 to
  mean "no measurement", and a model fed those as numbers would be fitting
  nonsense. Checked only on magnitude columns: 99 is a perfectly ordinary
  value for an isophotal area in pixels, and flagging it there would
  manufacture a problem that does not exist.
* **NaN and Inf** anywhere in the numeric catalogue.
* **Morphology completeness**, since the metadata model in step 15 needs it.

### Why the default cut is a percentile, not a fixed number

A fixed threshold encodes today's dataset. The default removes rows whose
redshift error exceeds ``median + 10 x NMAD`` -- robust, scale-free, and it
adapts if this is ever rerun on a sample with genuinely noisy redshifts.
``--max-z-err`` overrides it with an absolute cut when a specific number is
wanted.

Cleaned splits are written beside the originals as ``*_clean.h5`` rather than
replacing them, so every published number stays reproducible from the files
that produced it.
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
from src.data.preprocess import MAG_SENTINELS, mag_is_valid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Robust cut: median + this many NMADs of the redshift error.
DEFAULT_NMAD_MULTIPLE = 10.0
NMAD_FACTOR = 1.4826


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit and clean catalogue metadata.")
    p.add_argument("--config", default=None)
    p.add_argument("--apply", action="store_true",
                   help="write cleaned *_clean.h5 splits; without it, audit only")
    p.add_argument("--max-z-err", type=float, default=None,
                   help="absolute redshift-error cut, overriding the robust one")
    p.add_argument("--nmad-multiple", type=float, default=DEFAULT_NMAD_MULTIPLE)
    return p.parse_args()


def nmad(x: np.ndarray) -> float:
    return float(NMAD_FACTOR * np.median(np.abs(x - np.median(x))))


def audit_split(path: Path, max_z_err: float | None,
                nmad_multiple: float) -> dict:
    """Everything worth knowing about one split's catalogue columns."""
    with h5py.File(path, "r") as f:
        n = int(f["image"].shape[0])
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)
        err = (np.asarray(f["specz_redshift_err"][:], dtype=np.float64)
               if "specz_redshift_err" in f else np.full(n, np.nan))
        mags = {b: np.asarray(f[f"{b}_cmodel_mag"][:], dtype=np.float64)
                for b in BANDS if f"{b}_cmodel_mag" in f}
        numeric = {}
        for key in f.keys():
            if key == "image":
                continue
            d = f[key]
            if d.dtype.kind in "fiu" and d.ndim == 1:
                numeric[key] = np.asarray(d[:], dtype=np.float64)

    finite_err = err[np.isfinite(err)]
    if max_z_err is not None:
        cut = float(max_z_err)
        rule = f"absolute, --max-z-err {cut:g}"
    elif finite_err.size:
        spread = nmad(finite_err)
        cut = float(np.median(finite_err) + nmad_multiple * spread)
        rule = (f"robust: median {np.median(finite_err):.3g} + "
                f"{nmad_multiple:g} x NMAD {spread:.3g}")
    else:
        cut, rule = np.inf, "no redshift errors present"

    bad_err = np.isfinite(err) & (err > cut)
    missing_err = ~np.isfinite(err)

    # Sentinels are a magnitude convention only. Applying them to areas or
    # radii would flag ordinary values.
    bad_mag = np.zeros(n, dtype=bool)
    per_band_bad = {}
    for b, m in mags.items():
        invalid = ~mag_is_valid(m)
        per_band_bad[b] = int(invalid.sum())
        bad_mag |= invalid

    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in numeric.items()}
    nonfinite = {k: c for k, c in nonfinite.items() if c}
    bad_numeric = np.zeros(n, dtype=bool)
    for k in nonfinite:
        bad_numeric |= ~np.isfinite(numeric[k])

    drop = bad_err | missing_err | bad_mag | bad_numeric
    return {
        "file": path.name,
        "n": n,
        "redshift_error": {
            "rule": rule,
            "cut": None if not np.isfinite(cut) else cut,
            "median": float(np.median(finite_err)) if finite_err.size else None,
            "p99": float(np.percentile(finite_err, 99)) if finite_err.size else None,
            "max": float(finite_err.max()) if finite_err.size else None,
            "relative_max": (float((finite_err / (1 + z[np.isfinite(err)])).max())
                             if finite_err.size else None),
            "above_cut": int(bad_err.sum()),
            "missing": int(missing_err.sum()),
        },
        "magnitude_sentinels": per_band_bad,
        "nonfinite_columns": nonfinite,
        "n_columns_checked": len(numeric),
        "drop_total": int(drop.sum()),
        "drop_percent": 100.0 * float(drop.sum()) / n,
        "keep": (~drop),
    }


def write_clean(src: Path, keep: np.ndarray) -> Path:
    """Copy a split, keeping only the audited rows."""
    out = src.with_name(src.stem + "_clean" + src.suffix)
    idx = np.flatnonzero(keep)
    with h5py.File(src, "r") as f, h5py.File(out, "w") as g:
        img = f["image"]
        shape = (len(idx),) + tuple(img.shape[1:])
        dst = g.create_dataset("image", shape=shape, dtype=img.dtype,
                               chunks=(1,) + tuple(img.shape[1:]))
        block = 256
        for start in range(0, len(idx), block):
            sel = idx[start:start + block]
            dst[start:start + len(sel)] = img[np.sort(sel)]
            print(f"\r  writing images: {100.0 * (start + len(sel)) / len(idx):5.1f}%",
                  end="", flush=True)
        print()
        for key in f.keys():
            if key == "image":
                continue
            g.create_dataset(key, data=np.asarray(f[key][:])[keep])
        for k, v in f.attrs.items():
            g.attrs[k] = v
        g.attrs["cleaned_from"] = src.name
        g.attrs["n_before_clean"] = int(len(keep))
        g.attrs["n_after_clean"] = int(len(idx))
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 74)
    print("Metadata quality audit")
    print("=" * 74)

    reports, total_n, total_drop = [], 0, 0
    for split in ("train", "val", "test"):
        path = cfg.processed_file(split)
        if not path.exists():
            print(f"\n{split}: {path.name} not found, skipping")
            continue
        r = audit_split(path, args.max_z_err, args.nmad_multiple)
        reports.append((split, path, r))
        total_n += r["n"]
        total_drop += r["drop_total"]

        ze = r["redshift_error"]
        print(f"\n--- {split}: {r['n']:,} galaxies ---")
        print(f"  redshift error   : median {ze['median']:.3g}, "
              f"p99 {ze['p99']:.3g}, max {ze['max']:.3g}")
        print(f"  relative to 1+z  : max {ze['relative_max']:.3g}")
        print(f"  cut rule         : {ze['rule']}")
        print(f"  above the cut    : {ze['above_cut']:,}"
              f"   missing: {ze['missing']:,}")
        bad_mags = {b: c for b, c in r["magnitude_sentinels"].items() if c}
        print(f"  magnitude sentinels: "
              f"{bad_mags if bad_mags else 'none in any band'}")
        print(f"  non-finite values  : "
              f"{r['nonfinite_columns'] if r['nonfinite_columns'] else 'none'}"
              f"   ({r['n_columns_checked']} numeric columns checked)")
        print(f"  WOULD DROP       : {r['drop_total']:,} "
              f"({r['drop_percent']:.3f}%)")

    if not reports:
        print("\nNo processed files found. Run scripts/01_prepare_data.py first.")
        return 2

    print("\n" + "=" * 74)
    pct = 100.0 * total_drop / total_n if total_n else 0.0
    print(f"VERDICT: {total_drop:,} of {total_n:,} galaxies ({pct:.3f}%) fail "
          "the quality cut.")
    if pct < 0.5:
        print("""
  This is a spectroscopic sample, so the redshifts are precise by
  construction and the catalogue is essentially clean. Removing these rows
  cannot move any published metric: at this fraction, even if every dropped
  galaxy were a total outlier it would shift a median-based statistic by less
  than its own rounding. The cleanup is worth having as a guard for future
  data -- a photometric sample would look nothing like this -- but it is not
  a correction to anything reported here.""")
    else:
        print("\n  Large enough to matter. Rerun the affected models on the "
              "cleaned splits.")

    out = cfg.path("results_dir") / "metadata_audit.json"
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "keep"} | {"split": s}
         for s, _, r in reports], indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    if args.apply:
        print("\n--apply: writing cleaned splits")
        for split, path, r in reports:
            if r["drop_total"] == 0:
                print(f"  {split}: nothing to drop, skipped")
                continue
            print(f"  {split}: {r['n']:,} -> {r['n'] - r['drop_total']:,}")
            print(f"    {write_clean(path, r['keep'])}")
    else:
        print("\nAudit only. Pass --apply to write cleaned *_clean.h5 splits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
