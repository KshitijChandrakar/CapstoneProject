"""Cut the small working sample out of the full GalaxiesML HDF5.

RUN THIS ON THE MACHINE THAT HAS THE BIG FILE. It turns the ~17 GB training
file into a ~160 MB file holding only the galaxies we actually use, which is
small enough to send over Google Drive / WhatsApp / a USB stick.

    python make_sample.py "5x64x64_training_with_morphology.hdf5"

Optional:
    python make_sample.py BIGFILE.hdf5 --out sample.h5 --n 2000 --zmax 0.2

Only numpy and h5py are needed -- no PyTorch, no project code. The script reads
the images in small batches, so it never loads the big file into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

BANDS = ("g", "r", "i", "z", "y")
MAG_KEYS = tuple(f"{b}_cmodel_mag" for b in BANDS)
#: Everything the baselines need. Nothing else is copied.
EXTRA_KEYS = ("object_id", "specz_redshift") + MAG_KEYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a small, shareable sample from a full GalaxiesML HDF5."
    )
    p.add_argument("source", help="the big GalaxiesML .hdf5 file")
    p.add_argument("--out", default=None,
                   help="output file (default: sample_z<zmax>_<n>.h5)")
    p.add_argument("--n", type=int, default=2000,
                   help="how many galaxies to keep (default 2000)")
    p.add_argument("--zmax", type=float, default=0.2,
                   help="keep galaxies with specz_redshift < this (default 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed, matches pandas random_state (default 42)")
    p.add_argument("--batch", type=int, default=200,
                   help="images read per chunk (default 200; lower it if RAM is tight)")
    p.add_argument("--compress", action="store_true",
                   help="gzip the images (smaller file, slower to write and read)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: no such file: {src}")
        return 2

    out = Path(args.out) if args.out else Path(f"sample_z{args.zmax}_{args.n}.h5")

    print(f"Reading   : {src}  ({src.stat().st_size / 1e9:.1f} GB)")

    with h5py.File(src, "r") as f:
        missing = [k for k in ("image", "specz_redshift") if k not in f]
        if missing:
            print(f"ERROR: {src.name} is missing {missing}.")
            print(f"       It contains: {sorted(f.keys())[:15]} ...")
            return 2

        redshift = np.asarray(f["specz_redshift"][:], dtype=np.float64)
        n_total = redshift.size
        images = f["image"]
        print(f"Found     : {n_total:,} galaxies, image shape {images.shape[1:]}")

        # -- select ---------------------------------------------------------
        pool = np.flatnonzero(redshift < args.zmax)
        print(f"Redshift  : {pool.size:,} galaxies with specz_redshift < {args.zmax}")
        if pool.size == 0:
            print("ERROR: nothing passes the redshift cut.")
            return 2
        if args.n > pool.size:
            print(f"WARNING: only {pool.size:,} galaxies available; keeping all of them.")
            keep = pool
        else:
            # Same call pandas' DataFrame.sample(n, random_state=seed) makes,
            # so this reproduces the original sample exactly.
            rng = np.random.RandomState(args.seed)
            keep = pool[rng.choice(pool.size, size=args.n, replace=False)]
        keep = np.sort(keep)        # h5py needs a sorted index list
        n_keep = keep.size

        est_mb = n_keep * int(np.prod(images.shape[1:])) * 4 / 1e6
        print(f"Keeping   : {n_keep:,} galaxies  (~{est_mb:.0f} MB uncompressed)")

        # -- which 1-D columns can we copy? ---------------------------------
        available = [k for k in EXTRA_KEYS if k in f]
        for k in EXTRA_KEYS:
            if k not in f:
                print(f"WARNING: '{k}' is not in the file and will be missing.")

        # -- write ----------------------------------------------------------
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"compression": "gzip", "compression_opts": 4} if args.compress else {}

        with h5py.File(out, "w") as g:
            dst = g.create_dataset(
                "image",
                shape=(n_keep,) + images.shape[1:],
                dtype=np.float32,
                chunks=(min(32, n_keep),) + images.shape[1:],
                **kwargs,
            )
            for start in range(0, n_keep, args.batch):
                stop = min(start + args.batch, n_keep)
                dst[start:stop] = images[keep[start:stop]]
                pct = 100.0 * stop / n_keep
                print(f"\r  copying images: {pct:5.1f}%  ({stop:,}/{n_keep:,})",
                      end="", flush=True)
            print()

            for k in available:
                g.create_dataset(k, data=np.asarray(f[k][:])[keep])

            g.attrs["source_file"] = src.name
            g.attrs["redshift_cut"] = args.zmax
            g.attrs["sample_size"] = n_keep
            g.attrs["random_seed"] = args.seed
            g.attrs["n_in_source"] = n_total

    # -- verify --------------------------------------------------------------
    with h5py.File(out, "r") as g:
        oid = np.asarray(g["object_id"][:]) if "object_id" in g else np.array([])
        z = np.asarray(g["specz_redshift"][:])
        img = g["image"]
        digest = hashlib.sha256(np.sort(oid).tobytes()).hexdigest()[:16] if oid.size else "n/a"

        print(f"\nWrote     : {out}  ({out.stat().st_size / 1e6:.0f} MB)")
        print(f"  galaxies      : {img.shape[0]:,}")
        print(f"  image shape   : {img.shape[1:]}  dtype {img.dtype}")
        print(f"  redshift range: {z.min():.4f} to {z.max():.4f}")
        print(f"  datasets      : {sorted(g.keys())}")
        print(f"  object_id hash: {digest}   <-- send this so the sample can be verified")

        if z.max() >= args.zmax:
            print("  [!] WARNING: a redshift is above the cut -- something is wrong.")

    print(f"\nDone. Send '{out.name}' to the team.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
