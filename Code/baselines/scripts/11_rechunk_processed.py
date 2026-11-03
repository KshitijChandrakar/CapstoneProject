"""Step 11 -- rewrite the processed files with one galaxy per HDF5 chunk.

    python scripts/11_rechunk_processed.py            # all three splits
    python scripts/11_rechunk_processed.py --split train
    python scripts/11_rechunk_processed.py --dry-run  # report, change nothing

Why this exists
---------------
``preprocess.py`` originally wrote the image dataset with ``chunks=(32, ...)``.
HDF5 must read an entire chunk to return any element of it, and training reads
single random indices, so every galaxy cost a 2.6 MB read to deliver 80 KB --
32x amplification. Over one epoch of 84,911 shuffled reads that is 222 GB of
I/O to consume 7 GB of data, which starved the GPU down to 0.36 s/step on a
model that should run at well under 0.1. h5py's 1 MB default chunk cache cannot
absorb it, being smaller than a single chunk.

``preprocess.py`` now writes ``chunks=(1, ...)``, so files prepared from here on
are fine. This script fixes files already on disk without re-deriving them from
the 23 GB raw catalogue -- the filtering, the splits and the statistics stay
exactly as they are, and the pixel values are verified identical afterwards.

It is purely a storage-layout change. No number in any result moves.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VERIFY_SAMPLE = 400       # galaxies compared pixel-for-pixel after the rewrite
COPY_BLOCK = 256          # galaxies per read/write block


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rechunk processed HDF5 files.")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="one split instead of all three")
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="report the current chunking and exit")
    p.add_argument("--keep-original", action="store_true",
                   help="leave the old file beside the new one as .old")
    return p.parse_args()


def describe(path: Path) -> dict | None:
    """Current chunking of one file, or None if it cannot be read."""
    try:
        with h5py.File(path, "r") as f:
            img = f["image"]
            chunks = img.chunks
            per_galaxy = int(np.prod(img.shape[1:])) * img.dtype.itemsize
            return {
                "n": int(img.shape[0]),
                "shape": tuple(int(v) for v in img.shape[1:]),
                "dtype": str(img.dtype),
                "chunks": chunks,
                "galaxies_per_chunk": int(chunks[0]) if chunks else None,
                "chunk_bytes": (int(np.prod(chunks)) * img.dtype.itemsize
                                if chunks else None),
                "galaxy_bytes": per_galaxy,
                "compression": img.compression,
            }
    except (OSError, KeyError) as exc:
        print(f"  cannot read {path.name}: {exc}")
        return None


def report(path: Path, info: dict) -> bool:
    """Print the current layout. True if a rechunk would help."""
    print(f"\n{path.name}")
    print(f"  {info['n']:,} galaxies, {info['shape']} {info['dtype']}"
          f"{', ' + info['compression'] if info['compression'] else ''}")
    if info["chunks"] is None:
        print("  contiguous (not chunked) -- random reads are already direct")
        return False
    amp = info["chunk_bytes"] / info["galaxy_bytes"]
    print(f"  chunks {info['chunks']} = {info['chunk_bytes'] / 1e6:.2f} MB, "
          f"holding {info['galaxies_per_chunk']} galaxies")
    print(f"  a single random read pulls {amp:.0f}x what it needs")
    if amp <= 1.0:
        print("  already one galaxy per chunk -- nothing to do")
        return False
    per_epoch = info["n"] * info["chunk_bytes"]
    useful = info["n"] * info["galaxy_bytes"]
    print(f"  per epoch: {per_epoch / 1e9:,.1f} GB read to use "
          f"{useful / 1e9:.2f} GB")
    return True


def rechunk(path: Path, keep_original: bool) -> bool:
    """Rewrite one file with chunks=(1, ...). Returns True on success."""
    tmp = path.with_suffix(path.suffix + ".rechunk")
    if tmp.exists():
        tmp.unlink()

    started = time.time()
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        img = src["image"]
        n = int(img.shape[0])
        img_shape = tuple(int(v) for v in img.shape[1:])

        out = dst.create_dataset(
            "image", shape=(n,) + img_shape, dtype=img.dtype,
            chunks=(1,) + img_shape,
        )
        for start in range(0, n, COPY_BLOCK):
            stop = min(start + COPY_BLOCK, n)
            out[start:stop] = img[start:stop]
            print(f"\r  copying images: {100.0 * stop / n:5.1f}%  "
                  f"({stop:,}/{n:,})", end="", flush=True)
        print()

        # Everything that is not the image dataset is copied verbatim, so the
        # catalogue columns and the provenance attributes survive untouched.
        for key in src.keys():
            if key != "image":
                src.copy(key, dst)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["rechunked_from"] = str(img.chunks)
        dst.attrs["rechunked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime())

    # -- verify before anything is overwritten -------------------------------
    rng = np.random.default_rng(0)
    with h5py.File(path, "r") as a, h5py.File(tmp, "r") as b:
        n = int(a["image"].shape[0])
        if int(b["image"].shape[0]) != n:
            print(f"  FAILED: row count changed, {n:,} -> "
                  f"{int(b['image'].shape[0]):,}. Original left alone.")
            tmp.unlink()
            return False

        missing = set(a.keys()) - set(b.keys())
        if missing:
            print(f"  FAILED: datasets lost: {sorted(missing)}. "
                  "Original left alone.")
            tmp.unlink()
            return False

        idx = np.unique(rng.integers(0, n, size=min(VERIFY_SAMPLE, n)))
        # Endpoints too: an off-by-one in the block loop would hide in the middle.
        idx = np.unique(np.concatenate([[0, n - 1], idx]))
        for i in idx:
            left = np.asarray(a["image"][int(i)])
            right = np.asarray(b["image"][int(i)])
            # NaNs are expected and must survive as NaNs, so compare bitwise
            # rather than with ==, which is False for NaN against itself.
            if left.shape != right.shape or not np.array_equal(
                left, right, equal_nan=True
            ):
                print(f"  FAILED: galaxy {i} differs. Original left alone.")
                tmp.unlink()
                return False

        for key in a.keys():
            if key == "image":
                continue
            if not np.array_equal(np.asarray(a[key]), np.asarray(b[key]),
                                  equal_nan=True):
                print(f"  FAILED: column {key!r} differs. Original left alone.")
                tmp.unlink()
                return False

    print(f"  verified: {len(idx)} galaxies pixel-identical "
          f"(NaNs included), all columns identical")

    backup = path.with_suffix(path.suffix + ".old")
    path.replace(backup)
    tmp.replace(path)
    if keep_original:
        print(f"  original kept at {backup.name}")
    else:
        backup.unlink()

    took = time.time() - started
    print(f"  done in {took / 60:.1f} min")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    splits = [args.split] if args.split else ["train", "val", "test"]

    print("=" * 74)
    print("Rechunk processed files to one galaxy per HDF5 chunk")
    print("=" * 74)
    print("\nThe layout reported below is the CURRENT one, before any change.")
    print("32 galaxies per chunk is the problem; 1 is the goal.")

    todo: list[tuple[Path, dict]] = []
    for split in splits:
        path = cfg.processed_file(split)
        if not path.exists():
            print(f"\n{path.name}: not found, skipping")
            continue
        info = describe(path)
        if info and report(path, info):
            todo.append((path, info))

    if not todo:
        print("\nNothing to rechunk.")
        return 0

    if args.dry_run:
        print(f"\n--dry-run: {len(todo)} file(s) would be rewritten.")
        return 0

    # The rewrite needs room for a second copy of the largest file.
    biggest = max(p.stat().st_size for p, _ in todo)
    free = shutil.disk_usage(todo[0][0].parent).free
    print(f"\nFree space: {free / 1e9:.1f} GB. "
          f"Largest file: {biggest / 1e9:.1f} GB, and the rewrite needs room "
          f"for a second copy of it.")
    if free < biggest * 1.1:
        print("Not enough free space. Rechunk one split at a time with "
              "--split, or free some space first.")
        return 2

    ok = True
    for path, _ in todo:
        print(f"\n--- {path.name} ---")
        ok = rechunk(path, args.keep_original) and ok

    if not ok:
        print("\nAt least one file failed. The originals are intact.")
        return 1

    print("\nAll files rechunked. The pixel data is unchanged, so no result "
          "moves; only the read pattern does.")
    print("\nNext:  python scripts/08_train_b2.py --seed 42")
    return 0


if __name__ == "__main__":
    sys.exit(main())
