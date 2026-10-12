"""Redshift filtering and global normalisation statistics.

Both jobs stream the HDF5 in chunks: the unfiltered training file is ~17 GB and
the z<0.4 subset is still ~7 GB, so nothing here may assume the images fit in
RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.config import BANDS, Config

MAG_KEYS: tuple[str, ...] = tuple(f"{b}_cmodel_mag" for b in BANDS)

#: 1-D datasets copied into the filtered subset. Everything the baselines need,
#: plus the columns the later models are likely to want.
CARRY_KEYS: tuple[str, ...] = (
    "object_id",
    "specz_redshift",
    "specz_redshift_err",
    "specz_flag_homogeneous",
    "specz_mag_i",
) + MAG_KEYS

#: Values HSC/SExtractor catalogues use to mean "no measurement".
MAG_SENTINELS: tuple[float, ...] = (99.0, -99.0, -999.0, 9999.0)


def mag_is_valid(mag: np.ndarray) -> np.ndarray:
    """Boolean mask of usable magnitudes (finite and not a sentinel)."""
    ok = np.isfinite(mag)
    for sentinel in MAG_SENTINELS:
        ok &= np.abs(mag - sentinel) > 1e-6
    return ok


def select_indices(
    h5_path: Path, z_min: float, z_max: float, require_flag: bool
) -> np.ndarray:
    """Sorted indices of galaxies passing the redshift cut (and the flag).

    The cut is ``z_min <= specz_redshift < z_max``, matching the spec.
    """
    with h5py.File(h5_path, "r") as f:
        if "specz_redshift" not in f:
            raise KeyError(f"{h5_path.name} has no 'specz_redshift' dataset.")
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)
        keep = (z >= z_min) & (z < z_max)

        if require_flag:
            if "specz_flag_homogeneous" in f:
                flag = np.asarray(f["specz_flag_homogeneous"][:])
                keep &= flag.astype(bool)
            else:
                print(f"  [note] {h5_path.name} has no 'specz_flag_homogeneous'; "
                      "the flag filter was skipped.")

    return np.flatnonzero(keep)      # already ascending, as h5py requires


def filter_redshift_subset(
    input_h5: Path,
    output_h5: Path,
    z_min: float = 0.0,
    z_max: float = 0.4,
    require_flag: bool = True,
    chunk_size: int = 500,
    overwrite: bool = False,
) -> dict:
    """Write the galaxies inside the redshift window to a new HDF5 file.

    Images are copied chunk by chunk, so peak memory is roughly
    ``chunk_size * 5 * 64 * 64 * 4`` bytes (~40 MB at the default).
    """
    input_h5, output_h5 = Path(input_h5), Path(output_h5)
    if not input_h5.exists():
        raise FileNotFoundError(f"Raw file not found: {input_h5}")

    if output_h5.exists() and not overwrite:
        with h5py.File(output_h5, "r") as g:
            n = int(g["image"].shape[0])
        print(f"  {output_h5.name} already exists ({n:,} galaxies) -- skipping. "
              "Pass --overwrite to rebuild.")
        return {"output": str(output_h5), "n_kept": n, "skipped": True}

    keep = select_indices(input_h5, z_min, z_max, require_flag)
    n_keep = int(keep.size)
    if n_keep == 0:
        raise ValueError(
            f"No galaxy in {input_h5.name} satisfies {z_min} <= z < {z_max}."
        )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_h5.with_suffix(output_h5.suffix + ".partial")

    with h5py.File(input_h5, "r") as f, h5py.File(tmp, "w") as g:
        images = f["image"]
        n_total = int(images.shape[0])
        img_shape = tuple(images.shape[1:])

        dst = g.create_dataset(
            "image",
            shape=(n_keep,) + img_shape,
            dtype=np.float32,
            chunks=(min(32, n_keep),) + img_shape,
        )
        for start in range(0, n_keep, chunk_size):
            stop = min(start + chunk_size, n_keep)
            dst[start:stop] = images[keep[start:stop]]
            print(f"\r  copying images: {100.0 * stop / n_keep:5.1f}%  "
                  f"({stop:,}/{n_keep:,})", end="", flush=True)
        print()

        carried: list[str] = []
        for key in CARRY_KEYS:
            if key in f:
                g.create_dataset(key, data=np.asarray(f[key][:])[keep])
                carried.append(key)

        g.attrs["source_file"] = input_h5.name
        g.attrs["z_min"] = z_min
        g.attrs["z_max"] = z_max
        g.attrs["require_homogeneous_flag"] = require_flag
        g.attrs["n_in_source"] = n_total
        g.attrs["n_kept"] = n_keep

    tmp.replace(output_h5)

    missing = [k for k in CARRY_KEYS if k not in carried]
    if missing:
        print(f"  [note] not present in the source and therefore not copied: {missing}")

    print(f"  kept {n_keep:,} of {n_total:,} galaxies "
          f"({100.0 * n_keep / n_total:.1f}%) -> {output_h5.name} "
          f"({output_h5.stat().st_size / 1e9:.2f} GB)")

    return {
        "output": str(output_h5),
        "n_source": n_total,
        "n_kept": n_keep,
        "carried_keys": carried,
        "missing_keys": missing,
        "skipped": False,
    }


def compute_and_save_global_stats(
    train_h5_path: Path,
    output_json_path: Path,
    chunk_size: int = 500,
    eps: float = 1e-10,
) -> dict:
    """Per-band mean and std over every pixel of the TRAIN split.

    Streams the file, accumulating sums and sums of squares in float64, so a
    7 GB image array never has to be resident.

    Only the train split may be used: computing these on val or test would leak
    information about data the model is not supposed to have seen.
    """
    train_h5_path, output_json_path = Path(train_h5_path), Path(output_json_path)
    if not train_h5_path.exists():
        raise FileNotFoundError(f"Processed train file not found: {train_h5_path}")

    with h5py.File(train_h5_path, "r") as f:
        images = f["image"]
        n, n_bands = int(images.shape[0]), int(images.shape[1])
        if n_bands != len(BANDS):
            raise ValueError(f"Expected {len(BANDS)} bands, found {n_bands}.")

        total = np.zeros(n_bands, dtype=np.float64)
        total_sq = np.zeros(n_bands, dtype=np.float64)
        count = np.zeros(n_bands, dtype=np.float64)
        n_nonfinite = 0

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            batch = np.asarray(images[start:stop], dtype=np.float64)
            finite = np.isfinite(batch)
            n_nonfinite += int(np.count_nonzero(~finite))
            safe = np.where(finite, batch, 0.0)

            total += safe.sum(axis=(0, 2, 3))
            total_sq += (safe**2).sum(axis=(0, 2, 3))
            count += finite.sum(axis=(0, 2, 3))

            print(f"\r  scanning for statistics: {100.0 * stop / n:5.1f}%  "
                  f"({stop:,}/{n:,})", end="", flush=True)
        print()

    if np.any(count == 0):
        raise ValueError("A band has no finite pixels at all; cannot standardise.")

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(variance)

    if np.any(std <= 0):
        bad = [BANDS[i] for i in np.flatnonzero(std <= 0)]
        raise ValueError(f"Zero variance in band(s) {bad}; cannot standardise.")

    payload = {
        "bands": list(BANDS),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "eps": eps,
        "formula": "x_norm = (x - mean) / (std + eps)",
        "computed_on": train_h5_path.name,
        "split": "train only",
        "n_galaxies": n,
        "n_pixels_per_band": [int(c) for c in count],
        "n_nonfinite_pixels": n_nonfinite,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  per-band statistics (train split only):")
    for i, b in enumerate(BANDS):
        print(f"    {b}: mean {mean[i]:+.6g}   std {std[i]:.6g}")
    if n_nonfinite:
        print(f"  [note] {n_nonfinite:,} non-finite pixels were ignored.")
    print(f"  saved -> {output_json_path}")

    return payload


def load_global_stats(stats_json_path: Path) -> dict:
    """Read the saved statistics back, as numpy arrays.

    Returns a dict with ``mean``, ``std``, ``denom`` (= std + eps) and ``eps``.
    """
    stats_json_path = Path(stats_json_path)
    if not stats_json_path.exists():
        raise FileNotFoundError(
            f"{stats_json_path} not found. Run scripts/01_prepare_data.py first."
        )
    d = json.loads(stats_json_path.read_text(encoding="utf-8"))
    mean = np.asarray(d["mean"], dtype=np.float64)
    std = np.asarray(d["std"], dtype=np.float64)
    eps = float(d.get("eps", 1e-10))
    return {
        "bands": d.get("bands", list(BANDS)),
        "mean": mean,
        "std": std,
        "eps": eps,
        "denom": std + eps,
        "meta": d,
    }


def normalize(x: np.ndarray, stats: dict, band_idx: tuple[int, ...]) -> np.ndarray:
    """Standardise `x` of shape (..., C, H, W) using the given bands' stats."""
    idx = list(band_idx)
    mean = stats["mean"][idx].reshape(-1, 1, 1)
    denom = stats["denom"][idx].reshape(-1, 1, 1)
    return ((x - mean) / denom).astype(np.float32)


def denormalize(x: np.ndarray, stats: dict, band_idx: tuple[int, ...]) -> np.ndarray:
    """Invert :func:`normalize`, returning physical flux units."""
    idx = list(band_idx)
    mean = stats["mean"][idx].reshape(-1, 1, 1)
    denom = stats["denom"][idx].reshape(-1, 1, 1)
    return (x * denom + mean).astype(np.float32)


def verify_split_disjoint(processed_files: dict[str, Path]) -> dict:
    """Assert that no object_id appears in more than one split."""
    ids: dict[str, set[int]] = {}
    for split, path in processed_files.items():
        with h5py.File(path, "r") as f:
            if "object_id" not in f:
                raise KeyError(f"{path.name} has no 'object_id' dataset.")
            ids[split] = {int(v) for v in np.asarray(f["object_id"][:])}

    report: dict = {"counts": {k: len(v) for k, v in ids.items()}, "overlaps": {}}
    names = list(ids)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ka, kb = names[a], names[b]
            shared = ids[ka] & ids[kb]
            report["overlaps"][f"{ka}&{kb}"] = len(shared)
            assert not shared, (
                f"{len(shared)} object_id(s) appear in both {ka} and {kb}. "
                "The splits must be disjoint."
            )
    report["all_disjoint"] = True
    return report
