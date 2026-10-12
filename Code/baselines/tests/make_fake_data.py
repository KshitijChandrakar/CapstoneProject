"""Build fake GalaxiesML files with the real structure, for testing.

The magnitudes are measured off the generated images, so the B0 sign check is a
genuine measurement rather than something baked in. A few NaN pixels are
injected so the masking code actually runs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

BANDS = ("g", "r", "i", "z", "y")
IMG = 64
ZEROPOINT = 27.0

#: Rough HSC flux ratios of a red, low-redshift galaxy, relative to i.
#: y is slightly brighter than z, which is what makes the B0 sign testable.
BAND_FLUX_RATIO = {"g": 0.30, "r": 0.62, "i": 1.00, "z": 1.22, "y": 1.30}


def _profile(size: int, half_light: float, ellip: float, angle: float, rng) -> np.ndarray:
    """A smooth, centrally peaked galaxy profile, normalised to sum 1."""
    y, x = np.mgrid[:size, :size].astype(np.float64)
    cy = cx = (size - 1) / 2.0
    cy += rng.normal(0, 0.8)
    cx += rng.normal(0, 0.8)
    ca, sa = np.cos(angle), np.sin(angle)
    xr = (x - cx) * ca + (y - cy) * sa
    yr = -(x - cx) * sa + (y - cy) * ca
    r = np.sqrt(xr**2 + (yr / max(1.0 - ellip, 0.2)) ** 2)
    prof = np.exp(-(r / max(half_light, 1.0)) ** 0.6)
    return prof / prof.sum()


def _aperture(size: int, radius: float = 15.0) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    return ((yy - c) ** 2 + (xx - c) ** 2) <= radius**2


def make_fake_split(
    path: Path,
    n: int,
    seed: int,
    z_low: float = 0.01,
    z_high: float = 0.5,
    inject_bad: bool = True,
    id_offset: int = 0,
) -> dict:
    """Write one fake GalaxiesML-style HDF5 file."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, len(BANDS), IMG, IMG), dtype=np.float32)

    redshift = rng.uniform(z_low, z_high, size=n).astype(np.float32)
    object_id = (74643924559360000 + id_offset + np.arange(n) * 7).astype(np.int64)
    total_i = 10.0 ** rng.uniform(2.0, 4.2, size=n)
    # A few near-blank galaxies, so some apertures sum to a NEGATIVE flux and
    # the "exclude vs clamp" magnitude code path is genuinely exercised.
    n_blank = max(1, int(0.04 * n))
    total_i[rng.choice(n, size=n_blank, replace=False)] = 1e-3
    half_light = rng.uniform(2.5, 9.0, size=n)
    ellip = rng.uniform(0.0, 0.6, size=n)
    angle = rng.uniform(0, np.pi, size=n)
    noise = rng.uniform(0.4, 1.6, size=n)

    for k in range(n):
        prof = _profile(IMG, half_light[k], ellip[k], angle[k], rng)
        for bi, band in enumerate(BANDS):
            ratio = BAND_FLUX_RATIO[band] * float(np.exp(rng.normal(0, 0.09)))
            signal = total_i[k] * ratio * prof
            # Sky-subtracted noise: genuinely negative in the outskirts.
            images[k, bi] = (signal + rng.normal(0.0, noise[k], (IMG, IMG))).astype(
                np.float32
            )

    # Magnitudes measured from the images, in the same aperture the metrics use.
    ap = _aperture(IMG)
    mags: dict[str, np.ndarray] = {}
    for bi, band in enumerate(BANDS):
        flux = np.nansum(np.where(ap[None], images[:, bi], 0.0), axis=(1, 2))
        m = np.where(flux > 0, -2.5 * np.log10(np.maximum(flux, 1e-30)) + ZEROPOINT, 99.0)
        mags[band] = m.astype(np.float32)

    if inject_bad and n > 10:
        images[3, 4, 10:13, 20:22] = np.nan     # NaN target pixels
        images[5, 0, 40, 40] = np.nan           # NaN input pixel

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=images)
        f.create_dataset("object_id", data=object_id)
        f.create_dataset("specz_redshift", data=redshift)
        f.create_dataset("specz_redshift_err",
                         data=rng.uniform(1e-5, 1e-3, size=n).astype(np.float32))
        f.create_dataset("specz_flag_homogeneous", data=np.ones(n, dtype=bool))
        f.create_dataset("specz_mag_i", data=mags["i"])
        for band in BANDS:
            f.create_dataset(f"{band}_cmodel_mag", data=mags[band])

    return {
        "path": path,
        "n": n,
        "n_in_window": int(np.count_nonzero((redshift >= 0.0) & (redshift < 0.4))),
    }


def make_all_splits(raw_dir: Path, n_train: int = 400, n_val: int = 120,
                    n_test: int = 120) -> dict[str, dict]:
    """Build fake train / val / test files with disjoint object_ids."""
    specs = {
        "train": ("5x64x64_training_with_morphology.hdf5", n_train, 11, 0),
        "val": ("5x64x64_validation_with_morphology.hdf5", n_val, 22, 10_000_000),
        "test": ("5x64x64_testing_with_morphology.hdf5", n_test, 33, 20_000_000),
    }
    out: dict[str, dict] = {}
    for split, (name, n, seed, offset) in specs.items():
        out[split] = make_fake_split(raw_dir / name, n=n, seed=seed, id_offset=offset)
    return out
