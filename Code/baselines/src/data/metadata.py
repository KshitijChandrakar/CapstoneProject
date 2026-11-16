"""Catalogue features for the metadata model, joined to the processed splits.

The processed splits carry only what the image models need -- magnitudes,
redshift, object ids. The morphology columns stay in the 23 GB raw files. This
module joins them back on ``object_id``, so the metadata model sees exactly the
same galaxies, in the same order, as every other model in the ladder.

Joining rather than re-preparing is deliberate. Regenerating the processed
files with extra columns would rewrite the inputs that produced every published
number, for no benefit: the pixels would be identical and only the sidecar
columns would differ. A join costs one pass over the raw catalogue and leaves
the published artefacts untouched.

### What counts as a fair feature

**The target band's own measurements are excluded.** ``y_cmodel_mag`` is the
quantity being predicted, and every other ``y_*`` column -- half-light radius,
Sersic index, ellipticity, isophotal area -- is measured *from the y image the
model is not allowed to see*. Including them would not be a model, it would be
a lookup. ``EXCLUDE_PREFIXES`` enforces this and ``tests/test_m1.py`` checks
that no feature name starts with ``y_``.

Redshift is allowed: it is a spectroscopic measurement from an external survey,
available for every galaxy at prediction time, and B3 and P1 already use it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

#: Column prefixes that would leak the target. Everything measured from the
#: y image is off limits; so is the y magnitude itself.
EXCLUDE_PREFIXES: tuple[str, ...] = ("y_",)

#: Columns that are identifiers or coordinates rather than physics. Sky
#: position predicts nothing causal about a galaxy's spectrum -- including it
#: would let the model memorise patches of sky, which inflates test scores on
#: a survey with spatially clustered splits.
EXCLUDE_EXACT: tuple[str, ...] = (
    "object_id", "skymap_id", "coord", "specz_name",
    "ra", "dec", "specz_ra", "specz_dec",
    "x_coord", "y_coord", "x_coord_x", "x_coord_y", "y_coord_x", "y_coord_y",
    "specz_flag_homogeneous",
)

#: The target.
TARGET_COLUMN = "y_cmodel_mag"


def feature_columns(raw_path: Path) -> list[str]:
    """Every usable catalogue column, in a stable order."""
    with h5py.File(raw_path, "r") as f:
        keys = sorted(f.keys())
        out = []
        for k in keys:
            if k == "image" or k in EXCLUDE_EXACT:
                continue
            if any(k.startswith(p) for p in EXCLUDE_PREFIXES):
                continue
            d = f[k]
            if d.ndim != 1 or d.dtype.kind not in "fiu":
                continue
            out.append(k)
    return out


def load_split(raw_path: Path, processed_path: Path,
               columns: list[str] | None = None) -> dict:
    """Features and target for one split, aligned to the processed ordering.

    Returns ``X`` (n, n_features), ``y`` (n,), the column names, the object
    ids, and the redshift -- the last so results can be sliced by redshift the
    same way the image models' are.
    """
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    if columns is None:
        columns = feature_columns(raw_path)

    with h5py.File(processed_path, "r") as f:
        want_ids = np.asarray(f["object_id"][:])
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)

    with h5py.File(raw_path, "r") as f:
        raw_ids = np.asarray(f["object_id"][:])
        order = np.argsort(raw_ids)
        sorted_ids = raw_ids[order]
        pos = np.searchsorted(sorted_ids, want_ids)
        if np.any(pos >= len(sorted_ids)):
            raise KeyError("some processed object_ids are beyond the raw range; "
                           "the two files do not describe the same sample.")
        row = order[pos]
        if not np.array_equal(raw_ids[row], want_ids):
            missing = int((raw_ids[row] != want_ids).sum())
            raise KeyError(
                f"{missing} processed object_ids were not found in "
                f"{raw_path.name}. The metadata model would then be trained on "
                "a different set of galaxies than the image models, and the "
                "comparison would be meaningless."
            )
        X = np.stack([np.asarray(f[c][:], dtype=np.float64)[row]
                      for c in columns], axis=1)
        y = np.asarray(f[TARGET_COLUMN][:], dtype=np.float64)[row]

    return {"X": X, "y": y, "columns": list(columns),
            "object_id": want_ids, "redshift": z}


def fit_standardiser(X: np.ndarray) -> dict:
    """Per-column mean and std, from the training split only.

    Same discipline as the image pipeline: statistics never see validation or
    test data. Columns with zero variance get a std of 1 so they become
    constant zeros rather than NaNs.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_standardiser(X: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    return (X - mean) / std


def sanity_report(data: dict) -> dict:
    """What the features look like, for the run header."""
    X = data["X"]
    nonfinite = int((~np.isfinite(X)).sum())
    return {
        "n_galaxies": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "nonfinite_values": nonfinite,
        "target_median": float(np.median(data["y"])),
        "target_range": [float(np.min(data["y"])), float(np.max(data["y"]))],
        "leakage_check_no_y_columns": all(
            not c.startswith("y_") for c in data["columns"]),
    }
