"""Checks on M1, the catalogue-metadata network.

    python tests/test_m1.py

M1's result is only meaningful if it genuinely never sees the y band. Most of
these tests are about leakage, because that is the one failure that would make
the model look brilliant while meaning nothing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch
import yaml

from src.data.metadata import (
    EXCLUDE_EXACT,
    EXCLUDE_PREFIXES,
    TARGET_COLUMN,
    apply_standardiser,
    feature_columns,
    fit_standardiser,
    load_split,
    sanity_report,
)
from src.models.m1_metadata_nn import M1MetadataNN, build_m1, catalogue_baseline

PASS, FAIL = 0, 0


def check(label: str, got, want=True, tol: float | None = None) -> None:
    global PASS, FAIL
    if tol is not None:
        ok = bool(np.all(np.abs(np.asarray(got, dtype=float)
                                - np.asarray(want, dtype=float)) <= tol))
    else:
        ok = bool(got == want)
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def make_pair(tmp: Path, n: int = 40):
    """A tiny raw/processed pair that mimics the real files' structure."""
    rng = np.random.default_rng(0)
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    raw = tmp / "raw.hdf5"
    with h5py.File(raw, "w") as f:
        f.create_dataset("image", data=rng.normal(size=(n, 5, 4, 4)).astype(np.float32))
        f.create_dataset("object_id", data=ids)
        f.create_dataset("specz_redshift", data=rng.uniform(0, 0.4, n))
        f.create_dataset("specz_redshift_err", data=rng.uniform(1e-4, 1e-3, n))
        for b in ("g", "r", "i", "z", "y"):
            f.create_dataset(f"{b}_cmodel_mag", data=rng.uniform(16, 24, n))
            f.create_dataset(f"{b}_half_light_radius", data=rng.uniform(1, 9, n))
            f.create_dataset(f"{b}_sersic_index", data=rng.uniform(0.5, 6, n))
        for c in ("ra", "dec", "x_coord", "y_coord"):
            f.create_dataset(c, data=rng.uniform(0, 100, n))
        f.create_dataset("skymap_id", data=ids * 2)

    # The processed split keeps a shuffled subset, as the real one does.
    keep = rng.permutation(n)[: n // 2]
    proc = tmp / "proc.h5"
    with h5py.File(raw, "r") as src, h5py.File(proc, "w") as g:
        g.create_dataset("image", data=np.asarray(src["image"])[np.sort(keep)])
        g.create_dataset("object_id", data=ids[np.sort(keep)])
        g.create_dataset("specz_redshift",
                         data=np.asarray(src["specz_redshift"])[np.sort(keep)])
    return raw, proc, np.sort(keep)


# ------------------------------------------------------------------ leakage
def test_no_y_band_column_survives() -> None:
    """The failure that would make M1 a lookup table rather than a model."""
    print("\nleakage: the y band is excluded")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        raw, proc, _ = make_pair(tmp)
        cols = feature_columns(raw)
        check("the target itself is absent", TARGET_COLUMN not in cols)
        leaked = [c for c in cols if c.startswith("y_")]
        check(f"no y_* column at all (found {leaked})", leaked, [])
        check("but the other bands are present",
              all(any(c.startswith(b + "_") for c in cols)
                  for b in ("g", "r", "i", "z")))
        check("y_half_light_radius is excluded -- it is measured from the "
              "y image", "y_half_light_radius" not in cols)
        check("y_sersic_index likewise", "y_sersic_index" not in cols)


def test_identifiers_and_coordinates_are_excluded() -> None:
    """Sky position would let the model memorise patches of a clustered survey."""
    print("\nleakage: identifiers and sky position")
    with tempfile.TemporaryDirectory() as d:
        raw, _, _ = make_pair(Path(d))
        cols = feature_columns(raw)
        for bad in ("object_id", "skymap_id", "ra", "dec", "x_coord", "y_coord"):
            check(f"{bad} is not a feature", bad not in cols)
        check("the exclusion list is actually consulted",
              all(c not in cols for c in EXCLUDE_EXACT))


def test_rows_align_with_the_processed_split() -> None:
    """If the join misaligned, M1 would train on other galaxies' features."""
    print("\nthe join is row-exact")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        raw, proc, keep = make_pair(tmp)
        cols = feature_columns(raw)
        data = load_split(raw, proc, cols)
        with h5py.File(proc, "r") as f:
            want_ids = np.asarray(f["object_id"][:])
        with h5py.File(raw, "r") as f:
            raw_ids = np.asarray(f["object_id"][:])
            truth_mag = np.asarray(f[TARGET_COLUMN][:])
            gmag = np.asarray(f["g_cmodel_mag"][:])
        check("object ids are preserved in order",
              np.array_equal(data["object_id"], want_ids))
        lut = {int(i): k for k, i in enumerate(raw_ids)}
        rows = [lut[int(i)] for i in want_ids]
        check("the target matches the right galaxy",
              data["y"], truth_mag[rows], tol=0.0)
        gi = cols.index("g_cmodel_mag")
        check("a feature matches the right galaxy",
              data["X"][:, gi], gmag[rows], tol=0.0)
        check("X has one row per processed galaxy",
              data["X"].shape[0], len(want_ids))


def test_a_mismatched_join_is_refused() -> None:
    print("\na broken join raises rather than silently misaligning")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        raw, proc, _ = make_pair(tmp)
        with h5py.File(proc, "r+") as f:
            ids = np.asarray(f["object_id"][:])
            del f["object_id"]
            f.create_dataset("object_id", data=ids + 10_000_000)
        try:
            load_split(raw, proc, feature_columns(raw))
            check("unknown object ids are rejected", False)
        except KeyError:
            check("unknown object ids are rejected", True)


# ------------------------------------------------------------ standardising
def test_standardiser_uses_train_statistics_only() -> None:
    print("\nstandardising")
    rng = np.random.default_rng(1)
    Xtr = rng.normal(5.0, 3.0, size=(200, 4))
    stats = fit_standardiser(Xtr)
    out = apply_standardiser(Xtr, stats)
    check("train data becomes zero-mean", np.abs(out.mean(axis=0)).max() < 1e-12)
    check("and unit-variance", np.abs(out.std(axis=0) - 1).max() < 1e-12)

    Xte = rng.normal(9.0, 3.0, size=(50, 4))
    shifted = apply_standardiser(Xte, stats)
    check("test data is NOT re-centred on itself",
          np.abs(shifted.mean(axis=0)).max() > 0.5)

    const = np.ones((10, 2))
    s2 = fit_standardiser(const)
    check("a constant column does not divide by zero",
          bool(np.isfinite(apply_standardiser(const, s2)).all()))


# ------------------------------------------------------------------- model
def test_model_shape_and_counts() -> None:
    print("\nthe network")
    m = M1MetadataNN(59)
    check("16,001 parameters for 59 features at [128, 64]",
          m.count_parameters(), 16001)
    out = m(torch.randn(7, 59))
    check("output is one number per galaxy", tuple(out.shape), (7,))

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config"
                          / "m1_metadata.yaml").read_text(encoding="utf-8"))
    check("the config builds the same model",
          build_m1(cfg["model"], 59).count_parameters(), 16001)
    try:
        build_m1(dict(cfg["model"], expected_parameters=5), 59)
        check("a count mismatch is refused", False)
    except ValueError:
        check("a count mismatch is refused", True)
    try:
        M1MetadataNN(0)
        check("zero features is refused", False)
    except ValueError:
        check("zero features is refused", True)


def test_head_starts_at_the_training_median() -> None:
    """So training improves on "guess the middle", the honest floor."""
    print("\ninitialisation")
    m = M1MetadataNN(10, target_offset=17.6)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(16, 10))
    check("every prediction starts at the median magnitude",
          out.numpy(), np.full(16, 17.6), tol=1e-5)
    check("so the initial error is the spread of the target, not noise",
          float(out.std()) < 1e-6)


def test_it_can_learn_a_known_relation() -> None:
    print("\nit trains")
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(512, 6))
    y = 17.0 + 1.5 * X[:, 0] - 0.8 * X[:, 3]
    m = M1MetadataNN(6, hidden=(32,), dropout=0.0, target_offset=17.0)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    first = float((m(xt) - yt).abs().mean())
    for _ in range(400):
        opt.zero_grad()
        loss = (m(xt) - yt).abs().mean()
        loss.backward()
        opt.step()
    last = float((m(xt) - yt).abs().mean())
    check(f"L1 fell from {first:.3f} to {last:.3f}", last < first * 0.2)


def test_catalogue_baseline_is_a_real_reference() -> None:
    print("\nthe zero-parameter baseline")
    zmag = np.array([18.0, 19.0, 20.0])
    pred = catalogue_baseline(zmag, 0.18)
    check("it subtracts the median colour",
          pred, zmag - 0.18, tol=1e-12)
    check("and has no free parameters beyond that one number",
          pred.shape, zmag.shape)


def test_sanity_report_flags_leakage() -> None:
    print("\nthe sanity report")
    data = {"X": np.zeros((3, 2)), "y": np.array([1.0, 2.0, 3.0]),
            "columns": ["g_cmodel_mag", "r_cmodel_mag"],
            "object_id": np.arange(3), "redshift": np.zeros(3)}
    check("a clean feature set passes",
          sanity_report(data)["leakage_check_no_y_columns"], True)
    data["columns"] = ["g_cmodel_mag", "y_sersic_index"]
    check("a y_ column is caught",
          sanity_report(data)["leakage_check_no_y_columns"], False)


def main() -> int:
    print("=" * 70)
    print("M1 -- catalogue metadata network")
    print("=" * 70)
    test_no_y_band_column_survives()
    test_identifiers_and_coordinates_are_excluded()
    test_rows_align_with_the_processed_split()
    test_a_mismatched_join_is_refused()
    test_standardiser_uses_train_statistics_only()
    test_model_shape_and_counts()
    test_head_starts_at_the_training_median()
    test_it_can_learn_a_known_relation()
    test_catalogue_baseline_is_a_real_reference()
    test_sanity_report_flags_leakage()
    print("\n" + "=" * 70)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
