"""Checks on the physics layer that P1-P3 are built from.

    python tests/test_physics.py

These are the facts the physics models assume. If any of them breaks, the
models stop being physics-informed and become arbitrary arithmetic that happens
to run -- which is the one failure mode that would not show up as a worse loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import BANDS, INPUT_BAND_IDX, TARGET_BAND_IDX
from src.physics import bands, lines, sed

PASS, FAIL = 0, 0


def check(label: str, got, want=True, tol: float | None = None) -> None:
    global PASS, FAIL
    if tol is not None:
        ok = bool(np.all(np.abs(np.asarray(got, dtype=float)
                                - np.asarray(want, dtype=float)) <= tol))
    else:
        ok = bool(got == want) if not isinstance(got, np.ndarray) else bool(
            np.array_equal(got, want))
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


# ------------------------------------------------------------ band geometry
def test_wavelengths_are_ordered_and_match_script_05() -> None:
    print("\nfilter wavelengths")
    check("one wavelength per band", len(bands.LAMBDA_A), len(BANDS))
    check("wavelengths increase g -> y",
          bool(np.all(np.diff(bands.LAMBDA_A) > 0)))
    # Script 05 has used these two since the redshift sweep; the published
    # windows depend on them, so they are not free to drift.
    check("z is 8898 A, as script 05 assumes", bands.LAMBDA_EFF_A["z"], 8898.0)
    check("y is 9762 A, as script 05 assumes", bands.LAMBDA_EFF_A["y"], 9762.0)
    check("the target index really is y", BANDS[TARGET_BAND_IDX], "y")


def test_target_is_an_extrapolation() -> None:
    """The whole problem's shape. If this ever reads False, y leaked inputward."""
    print("\nthe target sits off the red end")
    geo = bands.extrapolation_baseline()
    check("y is redder than every input", geo["is_extrapolation"])
    check("the gap is 864 A", geo["gap_A"], 864.0, tol=0.5)
    check("that is ~9.7% in wavelength", geo["gap_fraction"], 0.0971, tol=5e-4)
    check("y is not among the inputs", TARGET_BAND_IDX not in INPUT_BAND_IDX)
    check("exactly four inputs", len(bands.LAMBDA_IN_A), 4)


def test_rest_frame_mapping() -> None:
    print("\nrest-frame wavelengths")
    check("at z=0 rest equals observed",
          bands.rest_wavelength(9762.0, 0.0), 9762.0, tol=1e-9)
    check("at z=1 rest is half observed",
          bands.rest_wavelength(9762.0, 1.0), 4881.0, tol=1e-9)
    grid = bands.rest_frame_grid([0.0, 1.0])
    check("grid shape is (galaxies, bands)", grid.shape, (2, len(BANDS)))
    check("row 0 is the observed set", grid[0], bands.LAMBDA_A, tol=1e-9)
    try:
        bands.rest_wavelength(5000.0, -1.5)
        check("z <= -1 is rejected", False)
    except ValueError:
        check("z <= -1 is rejected", True)


# ------------------------------------------------------------- line physics
def test_crossing_windows_reproduce_the_published_ones() -> None:
    """Independently derived here; script 05 hard-codes them in its docstring."""
    print("\nemission-line crossing windows")
    expected = {           # from 05_redshift_sweep.py's header
        "Ha": (0.36, 0.49),
        "[OIII]": (0.78, 0.95),
        "Hb": (0.83, 1.01),
        "4000A_break": (1.22, 1.44),
    }
    for w in lines.all_windows():
        lo, hi = expected[w["feature"]]
        check(f"{w['feature']} enters z at z={lo}",
              w["enters_blue_at_z"], lo, tol=0.005)
        check(f"{w['feature']} enters y at z={hi}",
              w["enters_red_at_z"], hi, tol=0.005)


def test_the_unconfirmed_feature_is_excluded_by_default() -> None:
    """Script 05 tested the 4000 A break and it was NOT significant."""
    print("\nonly confirmed features condition the model")
    check("4000A break is not in the default feature set",
          "4000A_break" not in lines.CONFIRMED_FEATURES)
    check("the three confirmed lines are",
          lines.CONFIRMED_FEATURES, ("Ha", "[OIII]", "Hb"))
    cols = lines.feature_matrix_columns()
    check("default conditioning is 3 features x 5 bands", len(cols), 15)
    check("no 4000A column by default",
          all("4000" not in c for c in cols))
    # It can still be requested, as an ablation.
    wide = lines.feature_matrix([0.5], features=("Ha", "4000A_break"))
    check("it is available when asked for explicitly", wide.shape, (1, 10))


def test_line_indicator_peaks_in_the_right_band() -> None:
    print("\nthe indicator fires on the right filter")
    # Ha enters z at z=0.356: at that redshift it should peak on z.
    z_enter = lines.redshift_entering(lines.REST_LINES_A["Ha"], "z")
    row = lines.line_in_band(lines.REST_LINES_A["Ha"], [z_enter])[0]
    check("Ha peaks on the z filter at its entering redshift",
          BANDS[int(np.argmax(row))], "z")
    check("and the peak is exactly 1.0", float(row.max()), 1.0, tol=1e-9)

    z_enter_y = lines.redshift_entering(lines.REST_LINES_A["Ha"], "y")
    row_y = lines.line_in_band(lines.REST_LINES_A["Ha"], [z_enter_y])[0]
    check("Ha peaks on y once it has moved there",
          BANDS[int(np.argmax(row_y))], "y")

    # At z=0 Ha sits at 6563 A, between r and i, inside neither by 600 A.
    row0 = lines.line_in_band(lines.REST_LINES_A["Ha"], [0.0])[0]
    check("at z=0 Ha is in no filter strongly", float(row0.max()) < 0.4)


def test_indicator_is_smooth_not_a_step() -> None:
    """A hard 0/1 indicator has no gradient, so nothing could learn from it."""
    print("\nthe indicator is differentiable")
    z_peak = lines.redshift_entering(lines.REST_LINES_A["Ha"], "z")
    zs = np.linspace(0.30, 0.42, 25)
    vals = lines.line_in_band(lines.REST_LINES_A["Ha"], zs)[:, BANDS.index("z")]
    check("it takes intermediate values, not just 0 and 1",
          bool(np.any((vals > 0.05) & (vals < 0.95))))
    check("it is bounded to [0, 1]",
          bool(np.all((vals >= 0.0) & (vals <= 1.0))))

    # The ramp peaks when the line sits on the filter centre and falls away
    # again as it passes beyond -- the line moves *through* the filter, it does
    # not arrive and stay. So monotonic rise holds only up to the peak.
    rising = zs <= z_peak
    falling = zs >= z_peak
    check("it rises as the line approaches the filter centre",
          bool(np.all(np.diff(vals[rising]) >= -1e-12)))
    check("and falls again as the line moves past",
          bool(np.all(np.diff(vals[falling]) <= 1e-12)))
    check("the maximum sits at the entering redshift",
          float(zs[int(np.argmax(vals))]), z_peak, tol=zs[1] - zs[0])


def test_feature_matrix_shape_and_order() -> None:
    print("\nconditioning matrix")
    zs = [0.1, 0.42, 0.88]
    m = lines.feature_matrix(zs)
    check("shape is (galaxies, features x bands)", m.shape, (3, 15))
    cols = lines.feature_matrix_columns()
    check("column names line up with the blocks", cols[:5],
          [f"Ha_in_{b}" for b in BANDS])
    # The galaxy at z=0.88 has [OIII] near z; the one at 0.1 has nothing.
    oiii_z = cols.index("[OIII]_in_z")
    check("[OIII] registers at z=0.88 and not at z=0.1",
          m[2, oiii_z] > m[0, oiii_z])
    try:
        lines.feature_matrix([0.3], features=("NotALine",))
        check("an unknown feature is rejected", False)
    except KeyError:
        check("an unknown feature is rejected", True)


# --------------------------------------------------------------- SED layer
def test_slope_and_extrapolation_are_exact_on_a_straight_line() -> None:
    print("\nSED extrapolation")
    # Build a pixel that is exactly linear in (log lambda, flux).
    lam = bands.LAMBDA_A
    slope, offset = 3.0, 1.5
    flux = offset + slope * (np.log10(lam) - np.log10(lam[0]))
    cube = flux[:, None, None] * np.ones((1, 2, 2))

    got = sed.spectral_slope(cube)
    check("the measured slope is the true one", got, np.full((2, 2), slope),
          tol=1e-9)

    pred = sed.extrapolate_to_target(cube)
    truth = flux[TARGET_BAND_IDX]
    check("extrapolating to y recovers the true y exactly",
          pred, np.full((2, 2), truth), tol=1e-9)


def test_extrapolation_handles_negative_flux() -> None:
    """Sky-subtracted pixels are legitimately negative; log F would explode."""
    print("\nnegative fluxes are allowed")
    lam = bands.LAMBDA_A
    flux = -0.4 + 0.2 * (np.log10(lam) - np.log10(lam[0]))
    cube = flux[:, None, None] * np.ones((1, 1, 1))
    pred = sed.extrapolate_to_target(cube)
    check("a wholly negative pixel still extrapolates finitely",
          bool(np.all(np.isfinite(pred))))
    check("and lands on the true value",
          pred, np.full((1, 1), flux[TARGET_BAND_IDX]), tol=1e-9)

    powered = sed.extrapolate_to_target(cube, power_law=True)
    check("power_law mode falls back rather than returning NaN",
          bool(np.all(np.isfinite(powered))))
    check("the fallback IS the linear answer", powered, pred, tol=1e-12)


def test_power_law_is_exact_on_a_power_law() -> None:
    print("\npower-law mode")
    lam = bands.LAMBDA_A
    alpha, norm = -1.7, 500.0
    flux = norm * (lam / lam[0]) ** alpha
    cube = flux[:, None, None] * np.ones((1, 1, 1))
    pred = sed.extrapolate_to_target(cube, power_law=True)
    check("it recovers the true y for a genuine power law",
          pred, np.full((1, 1), flux[TARGET_BAND_IDX]),
          tol=abs(flux[TARGET_BAND_IDX]) * 1e-9)


def test_design_matrix_and_target_share_a_normalisation() -> None:
    """The failure this guards: basis and target normalised differently, so the
    curve is evaluated at the wrong wavelength and nobody notices."""
    print("\nSED basis")
    for degree in (1, 2, 3):
        A = sed.design_matrix(degree=degree)
        check(f"degree {degree} basis is (4, {degree + 1})",
              A.shape, (4, degree + 1))

    # A known quadratic in normalised log-lambda must be recovered at y.
    degree = 2
    A = sed.design_matrix(degree=degree)
    coef = np.array([0.3, -1.1, 0.45])
    observed = A @ coef
    cube = np.concatenate([observed, [0.0]])[:, None, None] * np.ones((1, 1, 1))
    want = float(sed.target_vector(degree=degree) @ coef)
    got = sed.fit_sed_and_predict(cube, degree=degree)
    check("a fitted quadratic evaluates correctly at 9762 A",
          got, np.full((1, 1), want), tol=1e-6)

    try:
        sed.design_matrix(degree=4)
        check("a degree the 4 bands cannot support is rejected", False)
    except ValueError:
        check("a degree the 4 bands cannot support is rejected", True)


def test_sed_fit_ignores_the_target_channel() -> None:
    """Leakage check: y must never reach the predictor."""
    print("\nno target leakage")
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(len(BANDS), 4, 4))
    a = sed.fit_sed_and_predict(cube.copy())
    poisoned = cube.copy()
    poisoned[TARGET_BAND_IDX] += 1000.0
    b = sed.fit_sed_and_predict(poisoned)
    check("changing the y channel does not change the prediction",
          a, b, tol=1e-12)


def main() -> int:
    print("=" * 70)
    print("PHYSICS LAYER -- wavelengths, emission lines, SED extrapolation")
    print("=" * 70)
    test_wavelengths_are_ordered_and_match_script_05()
    test_target_is_an_extrapolation()
    test_rest_frame_mapping()
    test_crossing_windows_reproduce_the_published_ones()
    test_the_unconfirmed_feature_is_excluded_by_default()
    test_line_indicator_peaks_in_the_right_band()
    test_indicator_is_smooth_not_a_step()
    test_feature_matrix_shape_and_order()
    test_slope_and_extrapolation_are_exact_on_a_straight_line()
    test_extrapolation_handles_negative_flux()
    test_power_law_is_exact_on_a_power_law()
    test_design_matrix_and_target_share_a_normalisation()
    test_sed_fit_ignores_the_target_channel()
    print("\n" + "=" * 70)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
