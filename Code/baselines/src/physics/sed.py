"""Predicting y as a spectrum sampled at 9762 A, rather than as a 5th channel.

The physical statement behind P2: a pixel's five band fluxes are five samples
of one underlying spectral energy distribution. Predicting y means evaluating
that SED at 9762 A -- and since 9762 A is redward of every input, it is an
**extrapolation off the red end**, driven by the spectral slope there.

That is a different operation from what B1 and B2 do. B1 learns a fixed
weighted average; B2 learns an arbitrary function of the neighbourhood. Neither
is told that the four numbers are ordered in wavelength at all -- permute the
input channels and retrain, and both would reach the same loss. A model built
on this module cannot be permuted, because the wavelengths are in the
arithmetic.

### Why not a power law in log flux

The textbook move is F ~ lambda^alpha, fitted as a straight line in
(log lambda, log F). It cannot be used here without care: these are
**sky-subtracted** fluxes, so faint pixels are legitimately negative, and
log F is undefined for roughly every sky pixel in every cutout. Clamping them
at a small positive number would manufacture enormous fake slopes exactly where
the signal is weakest.

So the default works in (log lambda, F) -- linear in flux, log in wavelength.
It is well defined for negative flux, exact for the only extrapolation that
matters (864 A, about 9% in wavelength, where any smooth SED is locally
linear), and it degrades to "copy the red end" rather than to nonsense when the
pixel is pure noise. ``power_law=True`` enables the log-log form for the
pixels where every band used is positive, and falls back per pixel elsewhere;
the tests check that fallback actually happens.
"""

from __future__ import annotations

import numpy as np

from src.config import BANDS, INPUT_BAND_IDX, TARGET_BAND_IDX
from src.physics.bands import LAMBDA_A, LAMBDA_TARGET_A

#: Band-axis indices of the two reddest inputs, i and z, which carry the slope
#: that the extrapolation to y actually rides on.
RED_PAIR_IDX: tuple[int, int] = (BANDS.index("i"), BANDS.index("z"))


def _as_band_first(images: np.ndarray) -> np.ndarray:
    if images.ndim < 1:
        raise ValueError("expected an array with a band axis")
    return images


def spectral_slope(images: np.ndarray, pair: tuple[int, int] = RED_PAIR_IDX,
                   band_axis: int = 0) -> np.ndarray:
    """dF / dlog10(lambda) between two bands, per pixel.

    ``images`` has a band axis of length 5 (or 4 if only inputs are passed, in
    which case ``pair`` must index into that). The slope is a plain finite
    difference, which for two points is exact rather than a fit.
    """
    arr = _as_band_first(np.asarray(images, dtype=np.float64))
    blue_i, red_i = pair
    blue = np.take(arr, blue_i, axis=band_axis)
    red = np.take(arr, red_i, axis=band_axis)
    d_log_lambda = np.log10(LAMBDA_A[red_i]) - np.log10(LAMBDA_A[blue_i])
    if d_log_lambda == 0.0:
        raise ValueError("the two bands have the same wavelength; no slope")
    return (red - blue) / d_log_lambda


def extrapolate_to_target(
    images: np.ndarray,
    pair: tuple[int, int] = RED_PAIR_IDX,
    band_axis: int = 0,
    target_A: float = LAMBDA_TARGET_A,
    power_law: bool = False,
) -> np.ndarray:
    """Zero-parameter physical prediction of the target band.

    Continues the spectrum from the redder of ``pair`` out to ``target_A``
    along the measured slope. There is nothing to train: this is the physics
    baseline P2 is built on and must beat, in the same way B0 is the physics
    baseline for copying z.
    """
    arr = _as_band_first(np.asarray(images, dtype=np.float64))
    blue_i, red_i = pair
    red = np.take(arr, red_i, axis=band_axis)
    step = np.log10(target_A) - np.log10(LAMBDA_A[red_i])

    if not power_law:
        return red + spectral_slope(arr, pair, band_axis) * step

    # Log-log form, only where it is defined. A pixel with a non-positive flux
    # in either band has no power-law slope, so it keeps the linear answer
    # instead of being clamped into a fabricated one.
    blue = np.take(arr, blue_i, axis=band_axis)
    linear = red + spectral_slope(arr, pair, band_axis) * step
    usable = (blue > 0.0) & (red > 0.0)
    if not np.any(usable):
        return linear

    with np.errstate(divide="ignore", invalid="ignore"):
        d_log_lambda = np.log10(LAMBDA_A[red_i]) - np.log10(LAMBDA_A[blue_i])
        alpha = (np.log10(np.where(usable, red, 1.0))
                 - np.log10(np.where(usable, blue, 1.0))) / d_log_lambda
        powered = np.where(usable, red, 0.0) * 10.0 ** (alpha * step)
    return np.where(usable, powered, linear)


def design_matrix(n_bands: int = len(INPUT_BAND_IDX),
                  degree: int = 2,
                  normalise: bool = True) -> np.ndarray:
    """Polynomial basis in log wavelength over the input bands.

    Shape ``(n_bands, degree + 1)``. A model multiplies this by learned
    coefficients to get a smooth SED through the observed points, then
    evaluates that curve at the target wavelength -- so the smoothness is
    imposed by construction rather than hoped for.

    Degree 2 is the default because four input bands cannot constrain more than
    a quadratic without fitting their own noise exactly, leaving nothing to
    extrapolate with.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if degree > n_bands - 1:
        raise ValueError(
            f"degree {degree} over {n_bands} bands would interpolate the "
            "inputs exactly and extrapolate on noise; use at most "
            f"{n_bands - 1}."
        )
    lam = LAMBDA_A[list(INPUT_BAND_IDX)][:n_bands]
    x = np.log10(lam)
    if normalise:
        ref = np.log10(LAMBDA_A)
        x = (x - ref.mean()) / ref.std()
    return np.vander(x, degree + 1, increasing=True)


def target_vector(degree: int = 2, normalise: bool = True) -> np.ndarray:
    """The same basis evaluated at the target wavelength, shape ``(degree+1,)``.

    ``coefficients @ target_vector()`` is the predicted y flux. Keeping this
    beside ``design_matrix`` is deliberate: the two must share a normalisation
    or the extrapolation silently lands at the wrong wavelength, and the tests
    check exactly that by recovering a known quadratic.
    """
    x = np.log10(LAMBDA_TARGET_A)
    if normalise:
        ref = np.log10(LAMBDA_A)
        x = (x - ref.mean()) / ref.std()
    return np.vander(np.array([x]), degree + 1, increasing=True)[0]


def fit_sed_and_predict(images: np.ndarray, degree: int = 2,
                        band_axis: int = 0, ridge: float = 1e-8) -> np.ndarray:
    """Least-squares SED fit through the four inputs, evaluated at y.

    The reference implementation of P2's forward operation with no learned
    parameters at all, used to sanity-check the trainable version: if a model
    with two million parameters cannot beat a per-pixel quadratic fit, the
    parameters are not buying physics.

    ``ridge`` only keeps the normal equations conditioned; at 1e-8 it does not
    meaningfully bias the fit.
    """
    arr = _as_band_first(np.asarray(images, dtype=np.float64))
    bands = np.take(arr, list(INPUT_BAND_IDX), axis=band_axis)
    moved = np.moveaxis(bands, band_axis, -1)          # (..., 4)
    flat = moved.reshape(-1, moved.shape[-1])          # (n_pixels, 4)

    A = design_matrix(degree=degree)                   # (4, degree+1)
    gram = A.T @ A + ridge * np.eye(A.shape[1])
    coefficients = flat @ A @ np.linalg.inv(gram).T    # (n_pixels, degree+1)
    predicted = coefficients @ target_vector(degree=degree)
    return predicted.reshape(moved.shape[:-1])


def summary() -> str:
    """Log header describing the extrapolation this module performs."""
    blue_i, red_i = RED_PAIR_IDX
    return (
        f"SED extrapolation: slope measured between {BANDS[blue_i]} "
        f"({LAMBDA_A[blue_i]:.0f} A) and {BANDS[red_i]} "
        f"({LAMBDA_A[red_i]:.0f} A), continued to {BANDS[TARGET_BAND_IDX]} "
        f"({LAMBDA_TARGET_A:.0f} A) -- a step of "
        f"{LAMBDA_TARGET_A - LAMBDA_A[red_i]:.0f} A beyond the reddest input. "
        "Linear in flux, log in wavelength, because sky-subtracted fluxes are "
        "legitimately negative and log F is undefined for most sky pixels."
    )
