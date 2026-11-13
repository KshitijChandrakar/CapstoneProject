"""Subaru HSC filter wavelengths and the geometry of the prediction problem.

Everything physics-informed in P1-P3 starts here: the five filters are not
interchangeable channels, they are samples of a spectrum at known wavelengths,
and *y* sits at the red end just past *z*.

    g 4754 A   r 6175 A   i 7711 A   z 8898 A   y 9762 A

The z and y values match the ones script 05 has used since the redshift sweep;
g, r and i are the matching HSC-SSP effective wavelengths. They are gathered
here because three separate models and two scripts were about to need them, and
a constant duplicated in five files is a constant that will disagree in five
files.

Two facts set the whole problem:

* **y is an extrapolation, not an interpolation.** 9762 A is redward of every
  input band. No weighted average of g, r, i, z reaches it; the prediction has
  to come from the spectral slope at the red end, which is why the i-z pair
  carries so much of the information and why B1 gave z a standardised weight of
  +0.55.
* **z and y are only 864 A apart**, about 9% in wavelength. That closeness is
  why B0 works at all -- copying z gets within 6.5% of B1 -- and also why the
  remaining error is hard: the two filters see almost the same light, so what
  differs between them is subtle.
"""

from __future__ import annotations

import numpy as np

from src.config import BANDS, INPUT_BAND_IDX, TARGET_BAND_IDX

#: Effective wavelength per filter, Angstroms, indexed like ``BANDS``.
LAMBDA_EFF_A: dict[str, float] = {
    "g": 4754.0,
    "r": 6175.0,
    "i": 7711.0,
    "z": 8898.0,
    "y": 9762.0,
}

#: Same, as an array on the HDF5 band axis -- the order the images come in.
LAMBDA_A: np.ndarray = np.array(
    [LAMBDA_EFF_A[b] for b in BANDS], dtype=np.float64
)

#: The four inputs and the one target, split out for convenience.
LAMBDA_IN_A: np.ndarray = LAMBDA_A[list(INPUT_BAND_IDX)]
LAMBDA_TARGET_A: float = float(LAMBDA_A[TARGET_BAND_IDX])


def rest_wavelength(observed_A, redshift):
    """Rest-frame wavelength of light observed at ``observed_A``.

    lambda_rest = lambda_obs / (1 + z). Broadcasts, so a scalar wavelength and
    an array of redshifts give one value per galaxy.
    """
    observed = np.asarray(observed_A, dtype=np.float64)
    z = np.asarray(redshift, dtype=np.float64)
    if np.any(z <= -1.0):
        raise ValueError("redshift <= -1 has no rest frame; got min "
                         f"{float(np.min(z)):.4f}")
    return observed / (1.0 + z)


def rest_frame_grid(redshift):
    """The rest-frame wavelengths all five filters sample, per galaxy.

    Returns shape ``(n_galaxies, 5)`` in the HDF5 band order. This is the
    physical reason the z->y relation is not constant with redshift: the same
    filters look at different parts of the spectrum as the galaxy recedes, so
    whatever spectral feature sits between z and y changes.
    """
    z = np.atleast_1d(np.asarray(redshift, dtype=np.float64))
    return rest_wavelength(LAMBDA_A[None, :], z[:, None])


def log_lambda(normalise: bool = True) -> np.ndarray:
    """log10 of the five filter wavelengths, optionally centred and scaled.

    Models work in log wavelength because a galaxy spectrum is far closer to a
    power law in lambda than it is to linear. Normalising keeps the numbers
    O(1) so they can be fed to a network without dominating the first layer.
    """
    lg = np.log10(LAMBDA_A)
    if not normalise:
        return lg
    return (lg - lg.mean()) / lg.std()


def extrapolation_baseline() -> dict:
    """How far past the inputs the target sits, as plain numbers.

    Reported in the methods section, and used by the tests to catch anyone
    quietly turning the extrapolation into an interpolation by, for example,
    including y among the inputs.
    """
    reddest_input = float(LAMBDA_IN_A.max())
    return {
        "target_A": LAMBDA_TARGET_A,
        "reddest_input_A": reddest_input,
        "gap_A": LAMBDA_TARGET_A - reddest_input,
        "gap_fraction": (LAMBDA_TARGET_A - reddest_input) / reddest_input,
        "is_extrapolation": LAMBDA_TARGET_A > reddest_input,
        "input_span_A": (float(LAMBDA_IN_A.min()), reddest_input),
    }


def band_of_wavelength(observed_A) -> list[str | None]:
    """Which filter's effective wavelength is nearest, within half a gap.

    Used to say where an emission line lands. Returns None when the line falls
    outside the filter set rather than forcing it into the nearest band, since
    "nearest" is meaningless 3000 A away.
    """
    obs = np.atleast_1d(np.asarray(observed_A, dtype=np.float64))
    names = list(BANDS)
    out: list[str | None] = []
    for value in obs:
        gaps = np.abs(LAMBDA_A - value)
        k = int(np.argmin(gaps))
        # Half the distance to the nearest neighbouring filter, as a tolerance.
        if len(LAMBDA_A) > 1:
            neighbours = np.abs(np.diff(LAMBDA_A))
            tol = float(neighbours.min())
        else:                                   # pragma: no cover
            tol = np.inf
        out.append(names[k] if gaps[k] <= tol else None)
    return out


def summary() -> str:
    """One paragraph for a log header, so runs record their own geometry."""
    geo = extrapolation_baseline()
    bands = "  ".join(f"{b} {LAMBDA_EFF_A[b]:.0f} A" for b in BANDS)
    return (
        f"filters: {bands}\n"
        f"target y at {geo['target_A']:.0f} A is "
        f"{geo['gap_A']:.0f} A ({100 * geo['gap_fraction']:.1f}%) redward of "
        f"the reddest input (z at {geo['reddest_input_A']:.0f} A), so the "
        "prediction is an extrapolation off the end of the filter set."
    )
