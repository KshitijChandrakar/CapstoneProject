"""Which spectral features sit inside which filter, at a given redshift.

**Not part of the P1/P2 deliverable.** The coursework specifies two physics
models, P1 (PSF-aware) and P2 (SED extrapolation). This module is kept because
it encodes script 05's emission-line result and reproduces the crossing windows
the results section quotes -- it is analysis support, and the starting point if
a line-aware model is ever wanted. Nothing in P1 or P2 imports it.

It exists because of a measurement and a failure.

**The measurement** (script 05): refitting B0 independently per redshift bin
showed the z->y relation really does change with redshift, and it changes in
the windows where a strong emission line crosses between the two filters --
[OIII]+Hb with an excess of +0.28 (95% CI [+0.21, +0.40]) and Ha with +0.20
(CI [+0.10, +0.29]). So redshift carries real, measurable information about how
y relates to z.

**The failure** (B3): conditioning a U-Net on redshift through FiLM moved the
prediction by about 1% -- 0.0109, 0.0124 and 0.0092 across three seeds. Raw
redshift, handed to a network as a scalar, bought essentially nothing.

Those two facts are only consistent one way: **the information is in the
spectral geometry, not in the number z itself.** A model given z has to learn
the rest-frame wavelength mapping, the line list, and the filter edges from
85k galaxies before redshift means anything. A model given "Ha is currently
inside the z filter and outside y" is handed the physics directly.

That is what this module computes: the physically meaningful
parameterisation of redshift, as opposed to the raw scalar FiLM could not use.
"""

from __future__ import annotations

import numpy as np

from src.config import BANDS
from src.physics.bands import LAMBDA_A, LAMBDA_EFF_A

#: Rest-frame wavelengths of the features that matter over 0 < z < 4.
#:
#: Ha, [OIII] and Hb are the strong optical emission lines that script 05
#: predicted windows for and then found. The 4000 A break is a continuum
#: feature, not a line -- included because it is the strongest break in a
#: galaxy spectrum and script 05 tested it (and found it *not* significant,
#: +0.14 with CI [-0.00, +0.26], which is why it is flagged below rather than
#: trusted).
REST_LINES_A: dict[str, float] = {
    "Ha": 6563.0,
    "[OIII]": 5007.0,
    "Hb": 4861.0,
    "4000A_break": 4000.0,
}

#: Features whose effect script 05 actually confirmed at 95% confidence.
#: 4000A_break is deliberately absent: it was tested and came back not
#: significant, and a model that weights it as though it were confirmed would
#: be encoding a result we do not have.
CONFIRMED_FEATURES: tuple[str, ...] = ("Ha", "[OIII]", "Hb")

#: Half-width used to decide "inside" a filter, in Angstroms.
#:
#: HSC filters are roughly 1000-1400 A wide (FWHM), so half-width ~600 A is a
#: reasonable top-hat stand-in for a real transmission curve. It is a
#: deliberate approximation: the alternative is shipping five transmission
#: curves, and the quantity being built is a soft feature for a network, not a
#: synthetic photometry pipeline. Tests pin the consequences.
FILTER_HALF_WIDTH_A: float = 600.0


def observed_wavelength(rest_A, redshift):
    """Where a rest-frame feature lands for an observer, lambda_obs = lambda_rest (1+z)."""
    rest = np.asarray(rest_A, dtype=np.float64)
    z = np.asarray(redshift, dtype=np.float64)
    if np.any(z <= -1.0):
        raise ValueError(f"redshift <= -1 is unphysical; got {float(np.min(z)):.4f}")
    return rest * (1.0 + z)


def redshift_entering(rest_A: float, band: str) -> float:
    """The redshift at which a feature arrives at a filter's centre.

    Reproduces the window boundaries script 05 documented, e.g. Ha reaching z
    at redshift 0.36 and y at 0.49.
    """
    if band not in LAMBDA_EFF_A:
        raise KeyError(f"unknown band {band!r}; expected one of {list(BANDS)}")
    return LAMBDA_EFF_A[band] / float(rest_A) - 1.0


def line_in_band(rest_A, redshift, half_width_A: float = FILTER_HALF_WIDTH_A):
    """Soft indicator in [0, 1] for "this feature is inside this filter".

    Shape ``(n_galaxies, n_bands)``. The value is a smooth triangular ramp in
    wavelength rather than a hard 0/1 step, for two reasons: a real filter does
    not have vertical edges, and a step function has zero gradient everywhere,
    so a network conditioned on it could never learn how much being near an
    edge matters.
    """
    z = np.atleast_1d(np.asarray(redshift, dtype=np.float64))
    obs = observed_wavelength(rest_A, z)                  # (n,)
    distance = np.abs(obs[:, None] - LAMBDA_A[None, :])   # (n, 5)
    return np.clip(1.0 - distance / float(half_width_A), 0.0, 1.0)


def feature_matrix(redshift, features: tuple[str, ...] = CONFIRMED_FEATURES,
                   half_width_A: float = FILTER_HALF_WIDTH_A) -> np.ndarray:
    """Per-galaxy physical conditioning vector, shape ``(n, len(features) * 5)``.

    One soft indicator per (feature, filter) pair: "how far inside filter b is
    feature f for this galaxy". The physically meaningful form of
    redshift, as opposed to the raw scalar FiLM could not use.

    Only the features script 05 confirmed are included by default. Passing the
    4000 A break explicitly is allowed -- it is an ablation, not the default.
    """
    unknown = [f for f in features if f not in REST_LINES_A]
    if unknown:
        raise KeyError(f"unknown feature(s) {unknown}; known: "
                       f"{sorted(REST_LINES_A)}")
    blocks = [line_in_band(REST_LINES_A[f], redshift, half_width_A)
              for f in features]
    return np.concatenate(blocks, axis=1)


def feature_matrix_columns(
    features: tuple[str, ...] = CONFIRMED_FEATURES,
) -> list[str]:
    """Human-readable column names for ``feature_matrix``, same order."""
    return [f"{f}_in_{b}" for f in features for b in BANDS]


def crossing_windows(rest_A: float, blue: str = "z", red: str = "y") -> dict:
    """The redshift range over which a feature sits between two filters.

    Between ``redshift_entering(rest, blue)`` and ``redshift_entering(rest,
    red)`` the feature boosts the bluer filter but not the redder one, so the
    flux ratio between them departs from its usual value. That is exactly the
    excess script 05 measured, and this returns the window it was measured in.
    """
    z_blue = redshift_entering(rest_A, blue)
    z_red = redshift_entering(rest_A, red)
    lo, hi = sorted((z_blue, z_red))
    return {
        "rest_A": float(rest_A),
        "enters_blue_at_z": z_blue,
        "enters_red_at_z": z_red,
        "window": (lo, hi),
        "width": hi - lo,
        "blue": blue,
        "red": red,
    }


def all_windows(features: tuple[str, ...] = tuple(REST_LINES_A),
                blue: str = "z", red: str = "y") -> list[dict]:
    """Every feature's z->y crossing window, for figures and log headers."""
    out = []
    for name in features:
        w = crossing_windows(REST_LINES_A[name], blue, red)
        w["feature"] = name
        w["confirmed_by_script_05"] = name in CONFIRMED_FEATURES
        out.append(w)
    return sorted(out, key=lambda w: w["window"][0])


def summary() -> str:
    """Log header: the windows, and which of them the data actually supports."""
    lines = ["feature       rest     enters z   enters y   window",
             "-" * 56]
    for w in all_windows():
        mark = "" if w["confirmed_by_script_05"] else "   (not significant)"
        lines.append(
            f"{w['feature']:<12} {w['rest_A']:>6.0f} A  "
            f"{w['enters_blue_at_z']:>7.2f}   {w['enters_red_at_z']:>7.2f}   "
            f"{w['window'][0]:.2f}-{w['window'][1]:.2f}{mark}"
        )
    return "\n".join(lines)
