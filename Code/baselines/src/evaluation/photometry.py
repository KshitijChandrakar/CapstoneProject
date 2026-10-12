"""Aperture photometry: flux error, magnitude residuals, colour fidelity.

These are the primary metrics. A prediction can look good pixel-by-pixel and
still get a galaxy's brightness wrong, which is what actually matters
scientifically.
"""

from __future__ import annotations

import numpy as np

NMAD_FACTOR: float = 1.4826


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nmad(x: np.ndarray, factor: float = NMAD_FACTOR) -> float:
    """Normalised median absolute deviation: 1.4826 * median(|x - median(x)|).

    A robust stand-in for the standard deviation: a few catastrophic galaxies
    cannot inflate it the way they inflate a plain std.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(factor * np.median(np.abs(x - np.median(x))))


def robust_stats(x: np.ndarray, factor: float = NMAD_FACTOR) -> dict:
    """median / NMAD / mean / std / count of a residual array."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": float("nan"), "nmad": float("nan"),
                "mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "median": float(np.median(x)),
        "nmad": nmad(x, factor),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "n": int(x.size),
    }


def circular_aperture_mask(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> np.ndarray:
    """Pixels whose centre lies within `radius` of `center`."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = center
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def aperture_flux(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum of pixels inside the aperture, per galaxy. NaN pixels are skipped."""
    images = np.asarray(images, dtype=np.float64)
    if images.ndim == 4:
        if images.shape[1] != 1:
            raise ValueError(f"Expected one channel, got {images.shape}")
        images = images[:, 0]
    return np.nansum(np.where(mask[None, :, :], images, 0.0), axis=(1, 2))


def flux_to_mag(
    flux: np.ndarray,
    zeropoint: float = 27.0,
    handling: str = "exclude",
    clamp_floor: float = 1e-30,
) -> np.ndarray:
    """m = -2.5*log10(flux) + zeropoint.

    A non-positive aperture flux has no magnitude. Two ways to deal with it:

    ``exclude``
        return NaN, so the galaxy drops out of the magnitude statistics and
        gets counted as an exclusion. **Recommended.**
    ``clamp``
        the spec's ``max(flux, 1e-30)``, which turns a faint non-detection into
        a ~75 mag error. That is an artefact of the clamp, not a property of
        the model, and it wrecks the NMAD and outlier fraction.
    """
    flux = np.asarray(flux, dtype=np.float64)
    if handling == "clamp":
        return -2.5 * np.log10(np.maximum(flux, clamp_floor)) + zeropoint
    if handling != "exclude":
        raise ValueError(f"handling must be 'exclude' or 'clamp', got {handling!r}")
    out = np.full(flux.shape, np.nan, dtype=np.float64)
    pos = flux > 0
    out[pos] = -2.5 * np.log10(flux[pos]) + zeropoint
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def photometry_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_true: np.ndarray,
    aperture_radius: float = 15.0,
    aperture_center: tuple[float, float] | None = None,
    zeropoint: float = 27.0,
    outlier_threshold: float = 0.5,
    nmad_factor: float = NMAD_FACTOR,
    handling: str = "exclude",
    clamp_floor: float = 1e-30,
    return_arrays: bool = False,
) -> dict:
    """Aperture flux error, magnitude residuals and z-y colour error.

    `z_true` is the measured z-band image, so the colour error uses a real z
    measurement on both sides.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    shape = (y_true.shape[-2], y_true.shape[-1])
    if aperture_center is None:
        c = float(shape[-1] // 2)
        aperture_center = (c, c)
    mask = circular_aperture_mask(shape, aperture_center, aperture_radius)

    f_true = aperture_flux(y_true, mask)
    f_pred = aperture_flux(y_pred, mask)
    f_z = aperture_flux(z_true, mask)

    # -- relative flux error -----------------------------------------------
    rel = np.full(f_true.shape, np.nan)
    ok = np.isfinite(f_true) & np.isfinite(f_pred) & (f_true != 0)
    rel[ok] = (f_pred[ok] - f_true[ok]) / f_true[ok]

    # -- magnitudes ---------------------------------------------------------
    m_true = flux_to_mag(f_true, zeropoint, handling, clamp_floor)
    m_pred = flux_to_mag(f_pred, zeropoint, handling, clamp_floor)
    dmag = m_pred - m_true

    n_excl_true = int(np.count_nonzero(f_true <= 0))
    n_excl_pred = int(np.count_nonzero(f_pred <= 0))
    n_excluded = int(np.count_nonzero(~np.isfinite(dmag)))

    dmag_stats = robust_stats(dmag, nmad_factor)
    finite_dmag = dmag[np.isfinite(dmag)]
    outlier_pct = (
        float(np.mean(np.abs(finite_dmag) > outlier_threshold) * 100.0)
        if finite_dmag.size else float("nan")
    )

    # -- colour error -------------------------------------------------------
    # Same real z flux on both sides, so d(z-y) = -(m_pred - m_true) = -dmag.
    m_z = flux_to_mag(f_z, zeropoint, handling, clamp_floor)
    dcolour = (m_z - m_pred) - (m_z - m_true)

    # -- faint vs bright ----------------------------------------------------
    # Any constant offset in a prediction is a large fraction of a faint
    # galaxy's total flux, so this is where these baselines break.
    by_brightness: dict = {}
    usable = f_true[np.isfinite(f_true) & (f_true > 0)]
    if usable.size >= 4:
        cutoff = float(np.median(usable))
        faint = (f_true > 0) & (f_true <= cutoff) & np.isfinite(dmag)
        bright = (f_true > cutoff) & np.isfinite(dmag)
        by_brightness = {
            "split_on": "median true aperture flux",
            "median_true_flux": cutoff,
            "faint_half": robust_stats(dmag[faint], nmad_factor),
            "bright_half": robust_stats(dmag[bright], nmad_factor),
        }

    out = {
        "aperture": {
            "radius_px": float(aperture_radius),
            "center": [float(v) for v in aperture_center],
            "n_pixels": int(mask.sum()),
        },
        "nonpositive_flux_handling": handling,
        "relative_flux_error": robust_stats(rel, nmad_factor),
        "delta_mag": {
            **dmag_stats,
            "outlier_percent": outlier_pct,
            "outlier_threshold": outlier_threshold,
            "zeropoint": zeropoint,
            "n_excluded_nonpositive_true_flux": n_excl_true,
            "n_excluded_nonpositive_pred_flux": n_excl_pred,
            "n_excluded_total": n_excluded,
        },
        "colour_error_z_minus_y": {
            **robust_stats(dcolour, nmad_factor),
            "note": ("Same measured z flux on both sides, so this equals "
                     "-delta_mag exactly."),
            "n_nonpositive_z_flux": int(np.count_nonzero(f_z <= 0)),
        },
        "delta_mag_by_brightness": by_brightness,
    }

    if return_arrays:
        out["_arrays"] = {
            "flux_true": f_true, "flux_pred": f_pred, "flux_z": f_z,
            "rel_flux_error": rel, "dmag": dmag, "dcolour": dcolour,
        }
    return out


def compare_flux_handling(
    y_true: np.ndarray, y_pred: np.ndarray, z_true: np.ndarray, **kwargs
) -> dict:
    """Run the magnitude statistics both ways, to show what clamping costs.

    The spec says to clamp non-positive fluxes to 1e-30; doing so converts a
    faint non-detection into a ~75 mag error. Reporting both makes the effect
    visible instead of hidden.
    """
    kwargs.pop("handling", None)
    kwargs.pop("return_arrays", None)
    out = {}
    for mode in ("exclude", "clamp"):
        r = photometry_metrics(y_true, y_pred, z_true, handling=mode, **kwargs)
        out[mode] = {
            "delta_mag": r["delta_mag"],
            "colour_error_z_minus_y": r["colour_error_z_minus_y"],
        }
    return out
