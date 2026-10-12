"""Baseline B0 -- copy the z band, rescaled by the median train z-y colour.

Zero learnable parameters. This is the performance floor: a neural network
that cannot beat it has learned nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import BANDS
from src.data.preprocess import mag_is_valid


@dataclass
class CopyZBaseline:
    """y_hat(x, y) = z(x, y) * scale.

    The scale comes from the median ``m_z - m_y`` colour of the TRAIN split:

        scale = 10 ** (sign * 0.4 * median(m_z - m_y))

    **On the sign.** The spec sheet and slide 7 both write ``sign = -1``. That
    is wrong. A magnitude is ``m = -2.5*log10(F)``, so ``F = 10**(-0.4*m)`` and

        F_y / F_z = 10**(-0.4*m_y) / 10**(-0.4*m_z) = 10**(+0.4*(m_z - m_y))

    The dataset agrees: over all 204,573 training galaxies the team's EDA gives
    mean(m_z) = 19.325 and mean(m_y) = 19.154, so y is 0.171 mag *brighter*
    than z and the scale must exceed 1. The ``+`` sign gives 1.171; the ``-``
    sign gives 0.854, which would make the predicted y image dimmer than z and
    contradict the catalogue.

    :func:`verify_scale_sign` measures the real flux ratio from the images, so
    this is settled by evidence rather than by argument.
    """

    scale: float
    median_colour: float
    sign: int
    n_used: int
    n_dropped: int
    n_parameters: int = 0

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit_from_mags(
        cls, z_mag: np.ndarray, y_mag: np.ndarray, sign: int = +1
    ) -> "CopyZBaseline":
        """Fit the single constant from TRAIN-split catalogue magnitudes."""
        if sign not in (+1, -1):
            raise ValueError(f"sign must be +1 or -1, got {sign}")
        ok = mag_is_valid(np.asarray(z_mag)) & mag_is_valid(np.asarray(y_mag))
        if not ok.any():
            raise ValueError("No train galaxy has both a valid z and y magnitude.")
        colour = float(np.median(
            (np.asarray(z_mag)[ok] - np.asarray(y_mag)[ok]).astype(np.float64)
        ))
        return cls(
            scale=float(10.0 ** (sign * 0.4 * colour)),
            median_colour=colour,
            sign=sign,
            n_used=int(ok.sum()),
            n_dropped=int(ok.size - ok.sum()),
        )

    @classmethod
    def fit(cls, train_dataset, sign: int = +1) -> "CopyZBaseline":
        """Fit from a :class:`GalaxyDataset`, using its catalogue columns."""
        return cls.fit_from_mags(
            train_dataset.mags["z"], train_dataset.mags["y"], sign=sign
        )

    # -- prediction --------------------------------------------------------

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict the y band in physical flux units.

        `images` is (N, C, H, W) with C = 4 (g,r,i,z) or 5 (g,r,i,z,y); the z
        band is channel 3 either way. Returns (N, 1, H, W).
        """
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"Expected (N, C, H, W), got {images.shape}")
        c = images.shape[1]
        if c not in (4, len(BANDS)):
            raise ValueError(f"Expected 4 or {len(BANDS)} channels, got {c}")
        return (images[:, 3:4] * self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "model": "B0_copy_z",
            "n_parameters": self.n_parameters,
            "sign_used": self.sign,
            "formula": f"scale = 10**({self.sign:+d} * 0.4 * median(m_z - m_y))",
            "median_z_minus_y_mag": self.median_colour,
            "scale": self.scale,
            "n_train_galaxies_used": self.n_used,
            "n_train_galaxies_dropped_bad_mag": self.n_dropped,
        }


def verify_scale_sign(
    z_images: np.ndarray,
    y_images: np.ndarray,
    median_colour: float,
    aperture_mask: np.ndarray,
) -> dict:
    """Measure the true y/z flux ratio and compare it with both candidate scales.

    Uses TRAIN galaxies only. For each one we sum the flux inside the photometry
    aperture in z and in y; the median of those ratios is exactly what the scale
    is supposed to reproduce.

    Galaxies with non-positive aperture z flux are skipped -- a ratio through a
    near-zero denominator carries no information.
    """
    mask = aperture_mask[None, :, :]
    fz = np.nansum(np.where(mask, z_images, 0.0), axis=(1, 2)).astype(np.float64)
    fy = np.nansum(np.where(mask, y_images, 0.0), axis=(1, 2)).astype(np.float64)

    usable = (fz > 0) & np.isfinite(fz) & np.isfinite(fy)
    if not usable.any():
        raise ValueError("No train galaxy has positive aperture flux in z.")

    empirical = float(np.median(fy[usable] / fz[usable]))
    plus = float(10.0 ** (+0.4 * median_colour))
    minus = float(10.0 ** (-0.4 * median_colour))
    err_plus, err_minus = abs(plus - empirical), abs(minus - empirical)

    return {
        "empirical_median_flux_ratio_y_over_z": empirical,
        "scale_plus_sign": plus,
        "scale_minus_sign": minus,
        "abs_error_plus": err_plus,
        "abs_error_minus": err_minus,
        "median_z_minus_y_mag": float(median_colour),
        "matches": "plus" if err_plus <= err_minus else "minus",
        "n_galaxies_used": int(usable.sum()),
        "n_skipped_nonpositive_z_flux": int(usable.size - usable.sum()),
    }
