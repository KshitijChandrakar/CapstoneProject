"""Image-quality metrics: MAE, RMSE, PSNR, SSIM.

These are the secondary metrics. Photometric accuracy (see photometry.py) is
what an astronomy referee actually checks, but these are what a computer-vision
audience expects, so both are reported.

NaN pixels are masked out everywhere.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _as_2d_stack(a: np.ndarray) -> np.ndarray:
    """(N, 1, H, W) or (N, H, W) -> (N, H, W) float64."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 4:
        if a.shape[1] != 1:
            raise ValueError(f"Expected a single channel, got shape {a.shape}")
        a = a[:, 0]
    if a.ndim != 3:
        raise ValueError(f"Expected (N, H, W) or (N, 1, H, W), got {a.shape}")
    return a


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mean absolute error and root mean squared error over valid pixels."""
    t, p = _as_2d_stack(y_true), _as_2d_stack(y_pred)
    valid = np.isfinite(t) & np.isfinite(p)
    if not valid.any():
        raise ValueError("No valid pixels to score.")
    diff = np.where(valid, t - p, 0.0)
    n = float(valid.sum())
    return {
        "mae": float(np.abs(diff).sum() / n),
        "rmse": float(np.sqrt((diff**2).sum() / n)),
        "n_valid_pixels": int(n),
        "n_invalid_pixels": int(np.count_nonzero(~valid)),
    }


def psnr_ssim_per_galaxy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-galaxy PSNR and SSIM, averaged.

    ``data_range`` is ``max - min`` of that galaxy's *true* y image, so each
    galaxy is scored against its own dynamic range rather than a global one.
    """
    t_all, p_all = _as_2d_stack(y_true), _as_2d_stack(y_pred)
    psnrs: list[float] = []
    ssims: list[float] = []
    n_skipped = 0
    n_filled = 0

    for i in range(t_all.shape[0]):
        t, p = t_all[i], p_all[i]
        valid = np.isfinite(t) & np.isfinite(p)
        if not valid.any():
            n_skipped += 1
            continue

        data_range = float(t[valid].max() - t[valid].min())
        if not np.isfinite(data_range) or data_range <= 0:
            n_skipped += 1
            continue

        mse = float(((t[valid] - p[valid]) ** 2).mean())
        psnrs.append(float("inf") if mse == 0
                     else float(10.0 * np.log10(data_range**2 / mse)))

        # SSIM is a windowed statistic and cannot skip individual pixels, so a
        # masked pixel is set to the same value in both images, contributing
        # no difference.
        if valid.all():
            t_s, p_s = t, p
        else:
            n_filled += 1
            fill = float(t[valid].mean())
            t_s, p_s = np.where(valid, t, fill), np.where(valid, p, fill)
        ssims.append(float(structural_similarity(t_s, p_s, data_range=data_range)))

    finite_psnr = [v for v in psnrs if np.isfinite(v)]
    return {
        "psnr_mean": float(np.mean(finite_psnr)) if finite_psnr else float("nan"),
        "psnr_std": float(np.std(finite_psnr)) if finite_psnr else float("nan"),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "ssim_std": float(np.std(ssims)) if ssims else float("nan"),
        "n_galaxies_scored": len(ssims),
        "n_galaxies_skipped": n_skipped,
        "n_galaxies_with_filled_pixels_for_ssim": n_filled,
    }


def pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All image metrics in one call."""
    return {**mae_rmse(y_true, y_pred), **psnr_ssim_per_galaxy(y_true, y_pred)}
