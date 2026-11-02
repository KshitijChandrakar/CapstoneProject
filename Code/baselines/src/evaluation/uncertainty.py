"""Scoring predictive uncertainty -- B3's error bars, and whether they are honest.

B3 predicts a sigma per pixel. A sigma is only worth having if it is
*calibrated*: when the model says +/-0.1, the truth should fall inside that
band about 68% of the time. A model can have an excellent MAE and useless
error bars, and nothing in metrics.py or photometry.py would notice.

Three things are measured here, and they answer different questions:

* **NLL** -- one number combining accuracy and honesty. The headline score,
  and the only metric on which B3 can be compared to the physics models P1-P3
  that also output uncertainty.
* **Calibration** -- coverage at several levels, plus the standardised
  residual's spread. This catches a model that is simply over- or
  under-confident everywhere.
* **Sharpness and error correlation** -- a model can be perfectly calibrated
  by predicting one huge sigma everywhere. Sharpness (the median sigma) and
  the correlation between sigma and the actual error show whether the model
  knows *where* it is uncertain, which is the part the anomaly application
  needs.

All inputs are in physical flux units unless stated, so the numbers line up
with the rest of the results tables.
"""

from __future__ import annotations

import numpy as np

from src.config import TARGET_BAND_IDX

#: 0.5*log(2*pi) -- the constant term of a Gaussian NLL.
HALF_LOG_2PI: float = 0.9189385332046727


def sigma_to_physical(sigma_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Convert a standardised sigma to flux units.

    Standardisation is ``x_norm = (x - mu) / s`` with ``s = std + eps``, an
    affine map, so a standard deviation transforms by the scale alone:
    ``sigma_phys = sigma_norm * s``. No mean enters.
    """
    s = float(np.asarray(stats["denom"])[TARGET_BAND_IDX])
    return np.asarray(sigma_norm, dtype=np.float64) * s


def _flat_valid(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten to 1-D and drop pixels that are not finite everywhere."""
    t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(s) & (s > 0)
    return t[ok], p[ok], s[ok]


def gaussian_nll(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    full: bool = True,
) -> dict:
    """Mean Gaussian NLL per pixel, and its two parts.

    Splitting the score is worth the two extra lines: ``log_sigma_term`` is the
    price of claiming uncertainty and ``quadratic_term`` is how much the
    residual actually costs. A model that cheats by inflating sigma shows a
    large log term and a tiny quadratic one.
    """
    t, p, s = _flat_valid(y_true, y_pred, sigma)
    if t.size == 0:
        return {"nll": float("nan"), "n_pixels": 0}
    log_term = np.log(s)
    quad = 0.5 * ((t - p) / s) ** 2
    nll = log_term + quad + (HALF_LOG_2PI if full else 0.0)
    return {
        "nll": float(np.mean(nll)),
        "nll_median": float(np.median(nll)),
        "log_sigma_term": float(np.mean(log_term)),
        "quadratic_term": float(np.mean(quad)),
        "includes_2pi_constant": bool(full),
        "n_pixels": int(t.size),
    }


def calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.68, 0.95),
) -> dict:
    """Is the predicted sigma the right size?

    Computes the standardised residual ``(y - mu)/sigma``. If sigma is right,
    that has unit standard deviation, and the fraction inside +/-k sigma
    matches the Gaussian expectation.

    ``ratio`` is the headline: the standardised residual's own spread.

        ratio ~ 1    calibrated
        ratio > 1    over-confident (sigma too small) -- the dangerous
                     direction, because every anomaly score is then inflated
        ratio < 1    under-confident (sigma too large)

    The spread is reported both as a standard deviation and as an NMAD, since
    pixel residuals are heavy-tailed and the two disagree informatively: a
    normal std with a much smaller NMAD means a few catastrophic pixels, not a
    generally mis-scaled sigma.
    """
    from scipy.stats import norm  # only needed here; SciPy is already required

    t, p, s = _flat_valid(y_true, y_pred, sigma)
    if t.size == 0:
        return {"ratio_std": float("nan"), "n_pixels": 0}

    zscore = (t - p) / s
    cov = {}
    for level in levels:
        k = float(norm.ppf(0.5 + level / 2.0))
        cov[f"{level:.2f}"] = {
            "nominal": float(level),
            "k_sigma": k,
            "empirical": float(np.mean(np.abs(zscore) <= k)),
        }
    nmad = 1.4826 * float(np.median(np.abs(zscore - np.median(zscore))))
    return {
        "ratio_std": float(np.std(zscore)),
        "ratio_nmad": nmad,
        "zscore_mean": float(np.mean(zscore)),
        "zscore_median": float(np.median(zscore)),
        "coverage": cov,
        # Mean absolute gap between nominal and empirical coverage: one number
        # for "how far off is the whole curve", not just one level.
        "mean_abs_coverage_error": float(np.mean([
            abs(v["empirical"] - v["nominal"]) for v in cov.values()
        ])),
        "n_pixels": int(t.size),
    }


def sharpness_and_resolution(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray
) -> dict:
    """Does the model know *where* it is uncertain?

    Calibration alone is satisfiable by one big constant sigma. Two checks
    against that:

    * **sharpness** -- the median predicted sigma. Lower is better, given
      calibration holds.
    * **error correlation** -- Spearman correlation between the predicted
      sigma and the absolute residual, per pixel. A constant sigma scores 0.
      This is the property the anomaly search depends on.
    """
    from scipy.stats import spearmanr

    t, p, s = _flat_valid(y_true, y_pred, sigma)
    if t.size == 0:
        return {"sharpness_median_sigma": float("nan"), "n_pixels": 0}

    err = np.abs(t - p)
    # Spearman over tens of millions of pixels is slow and adds nothing over a
    # large sample, so it is estimated on a fixed, seeded subsample.
    n = min(200_000, t.size)
    idx = np.random.default_rng(0).choice(t.size, size=n, replace=False)
    rho = spearmanr(s[idx], err[idx]).statistic
    return {
        "sharpness_median_sigma": float(np.median(s)),
        "sharpness_mean_sigma": float(np.mean(s)),
        "sigma_spread_iqr": float(np.subtract(*np.percentile(s, [75, 25]))),
        "spearman_sigma_vs_abs_error": float(rho),
        "n_pixels_correlation": int(n),
        "n_pixels": int(t.size),
    }


def per_galaxy_nll(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, full: bool = True
) -> np.ndarray:
    """Mean NLL for each galaxy, shape (N,).

    This is the anomaly score. A galaxy scores highly when the model is both
    wrong and confident, which is the combination worth a human looking at --
    unlike plain residuals, where every faint noisy galaxy comes top.
    """
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    n = t.shape[0]
    t, p, s = t.reshape(n, -1), p.reshape(n, -1), s.reshape(n, -1)

    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(s) & (s > 0)
    nll = np.log(np.where(ok, s, 1.0)) + 0.5 * (
        (t - p) / np.where(ok, s, 1.0)
    ) ** 2
    if full:
        nll = nll + HALF_LOG_2PI
    counts = ok.sum(axis=1)
    totals = np.where(ok, nll, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def anomaly_candidates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    object_id: np.ndarray,
    top_k: int = 50,
) -> dict:
    """The galaxies the model got most confidently wrong.

    Reported as a shortlist, deliberately not as a discovery claim: a high
    score can equally mean a bad cutout, a nearby star or a calibration fault.
    Script 10 writes the list; a human looks at the images.
    """
    scores = per_galaxy_nll(y_true, y_pred, sigma)
    finite = np.isfinite(scores)
    order = np.argsort(-np.where(finite, scores, -np.inf))
    k = int(min(top_k, int(finite.sum())))
    top = order[:k]
    return {
        "top_k": k,
        "object_id": [int(v) for v in np.asarray(object_id).reshape(-1)[top]],
        "index": [int(v) for v in top],
        "nll": [float(v) for v in scores[top]],
        "median_nll_all_galaxies": float(np.nanmedian(scores)),
        "note": (
            "Ranked by mean per-pixel predictive NLL on the test split: wrong "
            "AND confident. A high score is a candidate to inspect, not a "
            "detection -- bad cutouts, blended neighbours and calibration "
            "faults rank here too."
        ),
    }


def uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.68, 0.95),
) -> dict:
    """Every uncertainty metric for one set of predictions, in flux units."""
    return {
        "nll": gaussian_nll(y_true, y_pred, sigma),
        "calibration": calibration(y_true, y_pred, sigma, levels=levels),
        "sharpness": sharpness_and_resolution(y_true, y_pred, sigma),
    }


def verdict(metrics: dict) -> dict:
    """Turn the calibration numbers into a statement, with its own threshold.

    A ratio within 10% of 1.0 counts as calibrated. That tolerance is a
    judgement, so it is recorded in the output rather than left implicit in
    prose somewhere.
    """
    cal = metrics.get("calibration", {})
    ratio = cal.get("ratio_std", float("nan"))
    sharp = metrics.get("sharpness", {})
    rho = sharp.get("spearman_sigma_vs_abs_error", float("nan"))

    if not np.isfinite(ratio):
        state = "unmeasured"
    elif ratio > 1.1:
        state = "over-confident"
    elif ratio < 0.9:
        state = "under-confident"
    else:
        state = "calibrated"
    return {
        "calibration_state": state,
        "ratio_std": float(ratio),
        "tolerance": 0.1,
        "knows_where_it_is_uncertain": bool(np.isfinite(rho) and rho > 0.2),
        "spearman_sigma_vs_abs_error": float(rho),
        "note": (
            "A ratio above 1 means sigma is too small, which inflates every "
            "anomaly score; below 1 means the model is hedging. The Spearman "
            "correlation guards against passing calibration with one constant "
            "sigma everywhere."
        ),
    }
