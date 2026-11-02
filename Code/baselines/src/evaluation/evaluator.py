"""Unified evaluation: run a model over a split and collect every metric.

Both baselines go through this same code, so the comparison is like for like.
Predictions are always scored in physical flux units, after denormalising.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import TARGET_BAND_IDX, Config
from src.data.preprocess import denormalize
from src.evaluation.metrics import pixel_metrics
from src.evaluation.photometry import compare_flux_handling, photometry_metrics
from src.evaluation.uncertainty import (
    anomaly_candidates,
    sigma_to_physical,
    uncertainty_metrics,
    verdict,
)


def _photometry_kwargs(cfg: Config, image_size: int) -> dict:
    ph = cfg.photometry
    return {
        "aperture_radius": float(cfg.dataset["aperture_radius"]),
        "aperture_center": cfg.aperture_center(image_size),
        "zeropoint": float(ph["zeropoint"]),
        "outlier_threshold": float(ph["outlier_dmag_threshold"]),
        "nmad_factor": float(ph["nmad_factor"]),
        "clamp_floor": float(ph["clamp_floor"]),
    }


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    stats: dict,
    device: torch.device,
    conditioning_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Run a torch model over a loader, returning flux-unit arrays.

    Gathers the true y, the predicted y (denormalised) and the true z image,
    which the colour metric needs. Handles all three trained models:

    * B1 and B2 return a tensor;
    * B3 returns ``(mu, logvar)``, and then ``y_sigma`` is in the result, also
      in flux units;
    * ``conditioning_key`` (``"redshift"`` for B3) is passed to the model as a
      second argument.

    Keeping one collector for every model is the point. The B0/B1 table was
    trustworthy because both went through identical evaluation code, and that
    property is worth more as the ladder grows.
    """
    model.eval()
    preds: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    oids: list[np.ndarray] = []
    redshifts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            if conditioning_key is None:
                out = model(x)
            else:
                out = model(x, batch[conditioning_key].to(device))

            if isinstance(out, tuple):
                mu, logvar = out
                # sigma = exp(0.5*logvar) in standardised units, then scaled by
                # the target band's divisor. Not denormalize(): a standard
                # deviation takes the scale but not the offset.
                sigma_norm = torch.exp(0.5 * logvar).cpu().numpy()
                sigmas.append(sigma_to_physical(sigma_norm, stats))
            else:
                mu = out

            preds.append(denormalize(mu.cpu().numpy(), stats, (TARGET_BAND_IDX,)))
            trues.append(batch["target_raw"].numpy())
            zs.append(batch["input_raw"][:, 3:4].numpy())
            oids.append(np.asarray(batch["object_id"]))
            redshifts.append(batch["redshift"].numpy())

    result = {
        "y_pred": np.concatenate(preds, axis=0),
        "y_true": np.concatenate(trues, axis=0),
        "z_true": np.concatenate(zs, axis=0),
        "object_id": np.concatenate(oids, axis=0),
        "redshift": np.concatenate(redshifts, axis=0),
    }
    if sigmas:
        result["y_sigma"] = np.concatenate(sigmas, axis=0)
    return result


def collect_raw_images(loader: torch.utils.data.DataLoader) -> dict[str, np.ndarray]:
    """Gather the unnormalised images a non-neural model (B0) needs."""
    trues: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    oids: list[np.ndarray] = []
    for batch in loader:
        trues.append(batch["target_raw"].numpy())
        zs.append(batch["input_raw"][:, 3:4].numpy())
        oids.append(np.asarray(batch["object_id"]))
    return {
        "y_true": np.concatenate(trues, axis=0),
        "z_true": np.concatenate(zs, axis=0),
        "object_id": np.concatenate(oids, axis=0),
    }


def evaluate_arrays(
    cfg: Config,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_true: np.ndarray,
    return_arrays: bool = False,
) -> dict:
    """Every metric for one set of predictions, in physical flux units."""
    image_size = int(y_true.shape[-1])
    kwargs = _photometry_kwargs(cfg, image_size)
    handling = str(cfg.photometry["nonpositive_flux_handling"])

    phot = photometry_metrics(
        y_true, y_pred, z_true, handling=handling, return_arrays=True, **kwargs
    )
    arrays = phot.pop("_arrays")

    result = {
        "pixel": pixel_metrics(y_true, y_pred),
        "photometry": phot,
        # Both conventions, so the cost of the spec's 1e-30 clamp is visible.
        "magnitude_handling_comparison": compare_flux_handling(
            y_true, y_pred, z_true, **kwargs
        ),
        "n_galaxies": int(y_true.shape[0]),
    }
    if return_arrays:
        result["arrays"] = arrays
    return result


def evaluate_with_uncertainty(
    cfg: Config,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_true: np.ndarray,
    y_sigma: np.ndarray,
    object_id: np.ndarray,
    top_k: int = 50,
    levels: tuple[float, ...] = (0.5, 0.68, 0.95),
    return_arrays: bool = False,
) -> dict:
    """Everything :func:`evaluate_arrays` reports, plus the uncertainty block.

    B3 is scored on exactly the same pixel and photometry metrics as B0, B1 and
    B2 -- so the ladder table stays comparable -- and then on whether its error
    bars are honest, which is the part only it can be asked.
    """
    result = evaluate_arrays(cfg, y_true, y_pred, z_true,
                             return_arrays=return_arrays)
    result["uncertainty"] = uncertainty_metrics(
        y_true, y_pred, y_sigma, levels=levels
    )
    result["uncertainty_verdict"] = verdict(result["uncertainty"])
    result["anomaly_candidates"] = anomaly_candidates(
        y_true, y_pred, y_sigma, object_id, top_k=top_k
    )
    return result


def summarise_over_seeds(per_seed: list[dict]) -> dict:
    """mean / std of every numeric leaf across seed runs."""
    if not per_seed:
        raise ValueError("No per-seed results to summarise.")

    def walk(path: list[str], node):
        if isinstance(node, dict):
            return {k: walk(path + [k], v) for k, v in node.items()}
        values: list[float] = []
        for run in per_seed:
            cur = run
            try:
                for key in path:
                    cur = cur[key]
            except (KeyError, TypeError):
                continue
            if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                if np.isfinite(cur):
                    values.append(float(cur))
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_seeds": len(values),
            "values": values,
        }

    template = per_seed[0]
    # "arrays" is bulk pixel data and "anomaly_candidates" is a ranked list of
    # object ids -- averaging either across seeds would be meaningless.
    skip = ("arrays", "anomaly_candidates")
    return {k: walk([k], v) for k, v in template.items() if k not in skip}
