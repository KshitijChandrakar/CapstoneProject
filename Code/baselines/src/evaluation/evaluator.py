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
) -> dict[str, np.ndarray]:
    """Run a torch model over a loader, returning flux-unit arrays.

    Gathers the true y, the predicted y (denormalised) and the true z image,
    which the colour metric needs.
    """
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    oids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            out = model(x).cpu().numpy()
            preds.append(denormalize(out, stats, (TARGET_BAND_IDX,)))
            trues.append(batch["target_raw"].numpy())
            zs.append(batch["input_raw"][:, 3:4].numpy())
            oids.append(np.asarray(batch["object_id"]))

    return {
        "y_pred": np.concatenate(preds, axis=0),
        "y_true": np.concatenate(trues, axis=0),
        "z_true": np.concatenate(zs, axis=0),
        "object_id": np.concatenate(oids, axis=0),
    }


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
    return {k: walk([k], v) for k, v in template.items() if k != "arrays"}
