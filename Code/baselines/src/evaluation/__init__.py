"""Image metrics, aperture photometry and the unified evaluator."""
from src.evaluation.evaluator import (
    collect_predictions, collect_raw_images, evaluate_arrays, summarise_over_seeds,
)
from src.evaluation.metrics import pixel_metrics
from src.evaluation.photometry import (
    aperture_flux, circular_aperture_mask, flux_to_mag, nmad, photometry_metrics,
)

__all__ = [
    "collect_predictions", "collect_raw_images", "evaluate_arrays",
    "summarise_over_seeds", "pixel_metrics", "aperture_flux",
    "circular_aperture_mask", "flux_to_mag", "nmad", "photometry_metrics",
]
