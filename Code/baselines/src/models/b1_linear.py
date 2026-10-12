"""Baseline B1 -- pixelwise linear combination, a 1x1 convolution.

    y_hat(x, y) = w_g*g(x,y) + w_r*r(x,y) + w_i*i(x,y) + w_z*z(x,y) + b

Exactly 5 learnable parameters, and no spatial context at all -- which is the
point. It tests whether the y band is just local spectral interpolation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import BANDS, INPUT_BAND_IDX, TARGET_BAND_IDX

INPUT_BANDS: tuple[str, ...] = tuple(BANDS[i] for i in INPUT_BAND_IDX)


class LinearBaseline(nn.Module):
    """A 1x1 convolution from the 4 input bands to the single target band."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        bias: bool = True,
        init: str = "default",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        if init == "zeros":
            nn.init.zeros_(self.conv.weight)
            if bias:
                nn.init.zeros_(self.conv.bias)
        elif init != "default":
            raise ValueError(f"init must be 'default' or 'zeros', got {init!r}")

        expected = in_channels * out_channels + (out_channels if bias else 0)
        got = self.count_parameters()
        assert got == expected, f"B1 must have {expected} parameters, got {got}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 1, H, W)."""
        if x.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
        return self.conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- weight readout ----------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """Learned coefficients in STANDARDISED space, plus the bias."""
        w = self.conv.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        out = {f"w_{b}": float(v) for b, v in zip(INPUT_BANDS, w)}
        out["bias"] = (
            float(self.conv.bias.detach().cpu().numpy().reshape(-1)[0])
            if self.conv.bias is not None else 0.0
        )
        return out

    def get_physical_weights(self, stats: dict) -> dict[str, float]:
        """The same model expressed in physical flux units.

        In standardised space the model reads

            (y - mu_y)/s_y = sum_b w_b * (x_b - mu_b)/s_b + b

        which rearranges to ``y = sum_b w_phys_b * x_b + b_phys`` with

            w_phys_b = w_b * s_y / s_b
            b_phys   = mu_y + s_y*b - sum_b w_phys_b * mu_b

        where ``s = std + eps`` is the exact divisor used when normalising.

        **These are the weights to compare across bands.** The standardised
        ones are each scaled by their own band's spread, so they cannot answer
        "does z dominate?" fairly.
        """
        w_std = self.conv.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        b_std = (
            float(self.conv.bias.detach().cpu().numpy().reshape(-1)[0])
            if self.conv.bias is not None else 0.0
        )
        return _to_physical(w_std, b_std, stats)

    def weights_report(self, stats: dict) -> dict:
        """Both spaces plus which band wins, ready to drop into JSON.

        The spec expects ``w_z > w_i > w_r > w_g``, reasoning from spectral
        proximity: z is the band nearest y in wavelength. That prediction is
        about which band is most *informative*, and the measure for that is the
        **standardised** weight -- the change in y per one standard deviation
        of band b. It is dimensionless and unaffected by how widely each band
        happens to vary.

        The physical weight answers a different question: how much y-flux per
        unit of b-flux. Because ``w_phys_b = w_b * s_y / s_b`` divides by the
        band's own spread, a narrow-spread band (r) gets inflated relative to a
        wide-spread one (i), so the physical ordering is not a clean ranking of
        importance. Both are reported; the expectation is checked against the
        standardised ordering.
        """
        std = self.get_weights()
        phys = self.get_physical_weights(stats)
        order_std = sorted(INPUT_BANDS, key=lambda b: std[f"w_{b}"], reverse=True)
        order_phys = sorted(INPUT_BANDS, key=lambda b: phys[f"w_{b}"], reverse=True)
        expected = ["z", "i", "r", "g"]
        return {
            "bands": list(INPUT_BANDS),
            "standardised": std,
            "physical_flux_space": phys,
            "standardised_ordering": order_std,
            "physical_ordering": order_phys,
            "largest_standardised_weight_band": order_std[0],
            "largest_physical_weight_band": order_phys[0],
            "expected_ordering": expected,
            # Primary check: the spec's prediction is about informativeness.
            "matches_expected_ordering": order_std == expected,
            "matches_expected_ordering_physical": order_phys == expected,
            "z_dominates_standardised": order_std[0] == "z",
            "z_dominates_physical": order_phys[0] == "z",
        }


def _to_physical(w_std: np.ndarray, b_std: float, stats: dict) -> dict[str, float]:
    """Convert standardised (weights, bias) into physical flux space."""
    denom, mean = stats["denom"], stats["mean"]
    idx = list(INPUT_BAND_IDX)
    s_y, mu_y = denom[TARGET_BAND_IDX], mean[TARGET_BAND_IDX]

    w_phys = w_std * (s_y / denom[idx])
    b_phys = mu_y + s_y * b_std - float(np.sum(w_phys * mean[idx]))

    out = {f"w_{b}": float(v) for b, v in zip(INPUT_BANDS, w_phys)}
    out["bias"] = float(b_phys)
    return out


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1 loss over valid pixels only, so NaN pixels contribute nothing."""
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def reference_optimum(
    x_norm: np.ndarray, y_norm: np.ndarray, stats: dict
) -> dict:
    """Solve for the best possible 5 parameters directly, without gradients.

    Because B1 is linear, its optimum can be computed outright. That turns
    "did training converge?" from a judgement call into a measured distance,
    and gives an independent check on the physical-weight conversion.

    Returns the exact L2 (least-squares) optimum and, when SciPy is available,
    the L1 optimum -- the one matching B1's actual loss function.
    """
    valid = (
        np.isfinite(y_norm).reshape(-1)
        & np.all(np.isfinite(x_norm), axis=1).reshape(-1)
    )
    a = x_norm.transpose(0, 2, 3, 1).reshape(-1, x_norm.shape[1]).astype(np.float64)
    b = y_norm.reshape(-1).astype(np.float64)
    a, b = a[valid], b[valid]
    if a.size == 0:
        raise ValueError("No valid pixels to solve the reference optimum.")
    design = np.hstack([a, np.ones((a.shape[0], 1))])

    l2, *_ = np.linalg.lstsq(design, b, rcond=None)
    out: dict = {
        "note": (
            "Directly solved optimum for the 5 parameters on the train split, "
            "in standardised space. A converged run lands close to the L1 row."
        ),
        "n_pixels_used": int(a.shape[0]),
        "l2_closed_form": _optimum_entry(l2, stats),
    }

    try:
        from scipy.optimize import minimize

        res = minimize(lambda t: np.abs(design @ t - b).mean(), l2, method="L-BFGS-B")
        out["l1_optimum"] = {
            **_optimum_entry(res.x, stats),
            "train_l1_loss": float(res.fun),
            "converged": bool(res.success),
        }
    except ImportError:
        out["l1_optimum"] = {"unavailable": "SciPy is not installed"}
    return out


def _optimum_entry(theta: np.ndarray, stats: dict) -> dict:
    """Format a solved (w_g, w_r, w_i, w_z, bias) vector in both spaces."""
    w_std, b_std = theta[:-1], float(theta[-1])
    phys = _to_physical(w_std, b_std, stats)
    std = {**{f"w_{b}": float(v) for b, v in zip(INPUT_BANDS, w_std)}, "bias": b_std}
    order_std = sorted(INPUT_BANDS, key=lambda b: std[f"w_{b}"], reverse=True)
    order_phys = sorted(INPUT_BANDS, key=lambda b: phys[f"w_{b}"], reverse=True)
    return {
        "standardised": std,
        "physical_flux_space": phys,
        "standardised_ordering": order_std,
        "physical_ordering": order_phys,
        "largest_standardised_weight_band": order_std[0],
        "largest_physical_weight_band": order_phys[0],
        "matches_expected_ordering": order_std == ["z", "i", "r", "g"],
    }
