"""P2 -- the SED model: predict y by extrapolating a spectrum, not mixing channels.

The hypothesis: **the four input bands are not four channels, they are four
samples of one spectrum at known wavelengths, and y is that spectrum evaluated
at 9762 A.**

That distinction has teeth. B1 and B2 are both invariant to permuting their
input channels -- shuffle g, r, i, z and retrain, and each reaches exactly the
same loss, because neither is ever told the inputs are ordered in wavelength.
P2 cannot be permuted. The wavelengths are in its arithmetic, so swapping two
bands breaks it, and ``tests/test_p2.py`` checks precisely that.

### What the model actually does

    inputs (g,r,i,z) --> [per-pixel SED fit] --> [evaluate at 9762 A] --> y
                              |
                         learned basis weights and a residual correction

The fit is a least-squares polynomial in log wavelength, solved **in closed
form per pixel**, not iterated. Its coefficients are then evaluated at the
target wavelength. Because the projection matrix depends only on the four
fixed input wavelengths, it is computed once at construction and applied as a
1x1 convolution -- so the "physics" costs four numbers and no runtime.

Three learnable pieces sit around it, each with a reason:

* **basis gains** (degree+1 numbers) let the model reweight the fitted
  coefficients -- in effect, how much to trust the curvature it measured when
  stepping 864 A beyond the data.
* **a scalar bias**, because sky subtraction leaves a small offset the SED
  fit cannot represent.
* **an optional residual head**, a small convolution that corrects what the
  smooth spectrum misses. Off by default: with it off, P2 is a *pure* physics
  model and its number says what spectral extrapolation alone can do.

### Why y is the hard direction

9762 A is redward of every input, so this is extrapolation, not
interpolation. The fit is anchored at i (7711 A) and z (8898 A) and then
continued 864 A past the end of the data. That is why the degree matters and
why it is capped: four points can support at most a cubic, but a cubic
interpolates the four inputs exactly, leaving the extrapolation riding on
fitted noise. Degree 2 is the default -- enough curvature to bend, not enough
to chase the data.

### Honest expectation

P2 has no spatial awareness at all: every pixel is predicted from its own four
values. So it inherits B1's ceiling on anything that needs a neighbourhood, and
script 07 showed a real part of the remaining error is spatial. P2 is expected
to **beat B1 clearly** -- it knows about wavelength and B1 does not -- while
falling short of B2. Where it lands between them is the measurement: it says
how much of B2's 16.2% was spectral rather than spatial.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import INPUT_BAND_IDX
from src.physics.sed import design_matrix, target_vector


def projection_to_coefficients(degree: int, ridge: float = 1e-8) -> np.ndarray:
    """The fixed (degree+1, 4) matrix taking four band fluxes to SED coefficients.

    Least squares, solved once: ``coef = (A^T A)^-1 A^T flux``. It depends only
    on the four input wavelengths, which never change, so it is a constant of
    the problem rather than something to learn or recompute per batch.
    """
    A = design_matrix(degree=degree)                      # (4, degree+1)
    gram = A.T @ A + ridge * np.eye(A.shape[1])
    return np.linalg.solve(gram, A.T)                     # (degree+1, 4)


class SEDExtrapolator(nn.Module):
    """Fit a smooth spectrum through the input bands; evaluate it at y.

    The fit and the evaluation are both fixed physics. What is learnable is how
    much weight each fitted coefficient carries into the extrapolation, which
    is the one genuinely uncertain part: a curvature measured over 4754-8898 A
    need not hold 864 A further out.
    """

    def __init__(self, degree: int = 2, ridge: float = 1e-8,
                 learn_gains: bool = True) -> None:
        super().__init__()
        self.degree = int(degree)
        proj = projection_to_coefficients(self.degree, ridge)      # (d+1, 4)
        evalv = target_vector(degree=self.degree)                  # (d+1,)

        self.register_buffer("projection",
                             torch.tensor(proj, dtype=torch.float32))
        self.register_buffer("evaluation",
                             torch.tensor(evalv, dtype=torch.float32))

        # Gains start at 1: the model begins as the exact least-squares
        # extrapolation, i.e. the zero-parameter physics answer, and has to
        # improve on that rather than on noise.
        self.gains = nn.Parameter(torch.ones(self.degree + 1),
                                  requires_grad=bool(learn_gains))
        self.bias = nn.Parameter(torch.zeros(1))

    def coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel SED coefficients, (N, degree+1, H, W)."""
        w = self.projection.view(self.degree + 1, -1, 1, 1)
        return torch.nn.functional.conv2d(x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coef = self.coefficients(x)
        weights = (self.evaluation * self.gains).view(1, -1, 1, 1)
        return (coef * weights).sum(dim=1, keepdim=True) + self.bias


class ResidualHead(nn.Module):
    """Small convolution correcting what a smooth spectrum cannot express.

    Zero-initialised on its last layer, so at the start of training P2 is
    exactly the SED extrapolation and this contributes nothing. Any gain it
    produces is therefore attributable to it, and setting ``residual: false``
    in the config removes it entirely for the pure-physics run.
    """

    def __init__(self, in_channels: int, hidden: int = 16,
                 kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size, padding=pad,
                      padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size, padding=pad,
                      padding_mode="reflect"),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class P2SED(nn.Module):
    """SED extrapolation to 9762 A, optionally with a learned residual."""

    def __init__(self, in_channels: int = len(INPUT_BAND_IDX),
                 degree: int = 2, ridge: float = 1e-8,
                 learn_gains: bool = True, residual: bool = False,
                 residual_hidden: int = 16,
                 residual_kernel: int = 3) -> None:
        super().__init__()
        if in_channels != len(INPUT_BAND_IDX):
            raise ValueError(
                f"P2 fits a spectrum through the {len(INPUT_BAND_IDX)} input "
                f"bands at their known wavelengths; got in_channels="
                f"{in_channels}. The wavelengths are part of the arithmetic, "
                "so the channel count is not free."
            )
        self.sed = SEDExtrapolator(degree=degree, ridge=ridge,
                                   learn_gains=learn_gains)
        self.residual = (ResidualHead(in_channels, residual_hidden,
                                      residual_kernel) if residual else None)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        out = self.sed(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out

    # -- reporting ---------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        res = (sum(p.numel() for p in self.residual.parameters())
               if self.residual is not None else 0)
        return {
            "sed_gains": int(self.sed.gains.numel()),
            "sed_bias": int(self.sed.bias.numel()),
            "residual": res,
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": self.count_parameters(),
        }

    def fitted_gains(self) -> dict:
        """What the model decided about trusting each SED coefficient.

        All gains near 1 means plain least-squares extrapolation was already
        right. A small gain on the quadratic term means the model learned not
        to trust measured curvature when stepping past the data -- a
        physically meaningful statement, and the kind of thing B2's two million
        parameters cannot be asked.
        """
        g = [float(v) for v in self.sed.gains.detach().cpu()]
        return {
            "degree": self.sed.degree,
            "terms": [f"log_lambda^{k}" for k in range(self.sed.degree + 1)],
            "gains": g,
            "bias": float(self.sed.bias.detach().cpu().item()),
            "departs_from_pure_physics": [abs(v - 1.0) > 0.05 for v in g],
        }


def build_p2(model_cfg: dict) -> P2SED:
    """Construct P2 from its config block."""
    model = P2SED(
        in_channels=int(model_cfg.get("in_channels", len(INPUT_BAND_IDX))),
        degree=int(model_cfg.get("degree", 2)),
        ridge=float(model_cfg.get("ridge", 1e-8)),
        learn_gains=bool(model_cfg.get("learn_gains", True)),
        residual=bool(model_cfg.get("residual", False)),
        residual_hidden=int(model_cfg.get("residual_hidden", 16)),
        residual_kernel=int(model_cfg.get("residual_kernel", 3)),
    )
    expected = model_cfg.get("expected_parameters")
    if expected is not None and model.count_parameters() != int(expected):
        raise ValueError(
            f"P2 has {model.count_parameters()} trainable parameters, config "
            f"expects {expected}. The count is part of the claim -- P2 is "
            "meant to reach well past B1 on a handful of parameters -- so it "
            "is not allowed to drift silently."
        )
    return model
