"""P1 -- the PSF-aware model: predict y as a spectrally mixed image, blurred.

The hypothesis, in one sentence: **y differs from z partly because it is
optically broader, so give the model a blur it can control instead of making it
learn one from scratch.**

Script 07 measured y as wider than z in all eight redshift bins (+0.05 to
+0.40 px of second-moment width), and the tests in ``tests/test_physics.py``
show what that implies -- no rescaling of z, global or per-pixel, changes an
image's width at all. Only a convolution can. B0 and B1 therefore could not
reach y however good their weights were; B2 could, but had to discover
convolution-as-PSF-matching among two million free parameters.

### The architecture, and what makes it physics-informed

    inputs (g,r,i,z) --> [spectral mixing] --> [Gaussian blur, sigma = f(z)] --> y

Two pieces, each small, each meaning something:

**Spectral mixing** is a 1x1 convolution -- exactly B1, five parameters. It
answers "what combination of bands has y's colour", and nothing else. It cannot
change any image's width, by construction.

**The blur** is a *separable Gaussian with one learnable width per redshift
bin*, initialised at the value script 07 measured. It answers "how much broader
is y", and nothing else. It cannot change colour, because the kernel is
normalised to sum to one.

The split is the point. In B2, "colour" and "width" are entangled across 1.9M
parameters and neither can be read off afterwards. Here the model's answer to
each physical question is a number you can print, compare against script 07's
measurement, and disagree with.

### What makes this a real test rather than a rigged one

The blur width is learnable, so **the model can reject the physics**. If the
fitted sigma collapses towards zero, P1 is saying the PSF difference does not
help at this redshift -- a finding, and one the training script reports
explicitly rather than burying. Initialising at the measured value only means
the model starts where the data says and moves if the data disagrees.

### Honest expectation

This project trains on ``0 <= z < 0.4``, where the measured width difference is
+0.05 to +0.10 px -- a kernel sigma under 1.3 px. The headroom is small and
known to be small. P1 is not expected to beat B2's 1.9M parameters on MAE; it
is expected to get **close to B2 with a few hundred parameters**, which is a
statement about where the information lives, not about who wins.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import INPUT_BAND_IDX
from src.physics.psf import MEASURED_Y_MINUS_Z_PX, TYPICAL_Z_WIDTH_PX, matching_sigma

#: Redshift bin edges for the per-bin blur width, inside this project's window.
#: Script 07's first two bins split at 0.36; the sweep's own bins split there
#: too, so the boundary is inherited rather than invented.
DEFAULT_Z_EDGES: tuple[float, ...] = (0.0, 0.36, 0.4)


def measured_sigma_for_edges(edges, z_width_px: float = TYPICAL_Z_WIDTH_PX):
    """Kernel sigma script 07 implies for each bin, used as the initial value."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (lo + hi)
        diff = None
        for (blo, bhi), value in MEASURED_Y_MINUS_Z_PX:
            if blo <= mid < bhi:
                diff = value
                break
        if diff is None:
            diff = MEASURED_Y_MINUS_Z_PX[0][1]
        out.append(float(matching_sigma(z_width_px, z_width_px + diff)))
    return out


class SpectralMix(nn.Module):
    """1x1 convolution over the input bands: B1, reused verbatim.

    Deliberately not deeper. Its whole job is colour, and script 06 showed a
    linear combination already captures that -- B1 reached within 6.5% of B0
    with five parameters, and the remaining error was shown to be spatial.
    """

    def __init__(self, in_channels: int = len(INPUT_BAND_IDX)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # Start at B0: copy z, scale by the measured colour ratio. Training
        # then has to improve on the physics baseline rather than on noise.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, -1, 0, 0] = 1.0      # z is the last input band
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LearnableGaussianBlur(nn.Module):
    """Separable Gaussian blur whose width is learned, one width per redshift bin.

    The width is stored as ``log_sigma`` so it cannot go negative, which would
    be meaningless, and so the optimiser takes multiplicative steps -- going
    from 0.9 px to 1.0 px is the same size of step as 9 px to 10 px.

    The kernel is rebuilt from the current sigma on every forward pass rather
    than being a free kernel that is merely initialised Gaussian. That is the
    constraint that makes this physics: the model may choose *how much* to
    blur, never *what shape* to blur with, so the fitted number is directly
    comparable to script 07's measurement.
    """

    #: A Gaussian truncated closer than this many sigma loses enough mass that,
    #: after renormalising, the kernel is measurably narrower than the sigma it
    #: claims. At 3.0 the lost tail is ~0.3%; at 1.6 it is ~11%.
    MIN_RADIUS_IN_SIGMA: float = 3.0

    def __init__(self, init_sigmas, radius: int = 8) -> None:
        super().__init__()
        sig = torch.tensor([max(float(s), 1e-3) for s in init_sigmas],
                           dtype=torch.float32)
        self.log_sigma = nn.Parameter(torch.log(sig))
        self.radius = int(radius)
        if self.radius < 1:
            raise ValueError(f"kernel radius must be >= 1, got {radius}")
        widest = float(sig.max())
        if widest > self.max_supported_sigma:
            raise ValueError(
                f"initial sigma {widest:.3f} px needs a kernel radius of at "
                f"least {math.ceil(self.MIN_RADIUS_IN_SIGMA * widest)}, but "
                f"radius is {self.radius} (supports sigma up to "
                f"{self.max_supported_sigma:.3f}). A kernel truncated this "
                "close to its centre is renormalised into something narrower "
                "than the sigma it reports, which would make P1's fitted "
                "width uninterpretable -- and the fitted width is the result."
            )
        self.register_buffer(
            "offsets",
            torch.arange(-self.radius, self.radius + 1, dtype=torch.float32),
            persistent=False,
        )

    @property
    def max_supported_sigma(self) -> float:
        """Widest blur this kernel size can represent without silent truncation."""
        return self.radius / self.MIN_RADIUS_IN_SIGMA

    def truncation_report(self) -> dict:
        """Whether any fitted width has outgrown the kernel it is drawn on.

        Training moves sigma, and nothing stops it moving past what the fixed
        kernel size can represent. Past that point the model keeps reporting a
        sigma it is not actually applying, so this is checked and printed after
        every run rather than assumed.
        """
        fitted = [float(v) for v in self.sigmas.detach().cpu()]
        limit = self.max_supported_sigma
        return {
            "fitted_sigma_px": fitted,
            "max_supported_sigma_px": limit,
            "kernel_radius_px": self.radius,
            "truncated": [f > limit for f in fitted],
            "any_truncated": any(f > limit for f in fitted),
        }

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def kernels(self) -> torch.Tensor:
        """(n_bins, 2*radius+1) normalised 1-D Gaussians, differentiable in sigma."""
        sigma = self.sigmas.clamp_min(1e-3).unsqueeze(1)          # (bins, 1)
        k = torch.exp(-0.5 * (self.offsets.unsqueeze(0) / sigma) ** 2)
        return k / k.sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, bin_index: torch.Tensor) -> torch.Tensor:
        """Blur each image with its own bin's kernel.

        ``x`` is (N, 1, H, W); ``bin_index`` is (N,) of longs. Reflection
        padding, for the same reason as in the physics module: a cutout's
        border is sky, and zero padding would pull the edges down and bias the
        aperture photometry every metric here is scored on.
        """
        k = self.kernels()[bin_index]                   # (N, K)
        n, _, h, w = x.shape
        pad = self.radius

        # Grouped convolution: one kernel per image, done as N groups.
        row = k.view(n, 1, 1, -1)
        col = k.view(n, 1, -1, 1)
        out = F.pad(x, (pad, pad, 0, 0), mode="reflect")
        out = F.conv2d(out.view(1, n, h, w + 2 * pad), row, groups=n)
        out = F.pad(out.view(n, 1, h, w), (0, 0, pad, pad), mode="reflect")
        out = F.conv2d(out.view(1, n, h + 2 * pad, w), col, groups=n)
        return out.view(n, 1, h, w)


class P1PSF(nn.Module):
    """Spectral mixing followed by a redshift-dependent, learnable PSF blur."""

    def __init__(self, in_channels: int = len(INPUT_BAND_IDX),
                 z_edges=DEFAULT_Z_EDGES, radius: int = 8,
                 z_width_px: float = TYPICAL_Z_WIDTH_PX,
                 learn_blur: bool = True) -> None:
        super().__init__()
        if len(z_edges) < 2:
            raise ValueError("need at least one redshift bin, i.e. two edges")
        self.z_edges = tuple(float(e) for e in z_edges)
        self.initial_sigmas = measured_sigma_for_edges(self.z_edges, z_width_px)

        self.mix = SpectralMix(in_channels)
        self.blur = LearnableGaussianBlur(self.initial_sigmas, radius=radius)
        if not learn_blur:
            # Ablation: hold the blur at exactly what script 07 measured, so
            # any gain is attributable to the measurement and not to fitting.
            self.blur.log_sigma.requires_grad_(False)

    def bin_of(self, redshift: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.z_edges[1:-1], device=redshift.device,
                             dtype=redshift.dtype)
        if edges.numel() == 0:
            return torch.zeros_like(redshift, dtype=torch.long)
        # right=True gives half-open bins, edges[i-1] <= z < edges[i], which is
        # the convention used everywhere else in this project: the dataset is
        # z_min <= z < z_max, and psf.measured_difference looks up bins the
        # same way. The default (right=False) is (lo, hi], so a galaxy sitting
        # exactly on an edge would have been given the neighbouring bin's blur
        # width -- a silent disagreement between the model and the measurement
        # it is supposed to be testing.
        return torch.bucketize(redshift, edges, right=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        mixed = self.mix(x)
        if cond is None:
            # No redshift supplied: use the first bin rather than failing, so
            # the model still runs in contexts that do not carry a redshift.
            idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            idx = self.bin_of(cond.reshape(-1).to(x.dtype))
        return self.blur(mixed, idx)

    # -- reporting ---------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        return {
            "spectral_mix": sum(p.numel() for p in self.mix.parameters()),
            "blur_widths": int(self.blur.log_sigma.numel()),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": self.count_parameters(),
        }

    def fitted_sigmas(self) -> dict:
        """What the model decided, beside what script 07 measured.

        The comparison P1 exists to make. A fitted sigma collapsing towards
        zero means the model is rejecting the PSF hypothesis at that redshift,
        which is a result to report rather than a bug to fix.
        """
        fitted = [float(v) for v in self.blur.sigmas.detach().cpu()]
        bins = [f"{lo:.2f}-{hi:.2f}"
                for lo, hi in zip(self.z_edges[:-1], self.z_edges[1:])]
        return {
            "bins": bins,
            "fitted_sigma_px": fitted,
            "measured_sigma_px": list(self.initial_sigmas),
            "ratio_fitted_to_measured": [
                (f / m if m > 0 else math.nan)
                for f, m in zip(fitted, self.initial_sigmas)
            ],
            "collapsed": [f < 0.1 * m for f, m in zip(fitted, self.initial_sigmas)],
            "truncation": self.blur.truncation_report(),
        }

    def mixing_weights(self) -> dict:
        w = self.mix.conv.weight.detach().cpu().numpy().reshape(-1)
        return {
            "weights": [float(v) for v in w],
            "bias": float(self.mix.conv.bias.detach().cpu().item()),
        }


def build_p1(model_cfg: dict) -> P1PSF:
    """Construct P1 from its config block."""
    edges = model_cfg.get("z_edges", DEFAULT_Z_EDGES)
    model = P1PSF(
        in_channels=int(model_cfg.get("in_channels", len(INPUT_BAND_IDX))),
        z_edges=edges,
        radius=int(model_cfg.get("kernel_radius", 8)),
        z_width_px=float(model_cfg.get("z_width_px", TYPICAL_Z_WIDTH_PX)),
        learn_blur=bool(model_cfg.get("learn_blur", True)),
    )
    expected = model_cfg.get("expected_parameters")
    if expected is not None and model.count_parameters() != int(expected):
        raise ValueError(
            f"P1 has {model.count_parameters()} trainable parameters, config "
            f"expects {expected}. P1's whole claim is that it reaches B2-like "
            "behaviour with a handful of parameters, so the count is part of "
            "the result and is not allowed to drift silently."
        )
    return model
