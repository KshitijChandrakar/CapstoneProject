"""PSF matching: the optics that make y spatially broader than z.

This is P1's physics, and it rests on the one thing script 07 was built to
measure. Cross-correlating each band against itself gives a width proxy per
band, and **y came out wider than z in every redshift bin**:

    z range        y - z width (px)
    0.00-0.36          +0.05
    0.36-0.49          +0.10
    0.49-0.78          +0.19
    0.78-1.01          +0.08
    1.01-1.22          +0.40
    1.22-1.44          +0.28
    1.44-2.00          +0.21
    2.00-4.00          +0.37

That sign is the whole argument for a spatial model. **No per-pixel rescaling
of z can make an image broader** -- multiplying every pixel by a number, or by
a different number each, leaves the second moment of a point source where it
was. Extra spatial extent can only come from a convolution. B0 and B1 were
therefore limited by optics, not by parameter count, and B2's 16.2% gain is
consistent with it having finally been able to blur.

### Why the sign matters more than the size

Two Gaussians convolve in quadrature: blurring a source of width s1 with a
kernel of width sk gives sqrt(s1^2 + sk^2). So turning a z-like image into a
y-like one needs

    s_kernel = sqrt(s_y^2 - s_z^2)

which is **a real number only when y is the broader of the two**. Script 07
found exactly that, in all eight bins. Had the sign gone the other way the
operation would have been deconvolution, which is ill-posed, noise-amplifying
and would have needed regularisation and an entirely different model. P1 is
well-posed because of a measurement, not because of an assumption -- and
``matching_sigma`` refuses rather than silently returning NaN if that ever
stops being true.

### An honest caveat about our redshift window

The project trains on ``0 <= z < 0.4``, which covers only the first bin and
part of the second, where the measured difference is **+0.05 to +0.10 px** --
small. The large differences (+0.28, +0.40) sit at redshifts this phase never
sees. So P1's headroom in this window is modest and should be expected to be;
a 2% gain would be a real result here, not a disappointment. The alternative
reading -- that the effect is too small to matter at all -- is also a finding,
and is what the ablation in the training script is for.
"""

from __future__ import annotations

import numpy as np

#: Measured second-moment width proxy difference, y minus z, per redshift bin.
#: Straight from script 07's published table. Only the z/y pair passed that
#: script's validity test, so only the z/y pair is recorded here.
MEASURED_Y_MINUS_Z_PX: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.00, 0.36), 0.05),
    ((0.36, 0.49), 0.10),
    ((0.49, 0.78), 0.19),
    ((0.78, 1.01), 0.08),
    ((1.01, 1.22), 0.40),
    ((1.22, 1.44), 0.28),
    ((1.44, 2.00), 0.21),
    ((2.00, 4.00), 0.37),
)

#: Typical z-band width proxy in the low-redshift bins, px. Used to turn a
#: *difference* in width into a kernel sigma, since quadrature needs both.
#: Script 07 reports 7.86 and 7.07 for the two bins inside this project's
#: window; 7.5 is the round number between them.
TYPICAL_Z_WIDTH_PX: float = 7.5


def measured_difference(redshift) -> np.ndarray:
    """The y-z width difference script 07 measured, at each galaxy's redshift.

    A lookup into the published table, not an interpolation: these are eight
    measurements with their own uncertainty, and smoothing between them would
    invent precision the sweep does not have. Redshifts beyond the table take
    the nearest bin's value.
    """
    z = np.atleast_1d(np.asarray(redshift, dtype=np.float64))
    out = np.empty_like(z)
    lows = np.array([lo for (lo, _), _ in MEASURED_Y_MINUS_Z_PX])
    highs = np.array([hi for (_, hi), _ in MEASURED_Y_MINUS_Z_PX])
    vals = np.array([v for _, v in MEASURED_Y_MINUS_Z_PX])
    for k, value in enumerate(z):
        inside = np.where((value >= lows) & (value < highs))[0]
        if inside.size:
            out[k] = vals[inside[0]]
        else:
            out[k] = vals[0] if value < lows[0] else vals[-1]
    return out


def matching_sigma(sigma_from, sigma_to, atol: float = 1e-12):
    """Kernel width that turns a ``sigma_from`` image into a ``sigma_to`` one.

    Gaussians add in quadrature, so the answer is sqrt(to^2 - from^2). It is
    real only when the target is the broader of the two.

    Raises rather than returning NaN when asked to sharpen. Convolution cannot
    narrow an image; that direction is deconvolution, which is ill-posed and
    noise-amplifying, and a model that quietly produced NaN kernels there would
    train to a plausible loss while doing nothing. Script 07 measured y to be
    broader than z in every bin, so in this project the error should never
    fire -- and if it does, the premise has changed and we want to be told.
    """
    a = np.asarray(sigma_from, dtype=np.float64)
    b = np.asarray(sigma_to, dtype=np.float64)
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("widths must be non-negative")
    diff = b ** 2 - a ** 2
    if np.any(diff < -atol):
        worst = float(np.min(diff))
        raise ValueError(
            "asked to narrow an image by convolution (target width below "
            f"source width; worst sigma^2 deficit {worst:.6g}). That is "
            "deconvolution, not blurring, and this module will not fake it. "
            "Script 07 measured y as broader than z in every redshift bin, so "
            "this means the inputs are wrong or that premise has changed."
        )
    return np.sqrt(np.clip(diff, 0.0, None))


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Normalised 1-D Gaussian, odd length, summing to exactly 1.

    A zero sigma gives the identity kernel ``[1.0]`` rather than a division by
    zero, so "no blurring needed" is representable and the caller does not need
    a special case.
    """
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0.0:
        return np.ones(1, dtype=np.float64)
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_kernel_2d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Separable 2-D Gaussian, normalised. Flux-preserving by construction."""
    k = gaussian_kernel_1d(sigma, radius)
    return np.outer(k, k)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve a 2-D image with a Gaussian, edges handled by reflection.

    Reflection rather than zero padding: a galaxy cutout's border is sky, not
    vacuum, and zero padding would pull the edges down and bias the aperture
    photometry that every metric in this project is scored on.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    k = gaussian_kernel_1d(sigma)
    if k.size == 1:
        return img.copy()
    pad = k.size // 2
    padded = np.pad(img, pad, mode="reflect")
    rows = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"),
                               axis=1, arr=padded)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"),
                               axis=0, arr=rows)


def width_from_second_moment(image: np.ndarray) -> float:
    """Flux-weighted RMS radius, the same width proxy script 07 uses.

    Only meaningful for non-negative, centrally concentrated images, so the
    tests use synthetic Gaussians. Real sky-subtracted cutouts contain negative
    pixels and this would be dominated by them -- which is why script 07
    measures width by cross-correlation instead.
    """
    img = np.asarray(image, dtype=np.float64)
    total = img.sum()
    if total <= 0:
        raise ValueError("second-moment width needs positive total flux")
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = (img * yy).sum() / total
    cx = (img * xx).sum() / total
    var = (img * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total
    return float(np.sqrt(var / 2.0))          # per-axis sigma


def expected_kernel_sigma(redshift, z_width_px: float = TYPICAL_Z_WIDTH_PX):
    """The blur P1 should need, from script 07's measurement, per galaxy.

    Turns a measured *difference* in width into the kernel sigma that produces
    it: if z has width s and y is (s + d) wide, the kernel is
    sqrt((s + d)^2 - s^2). Used to initialise P1's learnable kernel at the
    measured answer rather than at random, so training starts from the physics
    and adjusts, instead of rediscovering it.
    """
    d = measured_difference(redshift)
    return matching_sigma(z_width_px, z_width_px + d)


def summary() -> str:
    """Log header: the measurement, and what it implies for this z range."""
    lo = expected_kernel_sigma(0.1)[0]
    hi = expected_kernel_sigma(0.38)[0]
    return (
        "PSF matching (script 07): y is broader than z in all eight redshift "
        "bins, by +0.05 to +0.40 px.\n"
        "  Sign is what matters: convolution can only broaden, so y > z makes "
        "the matching kernel real and the problem well posed.\n"
        f"  In this project's window 0 <= z < 0.4 the difference is +0.05 to "
        f"+0.10 px, i.e. a kernel sigma of about {lo:.2f}-{hi:.2f} px. "
        "Modest by design; the large differences live at redshifts this phase "
        "does not train on."
    )
