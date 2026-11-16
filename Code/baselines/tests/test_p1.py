"""Checks on P1, the PSF-aware physics model.

    python tests/test_p1.py

P1's output is not just a prediction, it is a *measurement* -- the fitted blur
width, to be compared against script 07. So these tests care less about "does
it train" and more about "does the number it reports mean what it says".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.models.p1_psf import (
    DEFAULT_Z_EDGES,
    LearnableGaussianBlur,
    P1PSF,
    build_p1,
    measured_sigma_for_edges,
)
from src.physics import psf

PASS, FAIL = 0, 0


def check(label: str, got, want=True, tol: float | None = None) -> None:
    global PASS, FAIL
    if tol is not None:
        ok = bool(np.all(np.abs(np.asarray(got, dtype=float)
                                - np.asarray(want, dtype=float)) <= tol))
    else:
        ok = bool(got == want)
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


# ------------------------------------------------------------- the blur op
def test_torch_blur_matches_the_numpy_reference() -> None:
    """Two independent implementations, same answer, or neither is trusted."""
    print("\nblur: torch against the numpy reference")
    rng = np.random.default_rng(0)
    img = rng.normal(size=(64, 64))
    for sigma in (0.5, 0.8675, 1.2288, 2.5):
        ref = psf.blur(img, sigma, radius=8)
        b = LearnableGaussianBlur([sigma], radius=8)
        got = b(torch.tensor(img, dtype=torch.float32).view(1, 1, 64, 64),
                torch.tensor([0])).detach().numpy()[0, 0]
        check(f"sigma {sigma} agrees to float32 precision",
              float(np.abs(ref - got).max()) < 1e-5)


def test_kernel_truncation_is_refused_not_silent() -> None:
    """The bug this guards: a kernel too small for its sigma is renormalised
    into something narrower, so the model reports a width it never applied."""
    print("\nblur: truncation cannot happen quietly")
    b = LearnableGaussianBlur([1.0], radius=8)
    check("radius 8 supports sigma up to 2.667",
          b.max_supported_sigma, 8 / 3.0, tol=1e-9)
    check("a reasonable sigma is not flagged",
          b.truncation_report()["any_truncated"], False)

    try:
        LearnableGaussianBlur([5.0], radius=4)
        check("a sigma too wide for the kernel is refused at construction", False)
    except ValueError:
        check("a sigma too wide for the kernel is refused at construction", True)

    # Training can still push sigma past the limit; that must be detectable.
    b2 = LearnableGaussianBlur([2.0], radius=8)
    with torch.no_grad():
        b2.log_sigma.fill_(float(np.log(4.0)))
    rep = b2.truncation_report()
    check("a sigma that outgrew the kernel during training is reported",
          rep["any_truncated"], True)
    check("and the report names the limit it passed",
          rep["max_supported_sigma_px"], 8 / 3.0, tol=1e-9)


def test_kernels_are_normalised_so_blurring_preserves_flux() -> None:
    print("\nblur: flux is conserved")
    b = LearnableGaussianBlur([0.9, 1.3], radius=8)
    k = b.kernels()
    check("each bin's kernel sums to 1",
          [float(v) for v in k.sum(dim=1)], [1.0, 1.0], tol=1e-6)
    check("kernel length is 2*radius+1", int(k.shape[1]), 17)

    flat = torch.full((2, 1, 32, 32), 3.0)
    out = b(flat, torch.tensor([0, 1]))
    check("a flat image is unchanged by blurring",
          float((out - 3.0).abs().max()) < 1e-5)


def test_blur_widens_by_the_amount_it_claims() -> None:
    """End to end: ask for a width, measure what actually came out."""
    print("\nblur: the applied width is the requested width")
    n = 81
    yy, xx = np.mgrid[0:n, 0:n]
    s_from = 4.0
    src = np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * s_from ** 2))
    kernel_sigma = 2.0
    b = LearnableGaussianBlur([kernel_sigma], radius=8)
    out = b(torch.tensor(src, dtype=torch.float32).view(1, 1, n, n),
            torch.tensor([0])).detach().numpy()[0, 0]
    expected = float(np.hypot(s_from, kernel_sigma))
    check(f"a {s_from} px source blurred by {kernel_sigma} measures "
          f"{expected:.3f} px",
          psf.width_from_second_moment(out), expected, tol=0.05)


def test_sigma_is_differentiable() -> None:
    """If no gradient reaches log_sigma, the width is decorative, not fitted."""
    print("\nblur: the width actually trains")
    b = LearnableGaussianBlur([1.0], radius=8)
    x = torch.randn(2, 1, 16, 16)
    out = b(x, torch.tensor([0, 0]))
    out.pow(2).mean().backward()
    g = b.log_sigma.grad
    check("log_sigma receives a gradient", g is not None)
    check("and it is non-zero", bool(g.abs().sum() > 0))
    check("and finite", bool(torch.isfinite(g).all()))


# ----------------------------------------------------------------- the model
def test_parameter_count_is_seven() -> None:
    print("\nmodel: seven parameters")
    m = P1PSF()
    b = m.parameter_breakdown()
    check("five for spectral mixing (B1's count)", b["spectral_mix"], 5)
    check("two blur widths, one per redshift bin", b["blur_widths"], 2)
    check("seven trainable in total", b["trainable"], 7)

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "p1_psf.yaml")
        .read_text(encoding="utf-8"))
    built = build_p1(cfg["model"])
    check("the config builds the same model",
          built.parameter_breakdown()["trainable"], 7)

    bad = dict(cfg["model"], expected_parameters=99)
    try:
        build_p1(bad)
        check("a parameter-count mismatch is refused", False)
    except ValueError:
        check("a parameter-count mismatch is refused", True)


def test_initialised_at_script_07s_measurement() -> None:
    print("\nmodel: starts at the measured physics")
    m = P1PSF()
    want = measured_sigma_for_edges(DEFAULT_Z_EDGES)
    check("initial sigmas are the measured ones",
          [float(v) for v in m.blur.sigmas.detach()], want, tol=1e-5)
    check("the low-z bin is the narrower blur", want[0] < want[1])
    check("both are sub-pixel to ~1.3 px, as this z range implies",
          bool(0.5 < want[0] < want[1] < 1.5))


def test_spectral_mix_starts_at_b0_and_cannot_change_width() -> None:
    """The division of labour P1 is built on, asserted rather than assumed."""
    print("\nmodel: mixing does colour, never width")
    m = P1PSF()
    w = m.mixing_weights()
    check("it starts as copy-z (B0)", w["weights"], [0.0, 0.0, 0.0, 1.0],
          tol=1e-9)
    check("with zero bias", w["bias"], 0.0, tol=1e-9)

    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    src = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 4.0 ** 2))
    x = torch.zeros(1, 4, n, n)
    x[0, 3] = torch.tensor(src, dtype=torch.float32)
    mixed = m.mix(x).detach().numpy()[0, 0]
    check("the 1x1 mix leaves the width untouched",
          psf.width_from_second_moment(mixed),
          psf.width_from_second_moment(src), tol=1e-4)


def test_redshift_selects_the_right_bin() -> None:
    print("\nmodel: redshift routing")
    m = P1PSF(z_edges=(0.0, 0.36, 0.4))
    z = torch.tensor([0.0, 0.2, 0.3599, 0.36, 0.39])
    idx = m.bin_of(z)
    check("below 0.36 -> bin 0", [int(v) for v in idx[:3]], [0, 0, 0])
    check("0.36 and above -> bin 1", [int(v) for v in idx[3:]], [1, 1])

    # Half-open bins, matching the dataset filter (z_min <= z < z_max) and
    # psf.measured_difference. A galaxy exactly on an edge must get the SAME
    # bin from both, or the model is tested against a measurement it was not
    # given.
    from src.physics.psf import measured_difference
    for edge_z in (0.0, 0.36):
        model_bin = int(m.bin_of(torch.tensor([edge_z]))[0])
        lo = m.z_edges[model_bin]
        hi = m.z_edges[model_bin + 1]
        check(f"z={edge_z} lands in a bin containing it [{lo}, {hi})",
              bool(lo <= edge_z < hi))
    check("the physics lookup agrees at z=0.36",
          float(measured_difference(0.36)[0]), 0.10, tol=1e-12)

    x = torch.randn(2, 4, 32, 32)
    a = m(x, torch.tensor([0.1, 0.1]))
    b = m(x, torch.tensor([0.38, 0.38]))
    check("different bins give different outputs",
          float((a - b).abs().max()) > 1e-6)
    check("output shape is (N, 1, H, W)", tuple(a.shape), (2, 1, 32, 32))
    check("no redshift still runs", tuple(m(x).shape), (2, 1, 32, 32))


def test_the_model_can_reject_the_physics() -> None:
    """A collapsed sigma is a finding. It must be visible, not silently fine."""
    print("\nmodel: rejecting the hypothesis is reportable")
    m = P1PSF()
    rep = m.fitted_sigmas()
    check("at init nothing is collapsed", any(rep["collapsed"]), False)
    check("fitted equals measured at init",
          rep["ratio_fitted_to_measured"], [1.0, 1.0], tol=1e-4)

    with torch.no_grad():
        m.blur.log_sigma.fill_(float(np.log(1e-3)))
    rep2 = m.fitted_sigmas()
    check("a collapsed width is flagged", all(rep2["collapsed"]), True)
    check("the report carries both numbers for comparison",
          set(("fitted_sigma_px", "measured_sigma_px")) <= set(rep2))


def test_frozen_blur_ablation() -> None:
    print("\nmodel: the learn_blur ablation")
    m = P1PSF(learn_blur=False)
    check("the width is frozen", m.blur.log_sigma.requires_grad, False)
    check("so only the five mixing parameters train",
          m.count_parameters(), 5)
    check("but the blur is still applied at the measured value",
          [float(v) for v in m.blur.sigmas.detach()],
          measured_sigma_for_edges(DEFAULT_Z_EDGES), tol=1e-5)


def test_a_short_training_run_moves_sigma_towards_truth() -> None:
    """Synthetic data where the right answer is known, so "it trains" is
    checkable rather than merely plausible."""
    print("\nmodel: recovers a known blur from synthetic data")
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    n, true_sigma = 32, 1.6

    imgs, targets = [], []
    for _ in range(24):
        src = rng.normal(size=(n, n)).astype(np.float32)
        src = psf.blur(src, 2.0, radius=8).astype(np.float32)  # smooth input
        x = np.zeros((4, n, n), dtype=np.float32)
        x[3] = src
        imgs.append(x)
        targets.append(psf.blur(src, true_sigma, radius=8).astype(np.float32))

    x = torch.tensor(np.stack(imgs))
    y = torch.tensor(np.stack(targets)).unsqueeze(1)
    z = torch.full((len(imgs),), 0.1)

    m = P1PSF()
    start = float(m.blur.sigmas[0])
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = (m(x, z) - y).abs().mean()
        loss.backward()
        opt.step()
    end = float(m.blur.sigmas[0])

    check(f"sigma moved from {start:.3f} towards the true {true_sigma}",
          abs(end - true_sigma) < abs(start - true_sigma))
    check("and got reasonably close", end, true_sigma, tol=0.35)
    check("without truncating", m.blur.truncation_report()["any_truncated"],
          False)


def main() -> int:
    print("=" * 70)
    print("P1 -- PSF-aware physics model")
    print("=" * 70)
    test_torch_blur_matches_the_numpy_reference()
    test_kernel_truncation_is_refused_not_silent()
    test_kernels_are_normalised_so_blurring_preserves_flux()
    test_blur_widens_by_the_amount_it_claims()
    test_sigma_is_differentiable()
    test_parameter_count_is_seven()
    test_initialised_at_script_07s_measurement()
    test_spectral_mix_starts_at_b0_and_cannot_change_width()
    test_redshift_selects_the_right_bin()
    test_the_model_can_reject_the_physics()
    test_frozen_blur_ablation()
    test_a_short_training_run_moves_sigma_towards_truth()
    print("\n" + "=" * 70)
    print(f"{PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
