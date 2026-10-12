"""Step 7 -- pixel-to-pixel spatial cross-correlation between bands.

Script 06 answers "how does a pixel in band b relate to the SAME pixel in band
b'". This answers the harder version: **how does a pixel in band b relate to the
pixels AROUND it in band b'** -- i.e. how far does one band's light spread
relative to another's.

That matters for the project, not just for completeness:

  * If every band had the same PSF, the cross-correlation between any two bands
    would be exactly as narrow as each band's own auto-correlation, and a
    pixelwise model like B1 could in principle be perfect.
  * If the y band is blurrier than z (different seeing, different filter), then
    y contains spatial structure that no per-pixel rescaling can produce. B0 and
    B1 are then limited by optics, not by their parameter count -- and that is a
    hard argument for needing a spatial model (B2/B3) rather than a wider linear
    one.

For every redshift bin and every band pair it computes the normalised
cross-correlation as a function of pixel lag, then reduces it to:

  lag0          the value at zero offset (equals the Pearson correlation)
  peak_lag      the offset of the maximum. Non-zero means the bands are
                misregistered, which would be a data problem worth knowing.
  width_px      the second-moment width of the correlation peak -- a PSF proxy.
                Comparing a band's own width against the cross width tells you
                which band is blurrier.

Method: per galaxy, each band is mean-subtracted and zero-padded to twice its
size (so the FFT gives a true linear, not circular, correlation), then the
cross-power spectra are accumulated over all galaxies in the bin and inverted
once at the end. Exact, and one pass over the images.

    python scripts/07_pixel_cross_correlation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import BANDS, load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_EDGES = [0.0, 0.36, 0.49, 0.78, 1.01, 1.22, 1.44, 2.0, 4.0]
NB = len(BANDS)
PAIRS = list(combinations_with_replacement(range(NB), 2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pixel-to-pixel spatial cross-correlation between bands."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--max-lag", type=int, default=12,
                   help="lags kept in the output, in pixels")
    p.add_argument("--min-galaxies", type=int, default=30)
    p.add_argument("--chunk", type=int, default=200)
    return p.parse_args()


def peak_and_width(corr: np.ndarray, max_lag: int) -> dict:
    """Reduce a 2-D correlation map to peak offset and a width.

    `corr` comes straight from an inverse FFT, so zero lag sits at index
    [0, 0] with negative lags wrapped to the far end. fftshift moves zero lag
    to the centre before the window is cut -- without it the "peak" is just the
    corner of the crop, which is a silent and badly misleading bug.
    """
    shifted = np.fft.fftshift(corr)
    centre = shifted.shape[0] // 2
    c = shifted[centre - max_lag:centre + max_lag + 1,
                centre - max_lag:centre + max_lag + 1]
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    peak_y, peak_x = iy - max_lag, ix - max_lag

    # Second-moment width of the positive part around the centre, which is a
    # robust stand-in for a PSF FWHM without fitting a profile.
    yy, xx = np.mgrid[-max_lag:max_lag + 1, -max_lag:max_lag + 1]
    r2 = yy**2 + xx**2
    w = np.clip(c, 0.0, None)
    total = float(w.sum())
    width = float(np.sqrt((w * r2).sum() / total)) if total > 0 else float("nan")

    # Independent per-band read noise correlates only with itself, and only at
    # exactly zero lag. It therefore adds a spike to an AUTO-correlation that no
    # cross-correlation has, which biases any auto-vs-cross comparison. Dropping
    # the single central pixel removes it; this is the width psf_excess uses.
    w0 = w.copy()
    w0[max_lag, max_lag] = 0.0
    total0 = float(w0.sum())
    width0 = float(np.sqrt((w0 * r2).sum() / total0)) if total0 > 0 else float("nan")

    return {
        "lag0": float(c[max_lag, max_lag]),
        "peak_value": float(c[iy, ix]),
        "peak_lag_y": int(peak_y),
        "peak_lag_x": int(peak_x),
        "peak_lag_radius": float(np.hypot(peak_y, peak_x)),
        "width_px": width,
        "width_sq": float(width**2) if np.isfinite(width) else float("nan"),
        "width_px_excl_zero_lag": width0,
        "width_sq_excl_zero_lag": float(width0**2) if np.isfinite(width0) else float("nan"),
        "profile": [float(v) for v in c[max_lag, max_lag:]],   # radial slice
    }



def psf_excess(entry: dict, base: str, target: str = "y") -> float:
    """How much broader `target`'s PSF is than `base`'s, in pixels.

    An image is a galaxy convolved with a PSF, so

        cross-corr(base, target) = AC(galaxy) * P_base * P_target
        auto-corr(base, base)    = AC(galaxy) * P_base * P_base

    Both carry the same AC(galaxy) * P_base factor. Subtracting in quadrature
    cancels the galaxy's own size -- which otherwise dominates the raw widths
    and makes a real PSF difference look like nothing -- and leaves

        width^2(cross) - width^2(auto) = w^2(P_target) - w^2(P_base)

    The signed square root is therefore the extra blur of `target` over `base`
    in pixels. Negative means `target` is the sharper band.

    **Accuracy, measured rather than assumed.** tests/test_cross_correlation.py
    injects a known extra blur of sigma = 1.5 px into y and checks what comes
    back: roughly 0.8-1.8 px. The sign and the ranking are reliable; the
    magnitude is biased low and scattered by a few tenths of a pixel. The cause
    is the finite lag window -- the quadrature identity above holds exactly only
    for untruncated second moments, and clipping at +/-max_lag compresses the
    difference.

    Read this as a qualitative indicator ("is y blurrier, and against which
    bands") rather than a calibrated PSF measurement. The number is a lower
    bound.
    """
    cross = (entry["pairs"].get(f"{base}-{target}")
             or entry["pairs"].get(f"{target}-{base}"))
    auto = entry["pairs"].get(f"{base}-{base}")
    if not cross or not auto:
        return float("nan")
    d = cross["width_sq"] - auto["width_sq"]
    return float(np.sign(d) * np.sqrt(abs(d)))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    path = cfg.raw_file(args.split)
    if not path.exists():
        print(f"ERROR: raw file not found: {path}")
        return 2

    edges = sorted(args.edges)
    bins = list(zip(edges[:-1], edges[1:]))

    print("=" * 74)
    print("PIXEL-TO-PIXEL SPATIAL CROSS-CORRELATION BETWEEN BANDS")
    print("=" * 74)
    print(f"file   : {path.name}")
    print(f"pairs  : {len(PAIRS)} (including each band with itself)")
    print(f"maxlag : +/-{args.max_lag} px")

    with h5py.File(path, "r") as f:
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)
        images = f["image"]
        n_total, size = int(images.shape[0]), int(images.shape[-1])
        pad = 2 * size                      # zero-pad -> linear correlation

        bin_of = np.full(n_total, -1, dtype=np.int64)
        for i, (lo, hi) in enumerate(bins):
            bin_of[(z >= lo) & (z < hi)] = i
        counts = np.array([int((bin_of == i).sum()) for i in range(len(bins))])

        # Accumulated cross-power per (bin, pair), plus per-band energy.
        cross = {(i, p): np.zeros((pad, pad), dtype=np.complex128)
                 for i in range(len(bins)) for p in range(len(PAIRS))}
        energy = np.zeros((len(bins), NB), dtype=np.float64)
        used = np.zeros(len(bins), dtype=np.int64)

        print(f"\ngalaxies per bin: "
              + ", ".join(f"{lo:.2f}-{hi:.2f}:{c:,}" for (lo, hi), c in zip(bins, counts)))
        print("\nstreaming pass (FFT per galaxy per band):")

        for start in range(0, n_total, args.chunk):
            stop = min(start + args.chunk, n_total)
            which = bin_of[start:stop]
            if (which < 0).all():
                continue
            batch = np.asarray(images[start:stop], dtype=np.float64)

            for k in range(batch.shape[0]):
                bi = int(which[k])
                if bi < 0:
                    continue
                img = batch[k]
                if not np.isfinite(img).all():
                    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                # Mean-subtract each band, then zero-pad.
                img = img - img.mean(axis=(1, 2), keepdims=True)
                padded = np.zeros((NB, pad, pad), dtype=np.float64)
                padded[:, :size, :size] = img
                F = np.fft.rfft2(padded, axes=(1, 2))

                energy[bi] += (img**2).sum(axis=(1, 2))
                for pi, (a, b) in enumerate(PAIRS):
                    cross[(bi, pi)] += np.fft.irfft2(
                        F[a] * np.conj(F[b]), s=(pad, pad)
                    ).astype(np.complex128)
                used[bi] += 1

            print(f"\r  {100.0 * stop / n_total:5.1f}%  ({stop:,}/{n_total:,})",
                  end="", flush=True)
        print()

    # -- reduce -------------------------------------------------------------
    results: list[dict] = []
    for i, (lo, hi) in enumerate(bins):
        if counts[i] < args.min_galaxies or used[i] == 0:
            continue
        entry: dict = {
            "z_low": lo, "z_high": hi, "z_mid": 0.5 * (lo + hi),
            "n_galaxies": int(used[i]), "pairs": {},
        }
        for pi, (a, b) in enumerate(PAIRS):
            denom = np.sqrt(energy[i, a] * energy[i, b])
            if denom <= 0:
                continue
            corr = np.real(cross[(i, pi)]) / denom
            key = f"{BANDS[a]}-{BANDS[b]}"
            entry["pairs"][key] = peak_and_width(corr, args.max_lag)
        # Galaxy-profile-cancelled PSF difference of y against each input band.
        entry["psf_excess_y_over"] = {
            b: psf_excess(entry, b, "y") for b in BANDS[:4]
        }
        results.append(entry)

    if not results:
        print("\nERROR: no bin had enough galaxies.")
        return 2

    print_summary(results)
    out = {
        "split": args.split, "bands": list(BANDS), "max_lag": args.max_lag,
        "bins": results,
        "method": (
            "per galaxy: mean-subtract each band, zero-pad to 2x, accumulate "
            "cross-power spectra over the bin, invert once. Zero-padding makes "
            "this a linear (not circular) correlation, so there is no wrap-around."
        ),
        "how_to_read": (
            "width_px for a band with itself is that band's PSF proxy. If the "
            "y-y width exceeds the z-z width, y is intrinsically blurrier and no "
            "pixelwise rescaling of z can reproduce it -- a hard limit on B0/B1 "
            "that argues for a spatial model. peak_lag should be (0,0); anything "
            "else means the bands are misregistered."
        ),
    }
    jp = cfg.path("results_dir") / "pixel_cross_correlation.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = write_markdown(cfg, out)
    figs = make_plots(cfg, out)
    print(f"\nWrote:\n  {jp}\n  {md}")
    for p in figs:
        print(f"  {p}")
    return 0


def print_summary(results: list[dict]) -> None:
    print("\n" + "-" * 74)
    print("PSF proxy: spatial width of each band's own correlation (px)")
    print("-" * 74)
    print(f"{'z range':>12}{'n gal':>8}" + "".join(f"{b+'-'+b:>9}" for b in BANDS)
          + f"{'y vs z':>9}")
    for e in results:
        widths = []
        for b in BANDS:
            p = e["pairs"].get(f"{b}-{b}")
            widths.append(p["width_px"] if p else float("nan"))
        diff = widths[4] - widths[3]
        label = f"{e['z_low']:.2f}-{e['z_high']:.2f}"
        print(f"{label:>12}{e['n_galaxies']:>8,}"
              + "".join(f"{w:>9.2f}" for w in widths)
              + f"{diff:>+9.2f}")
    print("\n  Raw widths are dominated by galaxy size, so the y-vs-z column is a")
    print("  weak indicator. The table below cancels the galaxy profile.")

    print("\n" + "-" * 74)
    print("EXTRA BLUR of y over each input band (px, galaxy profile cancelled)")
    print("-" * 74)
    print(f"{'z range':>12}{'n gal':>8}" + "".join(f"{b + '->y':>10}" for b in BANDS[:4]))
    for e in results:
        ex = e.get("psf_excess_y_over", {})
        label = f"{e['z_low']:.2f}-{e['z_high']:.2f}"
        print(f"{label:>12}{e['n_galaxies']:>8,}"
              + "".join(f"{ex.get(b, float('nan')):>10.2f}" for b in BANDS[:4]))
    print("\n  Positive = y is blurrier than that band, so that much of y's")
    print("  structure cannot be produced by rescaling it. B0/B1 are then limited")
    print("  by optics rather than parameter count -> a spatial model is needed.")
    print("\n  CAVEAT: against a known injected blur of 1.50 px this estimator")
    print("  returns ~0.8-1.8 px. Trust the sign and the ranking; treat the")
    print("  magnitude as a lower bound, not a calibrated PSF width.")

    print("\n" + "-" * 74)
    print("Registration check: peak lag of each band against y (should be 0,0)")
    print("-" * 74)
    for e in results:
        offs = []
        for b in BANDS[:4]:
            p = e["pairs"].get(f"{b}-y") or e["pairs"].get(f"y-{b}")
            offs.append(f"{b}:({p['peak_lag_x']},{p['peak_lag_y']})" if p else f"{b}:-")
        print(f"  z={e['z_low']:.2f}-{e['z_high']:.2f}  " + "  ".join(offs))


def write_markdown(cfg, out: dict) -> Path:
    bands = out["bands"]
    L = [
        "# Pixel-to-pixel spatial cross-correlation between bands",
        "",
        f"Computed on the **{out['split']}** split, per redshift bin, for every "
        f"band pair, out to ±{out['max_lag']} px of lag.",
        "",
        "## What this measures, and why it matters",
        "",
        "Script 06 asked how a pixel in one band relates to the *same* pixel in "
        "another. This asks how it relates to the pixels *around* it — i.e. how "
        "far each band's light spreads.",
        "",
        "- `width_px` for a band against **itself** is that band's PSF proxy.",
        "- **If the y band is wider than z, then y contains spatial structure "
        "that no per-pixel rescaling of z can reproduce.** B0 and B1 would then "
        "be limited by optics rather than by parameter count — which is a hard "
        "argument for a spatial model (B2/B3) over a wider linear one.",
        "- `peak_lag` should be (0, 0). Anything else means the bands are "
        "misregistered, which would be a data problem.",
        "",
        "## PSF proxy per band (spatial width, px)",
        "",
        "| z range | n galaxies | " + " | ".join(f"{b}–{b}" for b in bands)
        + " | y − z |",
        "|---|---|" + "---|" * (len(bands) + 1),
    ]
    for e in out["bins"]:
        w = []
        for b in bands:
            p = e["pairs"].get(f"{b}-{b}")
            w.append(p["width_px"] if p else float("nan"))
        L.append(
            f"| {e['z_low']:.2f}–{e['z_high']:.2f} | {e['n_galaxies']:,} | "
            + " | ".join(f"{v:.2f}" for v in w)
            + f" | **{w[4] - w[3]:+.2f}** |"
        )

    L += ["", "## Zero-lag correlation and peak offset, each band against y", "",
          "| z range | " + " | ".join(f"{b}→y lag0" for b in bands[:4])
          + " | " + " | ".join(f"{b}→y peak" for b in bands[:4]) + " |",
          "|---|" + "---|" * 8]
    for e in out["bins"]:
        l0, pk = [], []
        for b in bands[:4]:
            p = e["pairs"].get(f"{b}-y") or e["pairs"].get(f"y-{b}")
            l0.append(f"{p['lag0']:.3f}" if p else "—")
            pk.append(f"({p['peak_lag_x']},{p['peak_lag_y']})" if p else "—")
        L.append(f"| {e['z_low']:.2f}–{e['z_high']:.2f} | " + " | ".join(l0)
                 + " | " + " | ".join(pk) + " |")
    L.append("")
    p = cfg.path("results_dir") / "pixel_cross_correlation.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def make_plots(cfg, out: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = out["bands"]
    binlist = out["bins"]
    figdir = cfg.path("figures_dir")
    written: list[Path] = []
    colours = {"g": "#4059ad", "r": "#4a9d5f", "i": "#e07a1f",
               "z": "#d1495b", "y": "#5f0f40"}

    # -- 1. PSF width per band vs redshift ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    mids = [e["z_mid"] for e in binlist]
    for b in bands:
        w = [e["pairs"].get(f"{b}-{b}", {}).get("width_px", np.nan) for e in binlist]
        ax1.plot(mids, w, "o-", color=colours[b], label=f"{b}", lw=1.8, ms=5)
    ax1.set_xlabel("spectroscopic redshift")
    ax1.set_ylabel("spatial correlation width (px)")
    ax1.set_title("How far each band's light spreads (PSF proxy)", fontsize=11)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, ncol=2)

    diff = [
        e["pairs"].get("y-y", {}).get("width_px", np.nan)
        - e["pairs"].get("z-z", {}).get("width_px", np.nan)
        for e in binlist
    ]
    ax2.plot(mids, diff, "D-", color="#1b263b", lw=2.0, ms=6)
    ax2.axhline(0, color="k", lw=0.9, ls="--")
    ax2.fill_between(mids, 0, diff, where=np.array(diff) > 0,
                     color="#d1495b", alpha=0.18)
    ax2.set_xlabel("spectroscopic redshift")
    ax2.set_ylabel("width(y) − width(z)   (px)")
    ax2.set_title("Is y blurrier than z?\nabove zero = structure B0/B1 cannot "
                  "reproduce by rescaling", fontsize=11)
    ax2.grid(alpha=0.3)
    fig.suptitle("Spatial extent of each band, by redshift", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = figdir / "band_psf_width_vs_redshift.png"
    fig.savefig(p, dpi=140); plt.close(fig); written.append(p)

    # -- 2. cross-correlation profiles, each band against y ----------------
    n = len(binlist)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow),
                             squeeze=False, sharey=True)
    for ax, e in zip(axes.ravel(), binlist):
        for b in bands[:4]:
            pr = (e["pairs"].get(f"{b}-y") or e["pairs"].get(f"y-{b}") or {}).get("profile")
            if pr:
                ax.plot(range(len(pr)), pr, "-", color=colours[b], label=f"{b}→y", lw=1.6)
        yy = e["pairs"].get("y-y", {}).get("profile")
        if yy:
            ax.plot(range(len(yy)), yy, "--", color="k", lw=1.3, label="y→y")
        ax.set_title(f"z {e['z_low']:.2f}–{e['z_high']:.2f}", fontsize=9.5)
        ax.set_xlabel("pixel lag"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("normalised correlation")
    axes[0][0].legend(fontsize=7.5)
    for ax in axes.ravel()[len(binlist):]:
        ax.axis("off")
    fig.suptitle(
        "How a pixel in each input band correlates with the surrounding pixels "
        "in y\nthe dashed line is y with itself — a band that falls off more "
        "slowly than it is blurrier than y",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = figdir / "pixel_cross_correlation_profiles.png"
    fig.savefig(p, dpi=140); plt.close(fig); written.append(p)
    return written


if __name__ == "__main__":
    sys.exit(main())
