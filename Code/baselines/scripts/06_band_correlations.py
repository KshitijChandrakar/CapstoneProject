"""Step 6 -- pixel-level band-to-band correlation, per redshift bin.

Answers: for each band, how much does a pixel in that band tell you about the
same pixel in the other bands, and how does that change with redshift?

**The trap.** A raw Pearson correlation between two bands over all pixels of a
galaxy image comes out at ~0.99 for every pair, because every band is bright in
the middle and faint at the edges. That number measures "galaxies have centres",
not spectral information, and it is useless for deciding which band predicts
which. So three complementary measures are reported:

  pearson    the raw correlation. Dominated by the shared light profile.
             Reported because it is what gets asked for, with the caveat.

  partial    correlation between two bands with the other three held fixed.
             This is the unique, non-redundant link -- "how much does this band
             tell you that the others do not already". It is the measure that
             actually answers the question.

  beta       standardised regression coefficients: predict each band from the
             other four. beta[t][b] is the change in band t per 1 sigma of
             band b, holding the rest fixed. For t = y these are exactly B1's
             standardised weights, so this generalises B1 to every band pair.

Each is computed over three pixel regions, because the answer differs between
the galaxy and the sky:

  all        every pixel in the cutout
  aperture   inside the r=15 px photometry aperture (galaxy-dominated)
  outskirts  outside it (sky/noise-dominated)

Everything comes from one streaming pass: the script accumulates per-bin
counts, sums and the 5x5 matrix of cross-products, which gives the exact
covariance without ever holding the images in memory.

    python scripts/06_band_correlations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import BANDS, load_config
from src.evaluation.photometry import circular_aperture_mask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0]
REGIONS = ("all", "aperture", "outskirts")
NB = len(BANDS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pixel-level band correlations per redshift bin."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES)
    p.add_argument("--min-galaxies", type=int, default=30)
    p.add_argument("--chunk", type=int, default=300)
    p.add_argument("--spearman-pixels", type=int, default=200_000,
                   help="pixels subsampled per bin for rank correlation; 0 skips")
    return p.parse_args()


class Accumulator:
    """Streaming count / sum / cross-product, giving an exact covariance."""

    def __init__(self) -> None:
        self.n = 0.0
        self.s = np.zeros(NB, dtype=np.float64)
        self.ss = np.zeros((NB, NB), dtype=np.float64)

    def add(self, x: np.ndarray) -> None:
        """`x` is (n_pixels, 5) float64, finite."""
        if x.size == 0:
            return
        self.n += x.shape[0]
        self.s += x.sum(axis=0)
        self.ss += x.T @ x

    def covariance(self) -> np.ndarray | None:
        if self.n < NB + 2:
            return None
        mean = self.s / self.n
        cov = self.ss / self.n - np.outer(mean, mean)
        return cov * (self.n / (self.n - 1.0))      # unbiased


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return np.clip(cov / np.outer(d, d), -1.0, 1.0)


def partial_correlation(corr: np.ndarray) -> np.ndarray:
    """Correlation of each pair with all other bands held fixed."""
    try:
        prec = np.linalg.inv(corr + 1e-12 * np.eye(NB))
    except np.linalg.LinAlgError:
        return np.full((NB, NB), np.nan)
    d = np.sqrt(np.clip(np.diag(prec), 1e-300, None))
    out = -prec / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -1.0, 1.0)


def standardised_betas(corr: np.ndarray) -> np.ndarray:
    """beta[t, b]: predict band t from the other four, in sigma units."""
    out = np.full((NB, NB), np.nan)
    for t in range(NB):
        others = [b for b in range(NB) if b != t]
        a = corr[np.ix_(others, others)]
        rhs = corr[others, t]
        try:
            sol = np.linalg.solve(a + 1e-10 * np.eye(len(others)), rhs)
        except np.linalg.LinAlgError:
            continue
        for b, v in zip(others, sol):
            out[t, b] = v
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    path = cfg.raw_file(args.split)
    if not path.exists():
        print(f"ERROR: raw file not found: {path}")
        print("Put the official GalaxiesML files in data/raw/.")
        return 2

    edges = sorted(args.edges)
    bins = list(zip(edges[:-1], edges[1:]))

    print("=" * 74)
    print("PIXEL-LEVEL BAND CORRELATIONS, PER REDSHIFT BIN")
    print("=" * 74)
    print(f"file  : {path.name}")
    print(f"bands : {list(BANDS)}")
    print(f"bins  : {len(bins)}")

    with h5py.File(path, "r") as f:
        z = np.asarray(f["specz_redshift"][:], dtype=np.float64)
        images = f["image"]
        n_total, size = int(images.shape[0]), int(images.shape[-1])

        ap = circular_aperture_mask(
            (size, size),
            cfg.aperture_center(size),
            float(cfg.dataset["aperture_radius"]),
        )
        region_mask = {"all": np.ones_like(ap), "aperture": ap, "outskirts": ~ap}
        for r, m in region_mask.items():
            print(f"  region '{r}': {int(m.sum()):,} pixels per galaxy")

        acc = {(i, r): Accumulator() for i in range(len(bins)) for r in REGIONS}
        samples: dict[tuple[int, str], list[np.ndarray]] = {
            (i, r): [] for i in range(len(bins)) for r in REGIONS
        }
        rng = np.random.default_rng(0)
        bin_of = np.full(n_total, -1, dtype=np.int64)
        for i, (lo, hi) in enumerate(bins):
            bin_of[(z >= lo) & (z < hi)] = i
        counts = np.array([int((bin_of == i).sum()) for i in range(len(bins))])

        print(f"\ngalaxies per bin: "
              f"{', '.join(f'{lo:.2f}-{hi:.2f}:{c:,}' for (lo, hi), c in zip(bins, counts))}")
        print("\nstreaming pass:")

        for start in range(0, n_total, args.chunk):
            stop = min(start + args.chunk, n_total)
            which = bin_of[start:stop]
            if (which < 0).all():
                continue
            batch = np.asarray(images[start:stop], dtype=np.float64)

            for i in np.unique(which[which >= 0]):
                sel = batch[which == i]                     # (k, 5, H, W)
                for r in REGIONS:
                    m = region_mask[r]
                    # (k, 5, npix) -> (k*npix, 5)
                    pix = sel[:, :, m].transpose(0, 2, 1).reshape(-1, NB)
                    pix = pix[np.isfinite(pix).all(axis=1)]
                    acc[(int(i), r)].add(pix)
                    if args.spearman_pixels and pix.shape[0]:
                        keep = min(
                            pix.shape[0],
                            max(1, args.spearman_pixels // max(int(counts[i]) // args.chunk, 1)),
                        )
                        idx = rng.choice(pix.shape[0], size=keep, replace=False)
                        samples[(int(i), r)].append(pix[idx].astype(np.float32))

            print(f"\r  {100.0 * stop / n_total:5.1f}%  ({stop:,}/{n_total:,})",
                  end="", flush=True)
        print()

    # -- reduce -------------------------------------------------------------
    results: list[dict] = []
    for i, (lo, hi) in enumerate(bins):
        if counts[i] < args.min_galaxies:
            continue
        entry: dict = {
            "z_low": lo, "z_high": hi, "z_mid": 0.5 * (lo + hi),
            "n_galaxies": int(counts[i]), "regions": {},
        }
        for r in REGIONS:
            cov = acc[(i, r)].covariance()
            if cov is None:
                continue
            corr = correlation_from_cov(cov)
            part = partial_correlation(corr)
            beta = standardised_betas(corr)
            block = {
                "n_pixels": float(acc[(i, r)].n),
                "pearson": corr.tolist(),
                "partial": part.tolist(),
                "beta_predict_row_from_others": beta.tolist(),
                "std_per_band": np.sqrt(np.diag(cov)).tolist(),
            }
            if args.spearman_pixels and samples[(i, r)]:
                pooled = np.concatenate(samples[(i, r)], axis=0).astype(np.float64)
                if pooled.shape[0] > NB + 2:
                    ranks = np.apply_along_axis(
                        lambda c: np.argsort(np.argsort(c)).astype(np.float64),
                        0, pooled,
                    )
                    block["spearman"] = correlation_from_cov(
                        np.cov(ranks, rowvar=False)
                    ).tolist()
                    block["n_pixels_spearman"] = int(pooled.shape[0])
            entry["regions"][r] = block
        results.append(entry)

    if not results:
        print("\nERROR: no bin had enough galaxies.")
        return 2

    print_summary(results)
    out = {
        "split": args.split, "bands": list(BANDS),
        "regions": list(REGIONS), "bins": results,
        "note": (
            "pearson is dominated by the shared galaxy light profile and is "
            "near 1 for every pair; partial and beta are the informative "
            "measures."
        ),
    }
    jp = cfg.path("results_dir") / "band_correlations.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = write_markdown(cfg, out)
    figs = make_plots(cfg, out)
    print(f"\nWrote:\n  {jp}\n  {md}")
    for p in figs:
        print(f"  {p}")
    return 0


def print_summary(results: list[dict]) -> None:
    yi = BANDS.index("y")
    print("\n" + "-" * 74)
    print("How much does each band uniquely say about y?  (partial correlation,")
    print("other bands held fixed -- aperture pixels)")
    print("-" * 74)
    print(f"{'z range':>12}{'n gal':>8}" + "".join(f"{b:>9}" for b in BANDS[:4])
          + f"{'pearson(z,y)':>14}")
    for e in results:
        blk = e["regions"].get("aperture")
        if blk is None:
            continue
        part = np.asarray(blk["partial"])
        pear = np.asarray(blk["pearson"])
        label = f"{e['z_low']:.2f}-{e['z_high']:.2f}"
        print(f"{label:>12}{e['n_galaxies']:>8,}"
              + "".join(f"{part[i, yi]:>9.3f}" for i in range(4))
              + f"{pear[BANDS.index('z'), yi]:>14.4f}")


def write_markdown(cfg, out: dict) -> Path:
    bands = out["bands"]
    yi = bands.index("y")
    L = [
        "# Pixel-level band correlations, per redshift bin",
        "",
        f"Computed on the **{out['split']}** split, over every pixel of every "
        "galaxy in each redshift bin.",
        "",
        "## Read this first",
        "",
        "A raw Pearson correlation between any two bands is **~0.99 everywhere**, "
        "because every band is bright in the galaxy centre and faint at the edge. "
        "That number measures *\"galaxies have centres\"*, not spectral "
        "information. It is reported for completeness, but the two measures "
        "below are the ones that answer the question.",
        "",
        "- **partial** — correlation between two bands with the other three held "
        "fixed. The unique, non-redundant link.",
        "- **beta** — standardised regression coefficient: change in the target "
        "band per 1σ of the source band, other bands fixed. For target y these "
        "are exactly B1's standardised weights.",
        "",
        "Three pixel regions are reported separately, because the galaxy and the "
        "sky behave differently: `all`, `aperture` (r=15 px, galaxy-dominated) "
        "and `outskirts` (sky/noise-dominated).",
        "",
        "## Unique contribution to y, by redshift (aperture pixels)",
        "",
        "| z range | n galaxies | " + " | ".join(f"partial({b},y)" for b in bands[:4])
        + " | pearson(z,y) |",
        "|---|---|" + "---|" * 5,
    ]
    for e in out["bins"]:
        blk = e["regions"].get("aperture")
        if blk is None:
            continue
        part = np.asarray(blk["partial"])
        pear = np.asarray(blk["pearson"])
        L.append(
            f"| {e['z_low']:.2f}–{e['z_high']:.2f} | {e['n_galaxies']:,} | "
            + " | ".join(f"{part[i, yi]:+.3f}" for i in range(4))
            + f" | {pear[bands.index('z'), yi]:.4f} |"
        )

    L += ["", "## Full matrices per bin", ""]
    for e in out["bins"]:
        L.append(f"### z = {e['z_low']:.2f} – {e['z_high']:.2f}  "
                 f"({e['n_galaxies']:,} galaxies)")
        for r, blk in e["regions"].items():
            L += ["", f"**{r}** — {blk['n_pixels']:,.0f} pixels", ""]
            for name, key in (("Pearson", "pearson"), ("Partial", "partial"),
                              ("Beta (row predicted from the others)",
                               "beta_predict_row_from_others")):
                M = np.asarray(blk[key])
                L += [f"*{name}*", "",
                      "| | " + " | ".join(bands) + " |",
                      "|---|" + "---|" * len(bands)]
                for i, b in enumerate(bands):
                    cells = ["—" if not np.isfinite(M[i, j]) else f"{M[i, j]:+.3f}"
                             for j in range(len(bands))]
                    L.append(f"| **{b}** | " + " | ".join(cells) + " |")
                L.append("")
    p = cfg.path("results_dir") / "band_correlations.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def make_plots(cfg, out: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = out["bands"]
    yi = bands.index("y")
    binlist = out["bins"]
    written: list[Path] = []
    figdir = cfg.path("figures_dir")

    # -- 1. partial-correlation heatmaps across redshift -------------------
    for region in ("aperture", "outskirts"):
        avail = [e for e in binlist if region in e["regions"]]
        if not avail:
            continue
        n = len(avail)
        fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 1.6, 3.5), squeeze=False)
        for ax, e in zip(axes[0], avail):
            M = np.asarray(e["regions"][region]["partial"])
            im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(bands)), bands, fontsize=8)
            ax.set_yticks(range(len(bands)), bands, fontsize=8)
            ax.set_title(f"z {e['z_low']:.2f}–{e['z_high']:.2f}\n"
                         f"n={e['n_galaxies']:,}", fontsize=9)
            for i in range(len(bands)):
                for j in range(len(bands)):
                    if i != j and np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                fontsize=6.5,
                                color="white" if abs(M[i, j]) > 0.55 else "black")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.03,
                     label="partial correlation")
        fig.suptitle(
            f"Unique band-to-band link, {region} pixels — partial correlation "
            "(other three bands held fixed)\n"
            "This is the informative measure; the raw Pearson correlation is "
            "~0.99 for every pair because all bands share the galaxy's light profile",
            fontsize=10.5,
        )
        fig.subplots_adjust(left=0.04, right=0.87, top=0.72, bottom=0.08, wspace=0.28)
        p = figdir / f"band_partial_corr_{region}.png"
        fig.savefig(p, dpi=140); plt.close(fig); written.append(p)

    # -- 2. contribution to y vs redshift ----------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    colours = {"g": "#4059ad", "r": "#4a9d5f", "i": "#e07a1f", "z": "#d1495b"}
    mids = [e["z_mid"] for e in binlist if "aperture" in e["regions"]]
    for bi, b in enumerate(bands[:4]):
        part = [np.asarray(e["regions"]["aperture"]["partial"])[bi, yi]
                for e in binlist if "aperture" in e["regions"]]
        beta = [np.asarray(e["regions"]["aperture"]["beta_predict_row_from_others"])[yi, bi]
                for e in binlist if "aperture" in e["regions"]]
        ax1.plot(mids, part, "o-", color=colours[b], label=f"{b} → y", lw=1.8, ms=5)
        ax2.plot(mids, beta, "s-", color=colours[b], label=f"{b} → y", lw=1.8, ms=5)
    for ax, ttl, yl in (
        (ax1, "Unique link to y (partial correlation)", "partial corr. with y"),
        (ax2, "Standardised effect on y (= B1's weights)", r"$\beta$  (per 1$\sigma$)"),
    ):
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("spectroscopic redshift")
        ax.set_ylabel(yl); ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(
        "How much each input band contributes to the y band, as a function of "
        "redshift (aperture pixels)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = figdir / "band_contribution_to_y_vs_redshift.png"
    fig.savefig(p, dpi=140); plt.close(fig); written.append(p)
    return written


if __name__ == "__main__":
    sys.exit(main())
