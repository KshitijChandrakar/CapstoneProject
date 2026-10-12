"""Step 4 -- compare B0 and B1: the benchmark table and the diagnostic plots.

    python scripts/04_benchmark_comparison.py

Writes results_table.md, results_table.tex and the figures the slides need.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.data.dataset import GalaxyDataset
from src.data.preprocess import denormalize, load_global_stats, normalize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Slide 10's column order.
COLUMNS = ["Model", "Params", "Test MAE", "PSNR (dB)", "SSIM",
           "Flux NMAD", "sigma(mag)", "Outlier %", "sigma(z-y)"]
DECIMALS = [6, 3, 4, 4, 4, 3, 4]
#: Whether a lower number is better, per metric column.
LOWER_BETTER = [True, False, False, True, True, True, True]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare B0 and B1.")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    return p.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/02_run_b0.py and "
            "scripts/03_train_b1.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(mean: float, std: float | None, nd: int) -> str:
    if std is None or not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def build_rows(b0: dict, variants: list[dict], split: str) -> list[tuple]:
    """(label, params, (mean, std) per metric) for B0 and each B1 variant."""
    m0 = b0["splits"][split]["B0"]
    p0, ph0 = m0["pixel"], m0["photometry"]
    rows = [(
        "B0 (Copy-z)", 0,
        (p0["mae"], None), (p0["psnr_mean"], None), (p0["ssim_mean"], None),
        (ph0["relative_flux_error"]["nmad"], None),
        (ph0["delta_mag"]["nmad"], None),
        (ph0["delta_mag"]["outlier_percent"], None),
        (ph0["colour_error_z_minus_y"]["nmad"], None),
    )]
    for b1 in variants:
        s = b1["metrics_over_seeds"]
        p1, ph1 = s["pixel"], s["photometry"]
        rows.append((
            b1.get("label", "B1 (Linear 1x1)"), b1["n_parameters"],
            (p1["mae"]["mean"], p1["mae"]["std"]),
            (p1["psnr_mean"]["mean"], p1["psnr_mean"]["std"]),
            (p1["ssim_mean"]["mean"], p1["ssim_mean"]["std"]),
            (ph1["relative_flux_error"]["nmad"]["mean"],
             ph1["relative_flux_error"]["nmad"]["std"]),
            (ph1["delta_mag"]["nmad"]["mean"], ph1["delta_mag"]["nmad"]["std"]),
            (ph1["delta_mag"]["outlier_percent"]["mean"],
             ph1["delta_mag"]["outlier_percent"]["std"]),
            (ph1["colour_error_z_minus_y"]["nmad"]["mean"],
             ph1["colour_error_z_minus_y"]["nmad"]["std"]),
        ))
    return rows



def _ensure_orderings(b1: dict) -> None:
    """Recompute band orderings from the reported weights, in place.

    Keeps the table correct regardless of which version of 03_train_b1.py
    produced the JSON, so fixing the table never requires retraining.
    """
    bands = b1["bands"]
    expected = ["z", "i", "r", "g"]
    w = b1["weights_over_seeds"]
    order_std = sorted(bands, key=lambda b: w["standardised"][f"w_{b}"]["mean"],
                       reverse=True)
    order_phys = sorted(bands, key=lambda b: w["physical_flux_space"][f"w_{b}"]["mean"],
                        reverse=True)
    b1["standardised_ordering_mean"] = order_std
    b1["physical_ordering_mean"] = order_phys
    b1["expected_ordering"] = expected
    b1["matches_expected_ordering"] = order_std == expected
    b1["matches_expected_ordering_physical"] = order_phys == expected
    b1["z_dominates"] = order_std[0] == "z"
    b1["z_dominates_physical"] = order_phys[0] == "z"

    # How close are the middle two? If r and i are nearly tied, the exact
    # ordering is not a robust claim and the report must say so.
    ranked = sorted((w["standardised"][f"w_{b}"]["mean"] for b in bands),
                    reverse=True)
    gap = abs(ranked[1] - ranked[2]) if len(ranked) > 2 else float("nan")
    spread = max(w["standardised"][f"w_{b}"]["std"] for b in bands)
    b1["middle_pair_gap_standardised"] = float(gap)
    # Robust only if the gap is large next to seed noise AND the two spaces agree.
    b1["ordering_is_robust"] = bool(gap > 3.0 * spread and order_std == order_phys)


def write_tables(cfg, b0: dict, variants: list[dict], split: str) -> tuple[Path, Path]:
    rows = build_rows(b0, variants, split)
    # Quote the weights of the best-converged variant: an unconverged run's
    # weights are mostly its random initialisation.
    b1 = min(variants, key=lambda v: (v.get("convergence") or {})
             .get("distance_to_solved_optimum_mean", 1e9))
    # Derive the orderings here rather than trusting whatever keys the training
    # script happened to write. Older b1_metrics.json files predate these keys,
    # and relying on them silently produced a blank ordering in the table.
    _ensure_orderings(b1)
    ds = cfg.dataset

    md = ["| " + " | ".join(COLUMNS) + " |",
          "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for label, params, *cells in rows:
        vals = [_fmt(m, s, nd) for (m, s), nd in zip(cells, DECIMALS)]
        md.append(f"| **{label}** | {params} | " + " | ".join(vals) + " |")

    improve: list[str] = []
    for ri, v in enumerate(variants, start=1):
        improve.append(f"### {v.get('label', 'B1')} vs B0")
        improve.append("")
        for name, (m0, _), (m1, _), lower in zip(
            COLUMNS[2:], rows[0][2:], rows[ri][2:], LOWER_BETTER
        ):
            if not (np.isfinite(m0) and np.isfinite(m1)) or m0 == 0:
                improve.append(f"- **{name}**: n/a")
                continue
            change = (m0 - m1) / abs(m0) * 100.0
            better = (m1 < m0) if lower else (m1 > m0)
            improve.append(
                f"- **{name}**: B0 {m0:.6g} -> {m1:.6g}  "
                f"({abs(change):.1f}% {'lower' if change > 0 else 'higher'}; "
                f"**{'better' if better else 'worse'}**)"
            )
        improve.append("")

    sign = b0["sign_verification"]
    phys = b1["weights_over_seeds"]["physical_flux_space"]
    std_w = b1["weights_over_seeds"]["standardised"]
    bands = b1["bands"]
    n_test = b0["splits"][split]["B0"]["n_galaxies"]

    body = [
        "# Baselines B0 & B1 — predicting the HSC y band from g, r, i, z",
        "",
        f"Redshift window {ds['z_min']} ≤ z < {ds['z_max']}. "
        f"Evaluated on the **{split}** split ({n_test:,} galaxies), in physical "
        f"(sky-subtracted flux) units. B1 is mean ± std over seeds {b1['seeds']}.",
        "",
        f"Aperture photometry: circular, radius {ds['aperture_radius']} px, centred "
        f"at {tuple(b0['splits'][split]['B0']['photometry']['aperture']['center'])}. "
        "NMAD = 1.4826 × median(|x − median(x)|). "
        f"Outlier % = galaxies with |Δm| > "
        f"{b0['splits'][split]['B0']['photometry']['delta_mag']['outlier_threshold']}.",
        "",
        "## Benchmark table (slide 10)",
        "",
        *md,
        "",
        "## 1. How much does B1 improve over B0?",
        "",
        *improve,
        "",
        "## 2. What are the learned weights? Does z dominate?",
        "",
        "The spec predicts `w_z > w_i > w_r > w_g`, reasoning from spectral "
        "proximity — z is the band nearest y in wavelength. That is a claim "
        "about which band is most **informative**, and the measure for that is "
        "the **standardised** weight: the change in y per one standard "
        "deviation of that band. It is dimensionless and does not depend on "
        "how widely each band happens to vary.",
        "",
        "The physical weight answers a different question — how much y-flux per "
        "unit of band flux, via `y_flux ≈ Σ_b w_b · x_b_flux + bias`. Since "
        "`w_phys_b = w_b · σ_y / σ_b` divides by each band's own spread, a "
        "narrow-spread band (r) is inflated relative to a wide-spread one (i), "
        "so the physical ordering is not a clean ranking of importance. Both "
        "are given below.",
        "",
        "| Band | w (physical) | w (standardised) |",
        "|---|---|---|",
    ]
    for b in bands:
        body.append(
            f"| {b} | {phys[f'w_{b}']['mean']:+.4f} ± {phys[f'w_{b}']['std']:.4f} "
            f"| {std_w[f'w_{b}']['mean']:+.4f} ± {std_w[f'w_{b}']['std']:.4f} |"
        )
    body += [
        f"| bias | {phys['bias']['mean']:+.4g} ± {phys['bias']['std']:.3g} "
        f"| {std_w['bias']['mean']:+.4f} ± {std_w['bias']['std']:.4f} |",
        "",
        f"- Ordering by **standardised** weight: "
        f"**{' > '.join(b1.get('standardised_ordering_mean', []))}**  "
        f"← compare with the spec's expected **z > i > r > g**",
        f"- Ordering by physical weight: "
        f"**{' > '.join(b1['physical_ordering_mean'])}**",
        "",
        f"**z dominates: {b1['z_dominates']}.** "
        f"Matches the spec's expected ordering: "
        f"**{b1['matches_expected_ordering']}**"
        + ("." if b1.get("ordering_is_robust", True)
           else " — but the exact sequence is not robust, see below."),
        "",
        ("" if b1.get("ordering_is_robust", True) else
         "> **The exact ordering is not robust.** The middle two bands differ by "
         f"only {b1.get('middle_pair_gap_standardised', float('nan')):.4f} in "
         "standardised weight, and the physical weights order them the other way "
         f"round ({' > '.join(b1['physical_ordering_mean'])}). z's lead is solid; "
         "the r-versus-i order is not. Claim \"z dominates\" rather than the "
         "full four-band sequence."),
        "",
    ]

    conv = b1.get("convergence")
    if conv:
        body += [
            "### Did B1 converge?",
            "",
            "B1 is linear with 5 parameters, so its optimum can be solved for "
            "directly. The distance below is between the trained weights and that "
            "solved optimum (standardised space) — small means converged.",
            "",
            f"- optimiser steps: **{b1['hyperparameters']['optimiser_steps']:,}**",
            f"- distance from the solved optimum: **{conv['distance_to_solved_optimum_mean']:.4f}**",
            f"- largest weight spread across seeds: {conv['max_weight_spread_across_seeds']:.4f}",
            f"- converged: **{conv['converged']}**",
            "",
        ]

    # -- failure modes ------------------------------------------------------
    body += [
        "## 3. Where do the baselines fail?",
        "",
        "Galaxies split at the median true aperture flux. Any constant offset in "
        "a prediction is a large fraction of a faint galaxy's total flux, so faint "
        "galaxies are where these models break.",
        "",
        "| Model | Δm NMAD, faint half | Δm NMAD, bright half |",
        "|---|---|---|",
    ]
    bs0 = b0["splits"][split]["B0"]["photometry"].get("delta_mag_by_brightness") or {}
    if bs0.get("faint_half"):
        body.append(f"| B0 | {bs0['faint_half']['nmad']:.4f} "
                    f"| {bs0['bright_half']['nmad']:.4f} |")
    bs1 = (b1["metrics_over_seeds"]["photometry"].get("delta_mag_by_brightness")
           or {})
    if bs1.get("faint_half"):
        body.append(f"| B1 | {bs1['faint_half']['nmad']['mean']:.4f} ± "
                    f"{bs1['faint_half']['nmad']['std']:.4f} "
                    f"| {bs1['bright_half']['nmad']['mean']:.4f} ± "
                    f"{bs1['bright_half']['nmad']['std']:.4f} |")

    # -- B0 sign ------------------------------------------------------------
    body += [
        "",
        "## 4. Which sign of the B0 formula is correct?",
        "",
        "The spec sheet and slide 7 write `scale = 10^(−0.4 × Δ(z−y))`. "
        "That sign is wrong. Since `m = −2.5·log₁₀(F)`:",
        "",
        "```",
        "F_y / F_z = 10^(−0.4·m_y) / 10^(−0.4·m_z) = 10^(+0.4·(m_z − m_y))",
        "```",
        "",
        "Measured on the training images rather than argued:",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| median(m_z − m_y) on train | {sign['median_z_minus_y_mag']:+.4f} mag |",
        f"| **measured** median(F_y / F_z) from the images | "
        f"**{sign['empirical_median_flux_ratio_y_over_z']:.6f}** |",
        f"| 10^(**+**0.4 × median) | {sign['scale_plus_sign']:.6f} "
        f"(error {sign['abs_error_plus']:.6f}) |",
        f"| 10^(**−**0.4 × median) | {sign['scale_minus_sign']:.6f} "
        f"(error {sign['abs_error_minus']:.6f}) |",
        "",
        f"The data matches the **{sign['matches']}** sign "
        f"({sign['n_galaxies_used']:,} train galaxies). "
        f"B0 therefore uses scale = {b0['model']['scale']:.6f}.",
        "",
        "Note: because the same measured z flux is used on both sides of the "
        "colour, Δ(z−y) = −Δm exactly, so the two NMAD columns are identical.",
        "",
    ]

    # -- clamp vs exclude ---------------------------------------------------
    cmp0 = b0["splits"][split]["B0"].get("magnitude_handling_comparison")
    if cmp0:
        body += [
            "## 5. A note on non-positive aperture fluxes",
            "",
            "The spec computes magnitudes as `−2.5·log₁₀(max(F, 1e-30))`. For a "
            "galaxy whose aperture flux is ≤ 0 (a faint non-detection, which is "
            "physically normal for sky-subtracted data) that clamp produces a "
            "~75 mag error — an artefact of the clamp, not a property of the "
            "model. Excluding those galaxies and reporting the count is the "
            "honest alternative. Both are shown:",
            "",
            "| Handling | Δm median | Δm NMAD | Outlier % |",
            "|---|---|---|---|",
        ]
        for mode in ("exclude", "clamp"):
            d = cmp0[mode]["delta_mag"]
            body.append(f"| {mode} | {d['median']:+.4f} | {d['nmad']:.4f} "
                        f"| {d['outlier_percent']:.2f} |")
        body += [
            "",
            f"Galaxies with non-positive true flux: "
            f"{cmp0['exclude']['delta_mag']['n_excluded_nonpositive_true_flux']:,}; "
            f"predicted flux: "
            f"{cmp0['exclude']['delta_mag']['n_excluded_nonpositive_pred_flux']:,}.",
            "",
        ]

    md_path = cfg.path("results_dir") / "results_table.md"
    md_path.write_text("\n".join(body), encoding="utf-8")

    # -- LaTeX --------------------------------------------------------------
    tex_cols = ["Model", "Params", "Test MAE $\\downarrow$", "PSNR (dB) $\\uparrow$",
                "SSIM $\\uparrow$", "Flux NMAD $\\downarrow$",
                "$\\sigma_{\\rm mag}$ $\\downarrow$", "Outlier \\% $\\downarrow$",
                "$\\sigma_{z-y}$ $\\downarrow$"]
    tex = [
        "% Auto-generated by scripts/04_benchmark_comparison.py",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Baselines B0 and B1 for HSC $y$-band prediction at "
        f"${ds['z_min']} \\leq z < {ds['z_max']}$, on the {split} split "
        f"({n_test:,} galaxies). Metrics are in physical flux units; B1 is "
        f"mean $\\pm$ std over {len(b1['seeds'])} seeds.}}",
        "  \\label{tab:baselines_b0_b1}",
        "  \\small",
        "  \\begin{tabular}{l r r r r r r r r}",
        "    \\hline",
        "    " + " & ".join(tex_cols) + " \\\\",
        "    \\hline",
    ]
    for label, params, *cells in rows:
        vals = [(f"${m:.{nd}f}$" if s is None or not np.isfinite(s)
                 else f"${m:.{nd}f} \\pm {s:.{nd}f}$")
                for (m, s), nd in zip(cells, DECIMALS)]
        tex.append(f"    {label.replace('1x1', '$1\\times1$')} & {params} & "
                   + " & ".join(vals) + " \\\\")
    tex += ["    \\hline", "  \\end{tabular}", "\\end{table}", ""]
    tex_path = cfg.path("results_dir") / "results_table.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")

    return md_path, tex_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(cfg, b0: dict, variants: list[dict], split: str) -> list[Path]:
    b1 = min(variants, key=lambda v: (v.get("convergence") or {})
             .get("distance_to_solved_optimum_mean", 1e9))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.path("figures_dir")
    written: list[Path] = []
    res_dir = cfg.path("results_dir")

    a0 = np.load(res_dir / f"arrays_b0_{split}.npz")
    seed0 = b1["seeds"][0]
    a1 = np.load(res_dir / f"arrays_{b1.get('tag','b1')}_seed{seed0}_{split}.npz")

    nm0 = b0["splits"][split]["B0"]["photometry"]["delta_mag"]["nmad"]
    nm1 = b1["per_seed"][0]["metrics"]["photometry"]["delta_mag"]["nmad"]

    # -- 1. magnitude residual histogram -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    d0 = a0["dmag"][np.isfinite(a0["dmag"])]
    d1 = a1["dmag"][np.isfinite(a1["dmag"])]
    lim = max(float(np.percentile(np.abs(np.concatenate([d0, d1])), 99)), 0.1)
    bins = np.linspace(-lim, lim, 60)
    ax.hist(d0, bins=bins, alpha=0.55, color="#d1495b", edgecolor="none",
            label=f"B0 (Copy-z)   NMAD {nm0:.4f}")
    ax.hist(d1, bins=bins, alpha=0.55, color="#3d5a80", edgecolor="none",
            label=f"B1 (Linear)   NMAD {nm1:.4f}")
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_xlabel(r"$\Delta m = m_{\rm pred} - m_{\rm true}$   (mag)")
    ax.set_ylabel("galaxies")
    ax.set_title("Aperture magnitude residuals on the test split\n"
                 "(narrower and centred on zero is better)")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "dmag_histogram.png"
    fig.savefig(p, dpi=140); plt.close(fig); written.append(p)

    # -- 2. predicted vs true aperture flux --------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 6.3))
    for arr, label, colour in ((a0, "B0 (Copy-z)", "#d1495b"),
                               (a1, f"B1 (Linear, seed {seed0})", "#3d5a80")):
        ok = (arr["flux_true"] > 0) & (arr["flux_pred"] > 0)
        ax.scatter(arr["flux_true"][ok], arr["flux_pred"][ok], s=5, alpha=0.25,
                   color=colour, edgecolors="none",
                   label=f"{label}  ({int(ok.sum()):,} galaxies)")
    pooled = np.concatenate([a0["flux_true"], a0["flux_pred"], a1["flux_pred"]])
    pooled = pooled[pooled > 0]
    lo, hi = float(pooled.min()) * 0.7, float(pooled.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="perfect (1:1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true y aperture flux")
    ax.set_ylabel("predicted y aperture flux")
    ax.set_title(f"Aperture flux, r = {cfg.dataset['aperture_radius']} px\n"
                 "(non-positive fluxes cannot appear on log axes)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    p = fig_dir / "flux_scatter.png"
    fig.savefig(p, dpi=140); plt.close(fig); written.append(p)

    # -- 3. learned weights -------------------------------------------------
    bands = b1["bands"]
    for space, fname, ylabel in (
        ("physical_flux_space", "b1_weights_physical.png",
         r"$w_{\rm phys}$  (flux-space weight)"),
        ("standardised", "b1_weights_standardised.png",
         r"$w$  (standardised space)"),
    ):
        w = b1["weights_over_seeds"][space]
        means = [w[f"w_{b}"]["mean"] for b in bands]
        stds = [w[f"w_{b}"]["std"] for b in bands]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        colours = ["#e07a1f" if b == "z" else "#4daf4a" for b in bands]
        ax.bar(bands, means, yerr=stds, capsize=6, color=colours,
               edgecolor="k", lw=0.6)
        for i, b in enumerate(bands):
            for v in w[f"w_{b}"]["values"]:
                ax.plot(i, v, "k.", ms=5, alpha=0.75)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("input band"); ax.set_ylabel(ylabel)
        title = "B1 learned 1x1 convolution weights"
        if space == "physical_flux_space":
            title += f"\nordering: {' > '.join(b1['physical_ordering_mean'])}"
        ax.set_title(f"{title}\nmean ± std over seeds {b1['seeds']}; "
                     "dots = individual seeds", fontsize=10)
        fig.tight_layout()
        p = fig_dir / fname
        fig.savefig(p, dpi=140); plt.close(fig); written.append(p)

    # -- 4. example galaxies ------------------------------------------------
    try:
        written.append(_galaxy_grid(cfg, b0, b1, a0, a1, seed0, split, fig_dir))
    except Exception as e:                                  # pragma: no cover
        print(f"  [note] galaxy grid skipped: {type(e).__name__}: {e}")

    return written


def _asinh(ref: np.ndarray) -> tuple[float, float, float]:
    finite = ref[np.isfinite(ref)]
    lo, hi = np.percentile(finite, [1.0, 99.5])
    soft = max(abs(hi), 1e-12) * 0.1
    return soft, float(np.arcsinh(lo / soft)), float(np.arcsinh(hi / soft))


def _galaxy_grid(cfg, b0, b1, a0, a1, seed0, split, fig_dir) -> Path:
    """true y | B0 | B1 | B0 residual | B1 residual, for typical and worst cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from src.config import INPUT_BAND_IDX, TARGET_BAND_IDX
    from src.models.b1_linear import LinearBaseline

    stats = load_global_stats(cfg.stats_file())
    ds = GalaxyDataset(cfg.processed_file(split), stats, is_train=False)

    dmag = a1["dmag"]
    score = np.where(np.isfinite(dmag), np.abs(dmag), np.inf)
    finite = np.flatnonzero(np.isfinite(score))
    order = finite[np.argsort(score[finite])]
    mid = order.size // 2
    rows = list(dict.fromkeys(
        list(order[max(mid - 1, 0): mid + 2][:3]) + list(order[-3:][::-1])
    ))[:6]

    ckpt = torch.load(cfg.path("checkpoints_dir")
                      / f"{b1.get('tag','b1')}_seed{seed0}.pt",
                      map_location="cpu", weights_only=False)
    model = LinearBaseline(
        in_channels=int(cfg.b1["model"]["in_channels"]),
        out_channels=int(cfg.b1["model"]["out_channels"]),
        bias=bool(cfg.b1["model"]["bias"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scale = b0["model"]["scale"]
    titles = ["true y", "B0 prediction", "B1 prediction",
              "B0 residual", "B1 residual"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(16.0, 2.95 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, gi in enumerate(rows):
        item = ds[int(gi)]
        x_raw = item["input_raw"].numpy()
        t = item["target_raw"].numpy()[0].astype(np.float64)
        p0 = (x_raw[3] * scale).astype(np.float64)

        xn = np.nan_to_num(normalize(x_raw[None], stats, INPUT_BAND_IDX))
        with torch.no_grad():
            pn = model(torch.from_numpy(xn)).numpy()
        p1 = denormalize(pn, stats, (TARGET_BAND_IDX,))[0, 0].astype(np.float64)

        soft, vmin, vmax = _asinh(t)
        res0, res1 = p0 - t, p1 - t
        rlim = max(float(np.nanpercentile(
            np.abs(np.concatenate([res0.ravel(), res1.ravel()])), 99)), 1e-12)

        panels = [
            (np.arcsinh(t / soft), vmin, vmax, "magma"),
            (np.arcsinh(p0 / soft), vmin, vmax, "magma"),
            (np.arcsinh(p1 / soft), vmin, vmax, "magma"),
            (res0, -rlim, rlim, "coolwarm"),
            (res1, -rlim, rlim, "coolwarm"),
        ]
        for c, (img, lo, hi, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)

        kind = "typical" if r < 3 else "WORST"
        axes[r, 0].set_ylabel(
            f"{kind}\nid {item['object_id']}\nz={float(item['redshift']):.3f}\n"
            f"$\\Delta m$: B0 {a0['dmag'][gi]:+.2f} / B1 {a1['dmag'][gi]:+.2f}",
            fontsize=8.5, rotation=0, labelpad=78, va="center", ha="center",
        )

    fig.suptitle(
        "Test galaxies: y-band truth, both baselines, and their residuals\n"
        "images share one asinh stretch per row; residuals share one symmetric "
        "scale per row (blue = under-predicted, red = over-predicted)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.075, 0, 1, 0.945))
    p = fig_dir / "galaxy_grid.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    ds.close()
    return p


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    res_dir = cfg.path("results_dir")

    b0 = _load(res_dir / "b0_metrics.json")
    variants = [_load(res_dir / "b1_metrics.json")]
    extra = res_dir / "b1_converged_metrics.json"
    if extra.exists():
        variants.append(json.loads(extra.read_text(encoding="utf-8")))

    md_path, tex_path = write_tables(cfg, b0, variants, args.split)
    plots = make_plots(cfg, b0, variants, args.split)

    print("=" * 74)
    print("BENCHMARK COMPARISON")
    print("=" * 74)
    print(md_path.read_text(encoding="utf-8"))
    print(f"\nWrote:\n  {md_path}\n  {tex_path}")
    for p in plots:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
