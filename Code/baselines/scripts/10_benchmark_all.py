"""Step 10 -- the full ladder: B0, B1, B2, B3 in one table.

    python scripts/10_benchmark_all.py
    python scripts/10_benchmark_all.py --models b0 b1 b2

Script 04 compares B0 and B1 and produces their figures; this one does not
replace it. It builds the ladder table across every model whose metrics JSON
exists, so it works after B2 alone and gains the B3 columns later, and it adds
the three things a four-model comparison needs that a two-model one did not:

* **how much each rung actually bought** -- the improvement over the rung
  below, next to the extra parameters it cost;
* **B3's calibration curve**, since an NLL without a calibration check is
  unreadable;
* **the anomaly shortlist**, written as a CSV for a human to look through.

Missing models are reported and skipped, never silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Display order of the ladder, with the file each model writes.
LADDER: tuple[tuple[str, str, str], ...] = (
    ("b0", "B0 (Copy-z)", "b0_metrics.json"),
    ("b1", "B1 (Linear 1x1)", "b1_metrics.json"),
    ("b2", "B2 (U-Net)", "b2_metrics.json"),
    ("b3", "B3 (U-Net+FiLM+NLL)", "b3_metrics.json"),
)

#: (column heading, metric path, decimals, lower_is_better)
METRICS: tuple[tuple[str, tuple[str, ...], int, bool], ...] = (
    ("Test MAE", ("pixel", "mae"), 4, True),
    ("PSNR (dB)", ("pixel", "psnr_mean"), 3, False),
    ("SSIM", ("pixel", "ssim_mean"), 4, False),
    ("Flux NMAD", ("photometry", "relative_flux_error", "nmad"), 4, True),
    ("sigma(mag)", ("photometry", "delta_mag", "nmad"), 4, True),
    ("Outlier %", ("photometry", "delta_mag", "outlier_percent"), 3, True),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the whole model ladder.")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of b0 b1 b2 b3 (default: whatever is present)")
    return p.parse_args()


def _dig(node: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def read_model(res_dir: Path, name: str, filename: str, split: str) -> dict | None:
    """One model's (mean, std) per metric, or None when it has not been run.

    B0 stores its metrics under ``splits[split]["B0"]`` with no seed spread;
    the trained models store ``metrics_over_seeds`` with mean and std. Both
    shapes are flattened to the same thing here so the table code stays
    simple.
    """
    path = res_dir / filename
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))

    if name == "b0":
        node = _dig(raw, ("splits", split, "B0"))
        if node is None:
            return None
        cells = {
            head: (_dig(node, mpath), None) for head, mpath, _, _ in METRICS
        }
        return {
            "label": "B0 (Copy-z)", "n_parameters": 0, "seeds": [],
            "cells": cells, "raw": raw, "uncertainty": None,
        }

    node = raw.get("metrics_over_seeds")
    if node is None:
        return None
    cells = {}
    for head, mpath, _, _ in METRICS:
        leaf = _dig(node, mpath)
        if isinstance(leaf, dict):
            cells[head] = (leaf.get("mean"), leaf.get("std"))
        else:
            cells[head] = (None, None)
    return {
        "label": raw.get("label") or name.upper(),
        "n_parameters": raw.get("n_parameters"),
        "seeds": raw.get("seeds", []),
        "cells": cells,
        "raw": raw,
        "uncertainty": _dig(node, ("uncertainty",)),
    }


def _fmt(mean, std, nd: int) -> str:
    if mean is None or not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def best_per_metric(models: dict) -> dict[str, str]:
    """Which model wins each metric, for bolding."""
    winners = {}
    for head, _, _, lower_better in METRICS:
        scored = [
            (name, m["cells"][head][0]) for name, m in models.items()
            if m["cells"][head][0] is not None and np.isfinite(m["cells"][head][0])
        ]
        if not scored:
            continue
        winners[head] = (min if lower_better else max)(scored, key=lambda t: t[1])[0]
    return winners


def write_table(cfg, models: dict, split: str) -> tuple[Path, Path]:
    """The ladder table, as markdown and LaTeX."""
    winners = best_per_metric(models)
    heads = ["Model", "Params"] + [
        f"{h} {'↓' if lower else '↑'}" for h, _, _, lower in METRICS
    ]

    md = ["| " + " | ".join(heads) + " |",
          "|" + "|".join(["---"] * len(heads)) + "|"]
    for name, m in models.items():
        cells = []
        for head, _, nd, _ in METRICS:
            mean, std = m["cells"][head]
            text = _fmt(mean, std, nd)
            if winners.get(head) == name and text != "--":
                text = f"**{text}**"
            cells.append(text)
        params = ("--" if m["n_parameters"] is None
                  else f"{m['n_parameters']:,}")
        md.append(f"| {m['label']} | {params} | " + " | ".join(cells) + " |")

    ds = cfg.dataset
    n_eval = _dig(models.get("b0", {}), ("raw", "splits", split, "B0", "n_galaxies"))
    md += [
        "",
        f"Evaluated on the official GalaxiesML **{split}** split"
        + (f": **{n_eval:,} galaxies**" if n_eval else "")
        + f", {ds['z_min']} ≤ z < {ds['z_max']}, in physical flux units. "
        "Trained models are the mean over seeds "
        f"{models.get('b1', models.get('b2', {})).get('seeds', [])}; "
        "± is the seed-to-seed standard deviation. Arrows mark the good "
        "direction and the best value in each column is bold.",
    ]

    gains = improvement_rows(models)
    if gains:
        md += [
            "",
            "### What each rung bought",
            "",
            "| Step | MAE change | Parameters added | MAE improvement per 10k params |",
            "|---|---|---|---|",
        ]
        for g in gains:
            md.append(
                f"| {g['step']} | {g['mae_before']:.4f} -> {g['mae_after']:.4f} "
                f"({g['percent']:+.1f}%) | {g['params_added']:,} | "
                f"{g['per_10k_params']:.3g} |"
            )
        md += [
            "",
            "The last column is the honest one. B1 bought 6.5% with five "
            "parameters; a U-Net buys whatever it buys with ~1.9 million. If "
            "the per-parameter column collapses, the ladder has reached the "
            "point where capacity is no longer the constraint -- which is "
            "exactly the argument for the physics models rather than a bigger "
            "network.",
        ]

    unc = uncertainty_section(models)
    if unc:
        md += [""] + unc

    md_path = cfg.path("results_dir") / "ladder_table.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{l" + "r" * (len(METRICS) + 1) + "}",
        r"\hline",
        " & ".join(["Model", "Params"] + [h for h, _, _, _ in METRICS])
        + r" \\",
        r"\hline",
    ]
    for name, m in models.items():
        cells = [_fmt(*m["cells"][head][:2], nd)
                 for head, _, nd, _ in METRICS]
        params = "--" if m["n_parameters"] is None else f"{m['n_parameters']:,}"
        label = m["label"].replace("_", r"\_")
        tex.append(f"{label} & {params} & " + " & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    tex_path = cfg.path("results_dir") / "ladder_table.tex"
    tex_path.write_text("\n".join(tex).replace("±", r"$\pm$") + "\n",
                        encoding="utf-8")
    return md_path, tex_path


def improvement_rows(models: dict) -> list[dict]:
    """MAE change and parameter cost from each rung to the next."""
    order = [n for n, _, _ in LADDER if n in models]
    rows = []
    for prev, cur in zip(order, order[1:]):
        a = models[prev]["cells"]["Test MAE"][0]
        b = models[cur]["cells"]["Test MAE"][0]
        pa = models[prev]["n_parameters"] or 0
        pb = models[cur]["n_parameters"] or 0
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        added = max(pb - pa, 0)
        rows.append({
            "step": f"{prev.upper()} -> {cur.upper()}",
            "mae_before": float(a),
            "mae_after": float(b),
            "percent": float((b - a) / a * 100.0),
            "params_added": int(added),
            "per_10k_params": float((a - b) / (added / 10_000.0)) if added else float("nan"),
        })
    return rows


def uncertainty_section(models: dict) -> list[str]:
    """B3's calibration block -- the NLL is unreadable without it."""
    b3 = models.get("b3")
    if not b3 or not b3.get("uncertainty"):
        return []
    u = b3["uncertainty"]
    nll = _dig(u, ("nll", "nll", "mean"))
    ratio = _dig(u, ("calibration", "ratio_std", "mean"))
    rho = _dig(u, ("sharpness", "spearman_sigma_vs_abs_error", "mean"))
    sharp = _dig(u, ("sharpness", "sharpness_median_sigma", "mean"))

    lines = [
        "### B3's uncertainty, and whether it can be trusted",
        "",
        "| Quantity | Value | Reads as |",
        "|---|---|---|",
    ]
    if nll is not None:
        lines.append(f"| NLL per pixel | {nll:.4f} | lower is better; the only "
                     "metric comparable to P1-P3 |")
    if ratio is not None:
        state = ("calibrated" if 0.9 <= ratio <= 1.1
                 else "over-confident (sigma too small)" if ratio > 1.1
                 else "under-confident (sigma too large)")
        lines.append(f"| (y-mu)/sigma spread | {ratio:.3f} | 1.0 is honest -- "
                     f"**{state}** |")
    for level in ("0.50", "0.68", "0.95"):
        emp = _dig(u, ("calibration", "coverage", level, "empirical", "mean"))
        if emp is not None:
            lines.append(f"| {level} interval coverage | {emp * 100:.1f}% | "
                         f"should be {float(level) * 100:.0f}% |")
    if sharp is not None:
        lines.append(f"| median sigma | {sharp:.4g} | sharpness: lower is "
                     "better, given calibration holds |")
    if rho is not None:
        lines.append(f"| Spearman(sigma, abs error) | {rho:+.3f} | 0 means one "
                     "constant sigma everywhere; >0.2 means it knows where it "
                     "is uncertain |")
    lines += [
        "",
        "Calibration is what makes the anomaly search possible. An "
        "over-confident model inflates every score, so the shortlist fills "
        "with ordinary galaxies; a model with a flat sigma cannot rank at all.",
    ]
    return lines


def write_calibration_figure(cfg, models: dict) -> Path | None:
    """Nominal vs empirical coverage for B3, with the diagonal to beat."""
    b3 = models.get("b3")
    if not b3 or not b3.get("uncertainty"):
        return None
    cov = _dig(b3["uncertainty"], ("calibration", "coverage"))
    if not cov:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted(float(k) for k in cov)
    emp = [_dig(cov, (f"{k:.2f}", "empirical", "mean")) for k in levels]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    ax.plot(levels, emp, "o-", color="#2a9d8f", lw=2.0, ms=7, label="B3")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage (fraction of pixels inside)")
    ax.set_title("B3 calibration: are the error bars the right size?")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.annotate("above the line = hedging\nbelow = over-confident",
                xy=(0.04, 0.88), xycoords="axes fraction", fontsize=8,
                color="#555555")
    fig.tight_layout()
    path = cfg.path("figures_dir") / "b3_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_ladder_figure(cfg, models: dict) -> Path | None:
    """MAE against parameter count, log x -- the diminishing-returns picture."""
    pts = [
        (m["n_parameters"] or 1, m["cells"]["Test MAE"][0], m["label"])
        for m in models.values()
        if m["cells"]["Test MAE"][0] is not None
    ]
    if len(pts) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts.sort(key=lambda t: t[0])
    xs = [max(p, 1) for p, _, _ in pts]
    ys = [v for _, v, _ in pts]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, ys, "o-", color="#264653", lw=2.0, ms=8)
    for x, y, label in pts:
        ax.annotate(label, xy=(max(x, 1), y), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("learnable parameters (log scale; B0 plotted at 1)")
    ax.set_ylabel("test MAE, flux units (lower is better)")
    ax.set_title("What each rung of the ladder bought")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = cfg.path("figures_dir") / "ladder_mae_vs_params.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_anomaly_csv(cfg, models: dict) -> Path | None:
    """B3's shortlist, as a CSV for a human to work through."""
    b3 = models.get("b3")
    if not b3:
        return None
    anom = _dig(b3, ("raw", "anomaly_candidates_seed0"))
    if not anom:
        return None
    path = cfg.path("results_dir") / "b3_anomaly_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "object_id", "test_index", "mean_pixel_nll"])
        for rank, (oid, idx, nll) in enumerate(
            zip(anom["object_id"], anom["index"], anom["nll"]), start=1
        ):
            w.writerow([rank, oid, idx, f"{nll:.6f}"])
    return path


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    res_dir = cfg.path("results_dir")

    wanted = args.models or [name for name, _, _ in LADDER]
    models: dict[str, dict] = {}
    absent: list[str] = []
    for name, _, filename in LADDER:
        if name not in wanted:
            continue
        entry = read_model(res_dir, name, filename, args.split)
        if entry is None:
            absent.append(name)
        else:
            models[name] = entry

    if not models:
        print("No model metrics found in " + str(res_dir))
        print("Run scripts 02, 03, 08 and 09 first.")
        return 2

    print("=" * 74)
    print(f"MODEL LADDER -- {args.split} split")
    print("=" * 74)
    if absent:
        print("Not run yet, so absent from the table: "
              + ", ".join(a.upper() for a in absent))

    md_path, tex_path = write_table(cfg, models, args.split)
    cal = write_calibration_figure(cfg, models)
    ladder = write_ladder_figure(cfg, models)
    anom = write_anomaly_csv(cfg, models)

    print()
    print(md_path.read_text(encoding="utf-8"))

    written = [p for p in (md_path, tex_path, cal, ladder, anom) if p]
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
