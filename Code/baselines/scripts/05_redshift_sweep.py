"""Step 5 -- does "y is a scaled copy of z" survive at higher redshift?

B0 assumes one global scale reproduces the y band from z. That holds while both
filters sample the same smooth stretch of the SED. It should fail where a
spectral feature lands in one filter but not the other -- and the z and y
filters are only 864 A apart, so the windows are narrow and predictable:

    Ha 6563 A      enters z at z=0.36, y at z=0.49
    [OIII] 5007 A  enters z at z=0.78, y at z=0.95
    Hb 4861 A      enters z at z=0.83, y at z=1.01
    4000 A break   enters z at z=1.22, y at z=1.44

This script measures B0's accuracy in redshift bins across the whole catalogue
and plots it against those predictions.

    python scripts/05_redshift_sweep.py

It needs only the raw files in data/raw -- no filtered subsets, and no
training, because B0 has zero parameters.

**It is fast.** Rather than re-filtering the 17 GB training file per bin, it
makes ONE streaming pass over the test images to get every galaxy's aperture
flux, then bins those numbers. Catalogue magnitudes are 1-D columns and cost
nothing to read. Expect a few minutes, dominated by the single image pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from src.config import load_config
from src.data.preprocess import mag_is_valid
from src.evaluation.photometry import (
    aperture_flux,
    circular_aperture_mask,
    flux_to_mag,
    robust_stats,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Default bins, chosen to straddle the predicted break points.
DEFAULT_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0]

#: (label, rest wavelength A). Used to mark the predicted break windows.
FEATURES = [
    ("Ha", 6563.0),
    ("[OIII]", 5007.0),
    ("Hb", 4861.0),
    ("4000A break", 4000.0),
]
Z_EFF, Y_EFF = 8898.0, 9762.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure B0's accuracy against redshift."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--edges", type=float, nargs="+", default=DEFAULT_EDGES,
                   help="redshift bin edges")
    p.add_argument("--eval-split", default="test", choices=("test", "val"),
                   help="which split to measure on (default: test)")
    p.add_argument("--min-per-bin", type=int, default=30,
                   help="skip bins with fewer galaxies than this")
    p.add_argument("--chunk", type=int, default=500)
    return p.parse_args()


def feature_windows() -> list[dict]:
    """Redshift range over which each feature is in z but not yet in y."""
    out = []
    for name, lam in FEATURES:
        out.append({
            "feature": name,
            "rest_wavelength_A": lam,
            "enters_z_band_at_z": Z_EFF / lam - 1.0,
            "enters_y_band_at_z": Y_EFF / lam - 1.0,
        })
    return out


def read_catalogue(path: Path) -> dict[str, np.ndarray]:
    """Redshift and the z/y magnitudes -- 1-D columns only, so this is cheap."""
    with h5py.File(path, "r") as f:
        for key in ("specz_redshift", "z_cmodel_mag", "y_cmodel_mag"):
            if key not in f:
                raise KeyError(f"{path.name} is missing '{key}'.")
        return {
            "z": np.asarray(f["specz_redshift"][:], dtype=np.float64),
            "mz": np.asarray(f["z_cmodel_mag"][:], dtype=np.float64),
            "my": np.asarray(f["y_cmodel_mag"][:], dtype=np.float64),
            "n": int(f["image"].shape[0]),
        }


def stream_aperture_fluxes(
    path: Path, radius: float, center_override, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    """One pass over the images: aperture flux in z and in y, per galaxy."""
    with h5py.File(path, "r") as f:
        images = f["image"]
        n, size = int(images.shape[0]), int(images.shape[-1])
        center = center_override if center_override is not None else (
            float(size // 2), float(size // 2)
        )
        mask = circular_aperture_mask((size, size), center, radius)

        fz = np.empty(n, dtype=np.float64)
        fy = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            batch = np.asarray(images[start:stop], dtype=np.float64)
            fz[start:stop] = aperture_flux(batch[:, 3], mask)
            fy[start:stop] = aperture_flux(batch[:, 4], mask)
            print(f"\r  aperture photometry: {100.0 * stop / n:5.1f}%  "
                  f"({stop:,}/{n:,})", end="", flush=True)
        print()
    return fz, fy


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    train_path = cfg.raw_file("train")
    eval_path = cfg.raw_file(args.eval_split)
    for p in (train_path, eval_path):
        if not p.exists():
            print(f"ERROR: raw file not found: {p}")
            print("Put the official GalaxiesML files in data/raw/ "
                  "(see docs/FOR_THE_DATA_HOLDER.md).")
            return 2

    radius = float(cfg.dataset["aperture_radius"])
    zp = float(cfg.photometry["zeropoint"])
    thresh = float(cfg.photometry["outlier_dmag_threshold"])

    print("=" * 74)
    print("B0 ACCURACY vs REDSHIFT -- does 'y = scale x z' hold outside z<0.4?")
    print("=" * 74)

    print(f"\nreading catalogue columns from {train_path.name} ...")
    train = read_catalogue(train_path)
    print(f"  {train['n']:,} train galaxies, z from {train['z'].min():.3f} "
          f"to {train['z'].max():.3f}")

    print(f"reading catalogue columns from {eval_path.name} ...")
    ev = read_catalogue(eval_path)
    print(f"  {ev['n']:,} {args.eval_split} galaxies")

    print(f"\none streaming pass for aperture fluxes "
          f"(r={radius:.0f} px) over {eval_path.name}:")
    fz, fy = stream_aperture_fluxes(
        eval_path, radius, cfg.dataset.get("aperture_center"), args.chunk
    )

    edges = sorted(args.edges)
    rows: list[dict] = []

    print(f"\n{'bin':>14}{'n_train':>9}{'n_eval':>8}{'med(z-y)':>10}"
          f"{'scale':>8}{'dm med':>9}{'dm NMAD':>9}{'outl%':>7}")
    print("-" * 74)

    for lo, hi in zip(edges[:-1], edges[1:]):
        tr_sel = (train["z"] >= lo) & (train["z"] < hi) \
            & mag_is_valid(train["mz"]) & mag_is_valid(train["my"])
        ev_sel = (ev["z"] >= lo) & (ev["z"] < hi)

        if tr_sel.sum() < args.min_per_bin or ev_sel.sum() < args.min_per_bin:
            print(f"{f'{lo:.2f}-{hi:.2f}':>14}{int(tr_sel.sum()):>9}"
                  f"{int(ev_sel.sum()):>8}   (too few galaxies -- skipped)")
            continue

        # B0 is fitted on the TRAIN galaxies of this bin only.
        colour = train["mz"][tr_sel] - train["my"][tr_sel]
        med_colour = float(np.median(colour))
        scale = float(10.0 ** (0.4 * med_colour))

        f_true, f_pred = fy[ev_sel], fz[ev_sel] * scale
        m_true = flux_to_mag(f_true, zp, "exclude")
        m_pred = flux_to_mag(f_pred, zp, "exclude")
        dmag = m_pred - m_true
        st = robust_stats(dmag)
        finite = dmag[np.isfinite(dmag)]
        outl = float(np.mean(np.abs(finite) > thresh) * 100.0) if finite.size else float("nan")

        rows.append({
            "z_low": lo, "z_high": hi,
            "z_mid": 0.5 * (lo + hi),
            "n_train": int(tr_sel.sum()),
            "n_eval": int(ev_sel.sum()),
            "median_z_minus_y_mag": med_colour,
            "colour_nmad_train": float(
                1.4826 * np.median(np.abs(colour - med_colour))
            ),
            "b0_scale": scale,
            "dmag_median": st["median"],
            "dmag_nmad": st["nmad"],
            "outlier_percent": outl,
            "n_scored": st["n"],
            "n_excluded_nonpositive": int(ev_sel.sum() - st["n"]),
            # Rest-frame wavelengths this bin's filters actually sample.
            "z_band_rest_A": Z_EFF / (1 + 0.5 * (lo + hi)),
            "y_band_rest_A": Y_EFF / (1 + 0.5 * (lo + hi)),
        })

        print(f"{f'{lo:.2f}-{hi:.2f}':>14}{int(tr_sel.sum()):>9}"
              f"{int(ev_sel.sum()):>8}{med_colour:>10.4f}{scale:>8.4f}"
              f"{st['median']:>+9.4f}{st['nmad']:>9.4f}{outl:>7.2f}")

    if not rows:
        print("\nERROR: no bin had enough galaxies.")
        return 2

    # -- verdict -----------------------------------------------------------
    base = rows[0]
    worst = max(rows, key=lambda r: r["dmag_nmad"])
    ratio = worst["dmag_nmad"] / base["dmag_nmad"] if base["dmag_nmad"] else float("nan")

    print("\n" + "=" * 74)
    print(f"lowest bin  z={base['z_low']:.2f}-{base['z_high']:.2f}: "
          f"dm NMAD {base['dmag_nmad']:.4f}")
    print(f"worst bin   z={worst['z_low']:.2f}-{worst['z_high']:.2f}: "
          f"dm NMAD {worst['dmag_nmad']:.4f}   ({ratio:.1f}x worse)")
    if ratio > 1.5:
        print("\n-> 'y is a scaled copy of z' DEGRADES with redshift. The single-scale")
        print("   assumption is specific to low z and cannot be assumed generally.")
    else:
        print("\n-> the single scale holds up across the range tested.")
    print("=" * 74)

    out = {
        "evaluated_on": args.eval_split,
        "aperture_radius_px": radius,
        "zeropoint": zp,
        "bins": rows,
        "predicted_break_windows": feature_windows(),
        "lowest_bin_dmag_nmad": base["dmag_nmad"],
        "worst_bin_dmag_nmad": worst["dmag_nmad"],
        "degradation_factor": ratio,
    }
    jp = cfg.path("results_dir") / "redshift_sweep.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = write_markdown(cfg, out)
    fig = make_plot(cfg, out)

    print(f"\nWrote:\n  {jp}\n  {md}\n  {fig}")
    return 0


def write_markdown(cfg, out: dict) -> Path:
    rows = out["bins"]
    lines = [
        "# Does \"y is a scaled copy of z\" hold at higher redshift?",
        "",
        f"B0 refitted and re-evaluated independently in each redshift bin, on the "
        f"**{out['evaluated_on']}** split. B0 has zero learnable parameters, so each "
        "bin gets its own median z−y colour and therefore its own scale.",
        "",
        "| z range | n train | n eval | median(m_z − m_y) | B0 scale | Δm median | "
        "Δm NMAD | Outlier % | z band rest | y band rest |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['z_low']:.2f}–{r['z_high']:.2f} | {r['n_train']:,} | "
            f"{r['n_eval']:,} | {r['median_z_minus_y_mag']:+.4f} | "
            f"{r['b0_scale']:.4f} | {r['dmag_median']:+.4f} | "
            f"**{r['dmag_nmad']:.4f}** | {r['outlier_percent']:.2f} | "
            f"{r['z_band_rest_A']:.0f} Å | {r['y_band_rest_A']:.0f} Å |"
        )
    lines += [
        "",
        f"Degradation from the lowest bin to the worst: "
        f"**{out['degradation_factor']:.1f}×** "
        f"({out['lowest_bin_dmag_nmad']:.4f} → {out['worst_bin_dmag_nmad']:.4f}).",
        "",
        "## Predicted break points",
        "",
        "The z and y filters are only 864 Å apart (8898 and 9762 Å observed), so "
        "the single-scale assumption should fail where a spectral feature sits in "
        "one filter but not the other:",
        "",
        "| Feature | Rest λ | Enters z band | Enters y band |",
        "|---|---|---|---|",
    ]
    for w in out["predicted_break_windows"]:
        lines.append(
            f"| {w['feature']} | {w['rest_wavelength_A']:.0f} Å | "
            f"z = {w['enters_z_band_at_z']:.2f} | z = {w['enters_y_band_at_z']:.2f} |"
        )
    lines += ["", "See `redshift_sweep.png` for the measurement against these "
              "predictions.", ""]
    p = cfg.path("results_dir") / "redshift_sweep.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def make_plot(cfg, out: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["bins"]
    mid = [r["z_mid"] for r in rows]
    xerr = [[r["z_mid"] - r["z_low"] for r in rows],
            [r["z_high"] - r["z_mid"] for r in rows]]
    nmad = [r["dmag_nmad"] for r in rows]
    colour = [r["median_z_minus_y_mag"] for r in rows]
    cnmad = [r["colour_nmad_train"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                   height_ratios=[1.35, 1])
    palette = ["#d1495b", "#e07a1f", "#4a9d5f", "#3d5a80"]

    # Predicted break windows, drawn behind the data.
    for (w, col) in zip(out["predicted_break_windows"], palette):
        for ax in (ax1, ax2):
            ax.axvspan(w["enters_z_band_at_z"], w["enters_y_band_at_z"],
                       color=col, alpha=0.16, zorder=0)
        # Label inside the axes, in axes-fraction y, so it never collides
        # with the title however large the NMAD values are.
        ax1.annotate(
            w["feature"],
            xy=(0.5 * (w["enters_z_band_at_z"] + w["enters_y_band_at_z"]), 0.97),
            xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=8.5, color=col, rotation=90,
            fontweight="bold",
        )

    ax1.errorbar(mid, nmad, xerr=xerr, fmt="o-", color="#1b263b", lw=1.8,
                 ms=6, capsize=3, zorder=3, label="B0 Δm NMAD (per bin)")
    ax1.set_ylabel(r"B0 magnitude error, $\Delta m$ NMAD  (mag)")
    ax1.set_title(
        "Does \"y is a scaled copy of z\" hold at higher redshift?\n"
        "B0 refitted independently in each redshift bin; shaded bands mark where "
        "a spectral feature\nsits in the z filter but not yet in y (the predicted "
        "failure points)",
        fontsize=11.5,
    )
    ax1.grid(alpha=0.3)
    ax1.margins(y=0.22)
    ax1.legend(loc="lower left", fontsize=9)

    ax2.errorbar(mid, colour, yerr=cnmad, xerr=xerr, fmt="s-", color="#5f0f40",
                 lw=1.8, ms=6, capsize=3, zorder=3,
                 label="median(m_z − m_y) ± NMAD on train")
    ax2.axhline(0, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("spectroscopic redshift")
    ax2.set_ylabel(r"$m_z - m_y$  (mag)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    p = cfg.path("figures_dir") / "redshift_sweep.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


if __name__ == "__main__":
    sys.exit(main())
