"""Step 12 -- train P1, the PSF-aware physics model (7 parameters).

    python scripts/12_train_p1.py              # all seeds from config.yaml
    python scripts/12_train_p1.py --seed 42    # one seed
    python scripts/12_train_p1.py --frozen     # blur pinned at script 07's value

Same data, same loss, same evaluation code and same photometry as B0-B3, so
the only thing that changes between rows of the ladder table is the model.

The run prints a physics readout afterwards: the blur width P1 fitted, beside
the width script 07 measured. That comparison is the result -- a fitted sigma
collapsing towards zero means P1 is rejecting the PSF hypothesis, which is a
finding to report rather than a failure to hide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.preprocess import load_global_stats
from src.models.p1_psf import build_p1
from src.physics import psf
from src.training.physics_runner import print_summary, run

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P1 (PSF-aware).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--frozen", action="store_true",
                   help="pin the blur at script 07's measured width, so any "
                        "gain is attributable to the measurement rather than "
                        "to the extra freedom")
    p.add_argument("--tag", default="p1")
    p.add_argument("--label", default=None)
    p.add_argument("--fresh", action="store_true",
                   help="discard seeds already in the metrics file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    block = cfg.model_config("p1")
    if args.lr is not None:
        block["training"]["lr"] = args.lr
    if args.epochs is not None:
        block["training"]["epochs"] = args.epochs
    if args.frozen:
        block["model"]["learn_blur"] = False
        block["model"]["expected_parameters"] = 5
    label = args.label or ("P1 (PSF frozen)" if args.frozen else "P1 (PSF)")

    missing = cfg.missing_processed_files()
    if missing:
        print("Processed files are missing:")
        for p in missing:
            print(f"  {p}")
        print("\nRun: python scripts/01_prepare_data.py")
        return 2

    stats = load_global_stats(cfg.stats_file())
    seeds = [args.seed] if args.seed is not None else cfg.seeds

    print("=" * 74)
    print(f"{label} -- spectral mix + learned PSF blur, seeds {seeds}")
    print("=" * 74)
    print(psf.summary())

    out = run(cfg, stats, seeds, tag=args.tag, model_key="p1",
              builder=build_p1, conditioning_key="redshift",
              report=lambda m: m.fitted_sigmas(), label=label,
              fresh=args.fresh)

    extra = ["", "  PHYSICS READOUT (the part only P1 can be asked)"]
    for r in out["physics_per_seed"]:
        for b, f, m_, c in zip(r["bins"], r["fitted_sigma_px"],
                               r["measured_sigma_px"], r["collapsed"]):
            note = "  <- COLLAPSED: P1 rejects the PSF hypothesis here" if c else ""
            extra.append(f"    z {b}: fitted sigma {f:.4f} px   "
                         f"script 07 measured {m_:.4f} px{note}")
        if r["truncation"]["any_truncated"]:
            extra.append("    [!] a fitted sigma outgrew the kernel radius; "
                         "the reported width is NOT the width applied.")
    print_summary(out, extra)
    print(f"\nWrote {cfg.path('results_dir') / (args.tag + '_metrics.json')}")
    print("Next:  python scripts/13_train_p2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
