"""Step 13 -- train P2, the SED extrapolation physics model (4 parameters).

    python scripts/13_train_p2.py               # all seeds from config.yaml
    python scripts/13_train_p2.py --seed 42     # one seed
    python scripts/13_train_p2.py --residual    # add the small conv correction

Same data, loss, evaluation and photometry as every other row of the ladder.

P2 treats the four inputs as four samples of one spectrum at known wavelengths
and evaluates that spectrum at y's 9762 A. It has no spatial awareness at all,
so where it lands between B1 and B2 measures how much of B2's gain was
spectral rather than spatial.

The run prints the fitted basis gains afterwards. All near 1 means plain
least-squares extrapolation was already right; a small gain on the quadratic
term means the model learned not to trust measured curvature 864 A beyond the
end of the data -- a physical statement B2 cannot be asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.preprocess import load_global_stats
from src.models.p2_sed import build_p2
from src.physics import bands, sed
from src.training.physics_runner import print_summary, run

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train P2 (SED extrapolation).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--degree", type=int, default=None)
    p.add_argument("--residual", action="store_true",
                   help="add the zero-initialised convolutional correction; "
                        "the gap against the pure run is the part of the "
                        "error that is not spectral")
    p.add_argument("--tag", default="p2")
    p.add_argument("--label", default=None)
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    block = cfg.model_config("p2")
    if args.lr is not None:
        block["training"]["lr"] = args.lr
    if args.epochs is not None:
        block["training"]["epochs"] = args.epochs
    if args.degree is not None:
        block["model"]["degree"] = args.degree
        block["model"].pop("expected_parameters", None)
    if args.residual:
        block["model"]["residual"] = True
        block["model"].pop("expected_parameters", None)
    label = args.label or ("P2 (SED+residual)" if args.residual else "P2 (SED)")

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
    print(f"{label} -- fit a spectrum, evaluate it at y, seeds {seeds}")
    print("=" * 74)
    print(bands.summary())
    print(sed.summary())

    out = run(cfg, stats, seeds, tag=args.tag, model_key="p2",
              builder=build_p2, conditioning_key=None,
              report=lambda m: m.fitted_gains(), label=label,
              fresh=args.fresh)

    extra = ["", "  PHYSICS READOUT (the part only P2 can be asked)"]
    for r in out["physics_per_seed"]:
        for term, gain, departs in zip(r["terms"], r["gains"],
                                       r["departs_from_pure_physics"]):
            note = "  <- departs from pure least squares" if departs else ""
            extra.append(f"    gain on {term:<14}: {gain:+.4f}{note}")
        extra.append(f"    bias            : {r['bias']:+.6f}")
    print_summary(out, extra)
    print(f"\nWrote {cfg.path('results_dir') / (args.tag + '_metrics.json')}")
    print("Next:  python scripts/10_benchmark_all.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
