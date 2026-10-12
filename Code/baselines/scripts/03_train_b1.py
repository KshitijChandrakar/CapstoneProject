"""Step 3 -- train Baseline B1 (1x1 convolution, 5 parameters).

    python scripts/03_train_b1.py              # all seeds from config.yaml
    python scripts/03_train_b1.py --seed 42    # one seed

Input and target are both standardised; training uses masked L1, AdamW and a
cosine schedule, keeping the best-validation checkpoint. Evaluation happens in
physical flux units after denormalising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import INPUT_BAND_IDX, TARGET_BAND_IDX, load_config
from src.data.dataset import GalaxyDataset, make_dataloader
from src.data.preprocess import load_global_stats, normalize
from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_arrays,
    summarise_over_seeds,
)
from src.models.b1_linear import LinearBaseline, reference_optimum
from src.training.logger import ExperimentLogger
from src.training.trainer import Trainer, set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline B1.")
    p.add_argument("--seed", type=int, default=None,
                   help="train a single seed instead of all of them")
    p.add_argument("--config", default=None)
    # Overrides, so a converged reference run can be produced without editing
    # the YAML that records the prescribed recipe.
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--init", choices=("default", "zeros"), default=None)
    p.add_argument("--tag", default="b1",
                   help="output filename stem (default: b1)")
    p.add_argument("--label", default=None,
                   help="human-readable name for the report")
    return p.parse_args()


def apply_overrides(cfg, args) -> str:
    """Apply CLI overrides to the loaded B1 config; return the run label."""
    if args.lr is not None:
        cfg.b1["training"]["lr"] = args.lr
    if args.epochs is not None:
        cfg.b1["training"]["epochs"] = args.epochs
    if args.init is not None:
        cfg.b1["model"]["init"] = args.init
    return args.label or ("B1 (as specified)" if args.tag == "b1" else args.tag)


def solve_reference_optimum(cfg, train_ds, stats) -> dict | None:
    """Solve B1's optimum directly, as a yardstick for convergence."""
    diag = cfg.b1.get("diagnostics", {})
    if not diag.get("compute_reference_optimum", True):
        return None

    n = min(int(diag.get("reference_optimum_max_galaxies", 4000)), len(train_ds))
    idx = np.linspace(0, len(train_ds) - 1, n).astype(int)
    x = np.empty((n, len(INPUT_BAND_IDX), *train_ds.image_shape[1:]), dtype=np.float32)
    y = np.empty((n, 1, *train_ds.image_shape[1:]), dtype=np.float32)
    for j, i in enumerate(idx):
        item = train_ds[int(i)]
        x[j] = item["input_raw"].numpy()
        y[j] = item["target_raw"].numpy()

    x = np.nan_to_num(normalize(x, stats, INPUT_BAND_IDX))
    y = normalize(y, stats, (TARGET_BAND_IDX,))
    out = reference_optimum(x, y, stats)
    out["n_galaxies_used"] = n
    return out


def train_one_seed(cfg, stats, seed: int, datasets: dict, tag: str = "b1") -> dict:
    """Train B1 with one seed and evaluate it on the test split."""
    set_seed(seed)
    device = cfg.device()
    mcfg, tcfg = cfg.b1["model"], cfg.b1["training"]

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    train_ds.augmenter.__init__(seed=seed) if train_ds.augmenter else None

    bs = int(cfg.loader["batch_size"])
    nw = int(cfg.loader["num_workers"])
    pin = bool(cfg.loader["pin_memory"]) and device != "cpu"

    train_loader = make_dataloader(train_ds, bs, True, nw, pin, seed=seed)
    val_loader = make_dataloader(val_ds, bs, False, nw, pin)
    test_loader = make_dataloader(test_ds, bs, False, nw, pin)

    model = LinearBaseline(
        in_channels=int(mcfg["in_channels"]),
        out_channels=int(mcfg["out_channels"]),
        bias=bool(mcfg["bias"]),
        init=str(mcfg.get("init", "default")),
    )
    n_params = model.count_parameters()
    assert n_params == int(mcfg["n_parameters"]), (
        f"B1 must have {mcfg['n_parameters']} parameters, got {n_params}"
    )
    init_weights = model.get_weights()

    logger = ExperimentLogger(f"{tag}_seed{seed}", cfg.path("results_dir"))
    steps = len(train_loader)
    print(f"\n--- B1, seed {seed} ---")
    print(f"  {n_params} learnable parameters   device: {device}")
    print(f"  train {len(train_ds):,} / val {len(val_ds):,} galaxies")
    print(f"  lr {tcfg['lr']:g}, {tcfg['epochs']} epochs x {steps:,} steps "
          f"= {tcfg['epochs'] * steps:,} optimiser steps")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        epochs=int(tcfg["epochs"]),
        device=device,
        logger=logger,
        early_stopping_patience=tcfg.get("early_stopping_patience"),
    )
    fit = trainer.fit()
    print(f"  best val L1 {fit['best_val_loss']:.5f} at epoch {fit['best_epoch']}")

    ckpt = trainer.save_checkpoint(
        cfg.path("checkpoints_dir") / f"{tag}_seed{seed}.pt",
        extra={"seed": seed, "config": cfg.b1},
    )

    weights = model.weights_report(stats)
    std, phys = weights["standardised"], weights["physical_flux_space"]
    print(f"  standardised : " + "  ".join(
        f"w_{b} {std[f'w_{b}']:+.4f}" for b in weights["bands"]
    ) + f"  b {std['bias']:+.4f}")
    print(f"  physical     : " + "  ".join(
        f"w_{b} {phys[f'w_{b}']:+.4f}" for b in weights["bands"]
    ) + f"  b {phys['bias']:+.4g}")
    print(f"  ordering by physical weight: "
          f"{' > '.join(weights['physical_ordering'])}"
          f"   (spec expects z > i > r > g: {weights['matches_expected_ordering']})")

    # -- evaluate on test, in flux units -----------------------------------
    preds = collect_predictions(model, test_loader, stats, torch.device(device))
    res = evaluate_arrays(
        cfg, preds["y_true"], preds["y_pred"], preds["z_true"], return_arrays=True
    )
    arrays = res.pop("arrays")
    np.savez(
        cfg.path("results_dir") / f"arrays_{tag}_seed{seed}_test.npz",
        **arrays, object_id=preds["object_id"],
    )

    final = model.get_weights()
    movement = float(np.linalg.norm(
        np.array([final[k] - init_weights[k] for k in final])
    ))
    logger.close()

    return {
        "model": "B1_pixelwise_linear",
        "seed": seed,
        "n_parameters": n_params,
        "hyperparameters": {
            **{k: tcfg[k] for k in ("lr", "weight_decay", "epochs", "loss",
                                    "optimizer", "scheduler")},
            "batch_size": bs,
            "init": mcfg.get("init", "default"),
            "optimiser_steps": int(tcfg["epochs"]) * steps,
        },
        "best_val_loss": fit["best_val_loss"],
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "stopped_early": fit["stopped_early"],
        "initial_standardised_weights": init_weights,
        "weight_movement_from_init": movement,
        "weights": weights,
        "checkpoint": str(ckpt),
        "history": fit["history"],
        "metrics": res,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    label = apply_overrides(cfg, args)

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
    print(f"{label} -- pixelwise linear (1x1 conv, 5 parameters), seeds {seeds}")
    print("=" * 74)

    datasets = {
        "train": GalaxyDataset(
            cfg.processed_file("train"), stats, is_train=True, augment=True,
            in_memory=cfg.use_in_memory(cfg.processed_file("train")),
        ),
        "val": GalaxyDataset(
            cfg.processed_file("val"), stats,
            in_memory=cfg.use_in_memory(cfg.processed_file("val")),
        ),
        "test": GalaxyDataset(
            cfg.processed_file("test"), stats,
            in_memory=cfg.use_in_memory(cfg.processed_file("test")),
        ),
    }
    print("images held in RAM: " + ", ".join(
        f"{k}={d.in_memory}" for k, d in datasets.items()
    ))

    optimum = solve_reference_optimum(cfg, datasets["train"], stats)
    per_seed = [train_one_seed(cfg, stats, s, datasets, args.tag) for s in seeds]
    for ds in datasets.values():
        ds.close()

    summary = summarise_over_seeds([r["metrics"] for r in per_seed])
    bands = per_seed[0]["weights"]["bands"]

    weight_summary: dict[str, dict] = {}
    for space in ("standardised", "physical_flux_space"):
        keys = [f"w_{b}" for b in bands] + ["bias"]
        weight_summary[space] = {
            k: {
                "mean": float(np.mean([r["weights"][space][k] for r in per_seed])),
                "std": float(np.std([r["weights"][space][k] for r in per_seed])),
                "values": [float(r["weights"][space][k]) for r in per_seed],
            }
            for k in keys
        }

    phys = weight_summary["physical_flux_space"]
    std_w = weight_summary["standardised"]
    order = sorted(bands, key=lambda b: phys[f"w_{b}"]["mean"], reverse=True)
    order_std = sorted(bands, key=lambda b: std_w[f"w_{b}"]["mean"], reverse=True)
    expected = ["z", "i", "r", "g"]

    out: dict = {
        "model": "B1_pixelwise_linear",
        "label": label,
        "tag": args.tag,
        "n_parameters": per_seed[0]["n_parameters"],
        "seeds": seeds,
        "bands": bands,
        "hyperparameters": per_seed[0]["hyperparameters"],
        "weights_over_seeds": weight_summary,
        "physical_ordering_mean": order,
        "standardised_ordering_mean": order_std,
        "largest_physical_weight_band": order[0],
        "largest_standardised_weight_band": order_std[0],
        "expected_ordering": expected,
        # The spec's prediction is about informativeness, which the
        # standardised weights measure. See LinearBaseline.weights_report.
        "matches_expected_ordering": bool(order_std == expected),
        "matches_expected_ordering_physical": bool(order == expected),
        "z_dominates": bool(order_std[0] == "z"),
        "z_dominates_physical": bool(order[0] == "z"),
        "reference_optimum": optimum,
        "metrics_over_seeds": summary,
        "per_seed": per_seed,
    }

    # -- convergence -------------------------------------------------------
    if optimum is not None:
        ref = optimum.get("l1_optimum", {}).get("standardised") \
            or optimum["l2_closed_form"]["standardised"]
        ref_vec = np.array([ref[f"w_{b}"] for b in bands] + [ref["bias"]])
        dists = [
            float(np.linalg.norm(
                np.array([r["weights"]["standardised"][f"w_{b}"] for b in bands]
                         + [r["weights"]["standardised"]["bias"]]) - ref_vec))
            for r in per_seed
        ]
        out["convergence"] = {
            "distance_to_solved_optimum_per_seed": dists,
            "distance_to_solved_optimum_mean": float(np.mean(dists)),
            "weight_movement_from_init": [r["weight_movement_from_init"]
                                          for r in per_seed],
            "max_weight_spread_across_seeds": float(max(
                weight_summary["standardised"][k]["std"]
                for k in weight_summary["standardised"]
            )),
            "converged": bool(np.mean(dists) < 0.1),
        }

    path = cfg.path("results_dir") / f"{args.tag}_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(out)
    print(f"\nWrote {path}")
    print("Next:  python scripts/04_benchmark_comparison.py")
    return 0


def print_summary(out: dict) -> None:
    s = out["metrics_over_seeds"]
    pix, pho = s["pixel"], s["photometry"]
    print("\n" + "-" * 74)
    print("B1 TEST METRICS, mean +/- std over seeds (physical flux units)")
    print("-" * 74)
    print(f"  MAE               : {pix['mae']['mean']:.6g} +/- {pix['mae']['std']:.3g}")
    print(f"  RMSE              : {pix['rmse']['mean']:.6g} +/- {pix['rmse']['std']:.3g}")
    print(f"  PSNR (dB)         : {pix['psnr_mean']['mean']:.3f} +/- "
          f"{pix['psnr_mean']['std']:.3f}")
    print(f"  SSIM              : {pix['ssim_mean']['mean']:.4f} +/- "
          f"{pix['ssim_mean']['std']:.4f}")
    rf = pho["relative_flux_error"]
    print(f"  rel. flux error   : median {rf['median']['mean']:+.4f}   "
          f"NMAD {rf['nmad']['mean']:.4f} +/- {rf['nmad']['std']:.4f}")
    dm = pho["delta_mag"]
    print(f"  delta mag         : median {dm['median']['mean']:+.4f}   "
          f"NMAD {dm['nmad']['mean']:.4f} +/- {dm['nmad']['std']:.4f}")
    print(f"  outliers |dm|>0.5 : {dm['outlier_percent']['mean']:.2f} % +/- "
          f"{dm['outlier_percent']['std']:.2f}")
    ce = pho["colour_error_z_minus_y"]
    print(f"  colour err d(z-y) : median {ce['median']['mean']:+.4f}   "
          f"NMAD {ce['nmad']['mean']:.4f}")
    bs = pho.get("delta_mag_by_brightness") or {}
    if bs.get("faint_half"):
        print(f"  delta mag by flux : faint half NMAD "
              f"{bs['faint_half']['nmad']['mean']:.4f}   bright half NMAD "
              f"{bs['bright_half']['nmad']['mean']:.4f}")

    phys = out["weights_over_seeds"]["physical_flux_space"]
    print("\n  physical-space weights (mean +/- std over seeds):")
    for b in out["bands"]:
        print(f"    w_{b} = {phys[f'w_{b}']['mean']:+.4f} +/- {phys[f'w_{b}']['std']:.4f}")
    print(f"    bias = {phys['bias']['mean']:+.4g} +/- {phys['bias']['std']:.3g}")
    print(f"  ordering (standardised): {' > '.join(out['standardised_ordering_mean'])}"
          f"   <- the spec's prediction is about this one")
    print(f"  ordering (physical)    : {' > '.join(out['physical_ordering_mean'])}")
    print(f"  z dominates: {out['z_dominates']};  matches spec's z>i>r>g: "
          f"{out['matches_expected_ordering']}")

    conv = out.get("convergence")
    if conv:
        print(f"\n  distance from the solved optimum: "
              f"{conv['distance_to_solved_optimum_mean']:.4f}")
        if conv["converged"]:
            print("  -> converged.")
        else:
            print("  [!] NOT converged: the weights still largely reflect the "
                  "initialisation, so do not read the band ordering off them.")


if __name__ == "__main__":
    sys.exit(main())
