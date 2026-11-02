"""Step 8 -- train Baseline B2 (U-Net, ~1.9M parameters).

    python scripts/08_train_b2.py              # all seeds from config.yaml
    python scripts/08_train_b2.py --seed 42    # one seed
    python scripts/08_train_b2.py --residual-on-z   # start the net near B0

Same data, same masked L1 loss, same evaluation code and same photometry as
B1, so the only thing that changes between the two rows of the results table
is the architecture. That is the whole design: B2 exists to answer one
question -- how much of the remaining error was spatial?

B2 is the first model here that can overfit, so unlike B1 it gets early
stopping and gradient clipping, both recorded in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import GalaxyDataset, make_dataloader
from src.data.preprocess import load_global_stats
from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_arrays,
    summarise_over_seeds,
)
from src.models.b2_unet import build_b2
from src.training.logger import ExperimentLogger
from src.training.trainer import Trainer, set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline B2 (U-Net).")
    p.add_argument("--seed", type=int, default=None,
                   help="train a single seed instead of all of them")
    p.add_argument("--config", default=None)
    # Overrides, so a variant can be run without editing the YAML that records
    # the prescribed recipe.
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None,
                   help="model width; changing it also changes the parameter "
                        "count, so expected_parameters is then not enforced")
    p.add_argument("--residual-on-z", action="store_true",
                   help="predict a correction to the z band instead of y "
                        "directly, which starts the model near B0")
    p.add_argument("--tag", default="b2", help="output filename stem")
    p.add_argument("--label", default=None,
                   help="human-readable name for the report")
    return p.parse_args()


def apply_overrides(cfg, args) -> str:
    """Apply CLI overrides to the loaded B2 config; return the run label."""
    b2 = cfg.model_config("b2")
    if args.lr is not None:
        b2["training"]["lr"] = args.lr
    if args.epochs is not None:
        b2["training"]["epochs"] = args.epochs
    if args.base_channels is not None:
        b2["model"]["base_channels"] = args.base_channels
        # The expected count belongs to the prescribed width only.
        b2["model"].pop("expected_parameters", None)
    if args.residual_on_z:
        b2["model"]["residual_on_z"] = True
    return args.label or ("B2 (U-Net)" if args.tag == "b2" else args.tag)


def train_one_seed(cfg, stats, seed: int, datasets: dict, tag: str = "b2") -> dict:
    """Train B2 with one seed and evaluate it on the test split."""
    set_seed(seed)
    device = cfg.device()
    b2 = cfg.model_config("b2")
    mcfg, tcfg = b2["model"], b2["training"]

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    if train_ds.augmenter:
        train_ds.augmenter.__init__(seed=seed)

    bs = int(cfg.loader["batch_size"])
    nw = int(cfg.loader["num_workers"])
    pin = bool(cfg.loader["pin_memory"]) and device != "cpu"

    train_loader = make_dataloader(train_ds, bs, True, nw, pin, seed=seed)
    val_loader = make_dataloader(val_ds, bs, False, nw, pin)
    test_loader = make_dataloader(test_ds, bs, False, nw, pin)

    model = build_b2(mcfg)
    n_params = model.count_parameters()

    logger = ExperimentLogger(f"{tag}_seed{seed}", cfg.path("results_dir"))
    steps = len(train_loader)
    print(f"\n--- B2, seed {seed} ---")
    print(f"  {n_params:,} learnable parameters   device: {device}")
    print(f"  base_channels {mcfg['base_channels']}, depth {mcfg['depth']}, "
          f"residual_on_z {bool(mcfg.get('residual_on_z', False))}")
    print(f"  train {len(train_ds):,} / val {len(val_ds):,} galaxies")
    print(f"  lr {tcfg['lr']:g}, up to {tcfg['epochs']} epochs x {steps:,} steps "
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
        grad_clip_norm=tcfg.get("grad_clip_norm"),
    )
    fit = trainer.fit()
    print(f"  best val L1 {fit['best_val_loss']:.5f} at epoch "
          f"{fit['best_epoch']}/{fit['epochs_run']}"
          f"{'  (early stop)' if fit['stopped_early'] else ''}")

    ckpt = trainer.save_checkpoint(
        cfg.path("checkpoints_dir") / f"{tag}_seed{seed}.pt",
        extra={"seed": seed, "config": b2, "n_parameters": n_params},
    )

    preds = collect_predictions(model, test_loader, stats, torch.device(device))
    res = evaluate_arrays(
        cfg, preds["y_true"], preds["y_pred"], preds["z_true"], return_arrays=True
    )
    arrays = res.pop("arrays")
    np.savez(
        cfg.path("results_dir") / f"arrays_{tag}_seed{seed}_test.npz",
        **arrays, object_id=preds["object_id"], redshift=preds["redshift"],
    )
    logger.close()

    return {
        "model": "B2_unet",
        "seed": seed,
        "n_parameters": n_params,
        "architecture": {
            k: mcfg[k] for k in ("base_channels", "depth", "norm", "groups",
                                 "activation", "dropout", "residual_on_z")
            if k in mcfg
        },
        "hyperparameters": {
            **{k: tcfg[k] for k in ("lr", "weight_decay", "epochs", "loss",
                                    "optimizer", "scheduler")},
            "batch_size": bs,
            "grad_clip_norm": tcfg.get("grad_clip_norm"),
            "early_stopping_patience": tcfg.get("early_stopping_patience"),
            "optimiser_steps_max": int(tcfg["epochs"]) * steps,
            "optimiser_steps_run": int(fit["epochs_run"]) * steps,
        },
        "best_val_loss": fit["best_val_loss"],
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "stopped_early": fit["stopped_early"],
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
    print(f"{label} -- U-Net, seeds {seeds}")
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

    per_seed = [train_one_seed(cfg, stats, s, datasets, args.tag) for s in seeds]
    for ds in datasets.values():
        ds.close()

    summary = summarise_over_seeds([r["metrics"] for r in per_seed])
    out = {
        "model": "B2_unet",
        "label": label,
        "tag": args.tag,
        "n_parameters": per_seed[0]["n_parameters"],
        "seeds": seeds,
        "architecture": per_seed[0]["architecture"],
        "hyperparameters": per_seed[0]["hyperparameters"],
        "metrics_over_seeds": summary,
        "per_seed": per_seed,
    }
    path = cfg.path("results_dir") / f"{args.tag}_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(out)
    print(f"\nWrote {path}")
    print("Next:  python scripts/09_train_b3.py")
    return 0


def print_summary(out: dict) -> None:
    s = out["metrics_over_seeds"]
    pix, pho = s["pixel"], s["photometry"]
    print("\n" + "-" * 74)
    print("B2 TEST METRICS, mean +/- std over seeds (physical flux units)")
    print("-" * 74)
    print(f"  parameters        : {out['n_parameters']:,}")
    print(f"  MAE               : {pix['mae']['mean']:.6g} +/- "
          f"{pix['mae']['std']:.3g}")
    print(f"  RMSE              : {pix['rmse']['mean']:.6g} +/- "
          f"{pix['rmse']['std']:.3g}")
    print(f"  PSNR (dB)         : {pix['psnr_mean']['mean']:.3f} +/- "
          f"{pix['psnr_mean']['std']:.3f}")
    print(f"  SSIM              : {pix['ssim_mean']['mean']:.4f} +/- "
          f"{pix['ssim_mean']['std']:.4f}")
    dm = pho["delta_mag"]
    print(f"  delta mag         : median {dm['median']['mean']:+.4f}   "
          f"NMAD {dm['nmad']['mean']:.4f} +/- {dm['nmad']['std']:.4f}")
    print(f"  outliers |dm|>0.5 : {dm['outlier_percent']['mean']:.2f} % +/- "
          f"{dm['outlier_percent']['std']:.2f}")
    bs = pho.get("delta_mag_by_brightness") or {}
    if bs.get("faint_half"):
        print(f"  delta mag by flux : faint half NMAD "
              f"{bs['faint_half']['nmad']['mean']:.4f}   bright half NMAD "
              f"{bs['bright_half']['nmad']['mean']:.4f}")
    print("\n  Compare against B1 with: python scripts/10_benchmark_all.py")


if __name__ == "__main__":
    sys.exit(main())
