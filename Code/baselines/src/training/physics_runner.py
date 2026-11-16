"""Shared training loop for the physics models P1 and P2.

One runner rather than two near-identical scripts. P1 and P2 differ in exactly
two ways -- which model is built, and whether it is handed redshift -- so
everything else lives here and the two scripts stay thin.

Using the same Trainer, datasets, evaluator and photometry as B0-B3 is the
point, not a convenience: the ladder table is only a fair comparison if the
only thing that changed between rows is the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import GalaxyDataset, make_dataloader
from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_arrays,
    merge_with_existing,
    summarise_over_seeds,
)
from src.training.logger import ExperimentLogger
from src.training.trainer import Trainer, set_seed


def build_datasets(cfg, stats) -> dict:
    """The same three splits every other model in this project uses."""
    return {
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


def train_one_seed(cfg, stats, seed: int, datasets: dict, *, tag: str,
                   model_key: str, builder, conditioning_key: str | None,
                   report) -> dict:
    """Train one seed of a physics model and evaluate it on the test split."""
    set_seed(seed)
    device = cfg.device()
    block = cfg.model_config(model_key)
    mcfg, tcfg = block["model"], block["training"]

    train_ds, val_ds, test_ds = (datasets["train"], datasets["val"],
                                 datasets["test"])
    if train_ds.augmenter:
        train_ds.augmenter.__init__(seed=seed)

    bs = int(cfg.loader["batch_size"])
    nw = int(cfg.loader["num_workers"])
    pin = bool(cfg.loader["pin_memory"]) and device != "cpu"

    train_loader = make_dataloader(train_ds, bs, True, nw, pin, seed=seed)
    val_loader = make_dataloader(val_ds, bs, False, nw, pin)
    test_loader = make_dataloader(test_ds, bs, False, nw, pin)

    model = builder(mcfg)
    n_params = model.count_parameters()
    breakdown = model.parameter_breakdown()

    logger = ExperimentLogger(f"{tag}_seed{seed}", cfg.path("results_dir"))
    steps = len(train_loader)
    print(f"\n--- {tag.upper()}, seed {seed} ---")
    print(f"  {n_params} learnable parameters   device: {device}")
    print(f"  {breakdown}")
    print(f"  train {len(train_ds):,} / val {len(val_ds):,} galaxies")
    print(f"  lr {tcfg['lr']:g}, up to {tcfg['epochs']} epochs x {steps:,} "
          f"steps = {tcfg['epochs'] * steps:,} optimiser steps")

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
        conditioning_key=conditioning_key,
    )
    fit = trainer.fit()
    print(f"  best val L1 {fit['best_val_loss']:.5f} at epoch "
          f"{fit['best_epoch']}/{fit['epochs_run']}"
          f"{'  (early stop)' if fit['stopped_early'] else ''}")

    ckpt = trainer.save_checkpoint(
        cfg.path("checkpoints_dir") / f"{tag}_seed{seed}.pt",
        extra={"seed": seed, "config": block, "n_parameters": n_params},
    )

    preds = collect_predictions(model, test_loader, stats,
                                torch.device(device),
                                conditioning_key=conditioning_key)
    res = evaluate_arrays(cfg, preds["y_true"], preds["y_pred"],
                          preds["z_true"], return_arrays=True)
    arrays = res.pop("arrays")
    np.savez(cfg.path("results_dir") / f"arrays_{tag}_seed{seed}_test.npz",
             **arrays, object_id=preds["object_id"],
             redshift=preds["redshift"])
    logger.close()

    # The physics readout: what the model decided, beside what was measured.
    physics = report(model)
    print("  physics readout:")
    for line in json.dumps(physics, indent=4).splitlines():
        print(f"    {line}")

    return {
        "model": tag,
        "seed": seed,
        "n_parameters": n_params,
        "parameter_breakdown": breakdown,
        "architecture": dict(mcfg),
        "hyperparameters": {
            **{k: tcfg[k] for k in ("lr", "weight_decay", "epochs", "loss",
                                    "optimizer", "scheduler") if k in tcfg},
            "batch_size": bs,
            "grad_clip_norm": tcfg.get("grad_clip_norm"),
            "early_stopping_patience": tcfg.get("early_stopping_patience"),
            "optimiser_steps_run": int(fit["epochs_run"]) * steps,
        },
        "best_val_loss": fit["best_val_loss"],
        "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"],
        "stopped_early": fit["stopped_early"],
        "checkpoint": str(ckpt),
        "history": fit["history"],
        "physics": physics,
        "metrics": res,
    }


def run(cfg, stats, seeds, *, tag: str, model_key: str, builder,
        conditioning_key: str | None, report, label: str,
        fresh: bool = False) -> dict:
    """Train every requested seed, merge with any already on disk, save."""
    datasets = build_datasets(cfg, stats)
    print("images held in RAM: " + ", ".join(
        f"{k}={d.in_memory}" for k, d in datasets.items()))

    per_seed = [
        train_one_seed(cfg, stats, s, datasets, tag=tag, model_key=model_key,
                       builder=builder, conditioning_key=conditioning_key,
                       report=report)
        for s in seeds
    ]
    for ds in datasets.values():
        ds.close()

    path = cfg.path("results_dir") / f"{tag}_metrics.json"
    if not fresh:
        per_seed = merge_with_existing(per_seed, path, label)

    out = {
        "model": tag,
        "label": label,
        "tag": tag,
        "n_parameters": per_seed[0]["n_parameters"],
        "seeds": [r["seed"] for r in per_seed],
        "architecture": per_seed[0]["architecture"],
        "hyperparameters": per_seed[0]["hyperparameters"],
        "physics_per_seed": [r["physics"] for r in per_seed],
        "metrics_over_seeds": summarise_over_seeds(
            [r["metrics"] for r in per_seed]),
        "per_seed": per_seed,
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def print_summary(out: dict, extra_lines=()) -> None:
    s = out["metrics_over_seeds"]
    pix, pho = s["pixel"], s["photometry"]
    print("\n" + "-" * 74)
    print(f"{out['label']} TEST METRICS, mean +/- std over seeds "
          "(physical flux units)")
    print("-" * 74)
    print(f"  parameters        : {out['n_parameters']}")
    print(f"  seeds             : {out['seeds']}")
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
    print(f"  outliers |dm|>0.5 : {dm['outlier_percent']['mean']:.2f} %")
    for line in extra_lines:
        print(line)
    print("\n  Full ladder: python scripts/10_benchmark_all.py")
