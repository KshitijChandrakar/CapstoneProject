"""Step 15 -- train M1, the catalogue-metadata network (no pixels).

    python scripts/15_train_m1.py              # all seeds from config.yaml
    python scripts/15_train_m1.py --seed 42

M1 predicts ``y_cmodel_mag`` from 59 catalogue columns -- magnitudes,
morphology, redshift -- and never sees an image. It exists to measure how much
of the y band the catalogue already contains, which is a question the image
ladder cannot answer from the inside.

The run reports M1 beside a zero-parameter catalogue baseline (y = z magnitude
minus the median z-y colour), because a network that cannot beat one number is
not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.data.metadata import (
    apply_standardiser,
    feature_columns,
    fit_standardiser,
    load_split,
    sanity_report,
)
from src.models.m1_metadata_nn import build_m1, catalogue_baseline
from src.training.trainer import set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NMAD_FACTOR = 1.4826


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M1 (catalogue metadata NN).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default="m1")
    p.add_argument("--label", default=None)
    return p.parse_args()


def magnitude_metrics(pred: np.ndarray, truth: np.ndarray,
                      threshold: float = 0.5) -> dict:
    """The same magnitude statistics the image models are scored on."""
    d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    med = float(np.median(d))
    return {
        "n": int(d.size),
        "delta_mag_median": med,
        "delta_mag_nmad": float(NMAD_FACTOR * np.median(np.abs(d - med))),
        "delta_mag_mae": float(np.mean(np.abs(d))),
        "delta_mag_rmse": float(np.sqrt(np.mean(d ** 2))),
        "outlier_percent": float(100.0 * np.mean(np.abs(d) > threshold)),
    }


def raw_path_for(cfg, split: str) -> Path:
    return cfg.raw_file(split)


def train_one_seed(cfg, data, stats, seed: int, mcfg, tcfg, tag: str) -> dict:
    set_seed(seed)
    device = cfg.device()

    Xtr = torch.tensor(apply_standardiser(data["train"]["X"], stats),
                       dtype=torch.float32)
    ytr = torch.tensor(data["train"]["y"], dtype=torch.float32)
    Xva = torch.tensor(apply_standardiser(data["val"]["X"], stats),
                       dtype=torch.float32)
    yva = torch.tensor(data["val"]["y"], dtype=torch.float32)
    Xte = torch.tensor(apply_standardiser(data["test"]["X"], stats),
                       dtype=torch.float32)

    offset = float(np.median(data["train"]["y"]))
    model = build_m1(mcfg, Xtr.shape[1], target_offset=offset).to(device)
    n_params = model.count_parameters()

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]),
                           weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = int(tcfg["batch_size"])
    patience = int(tcfg.get("early_stopping_patience") or epochs)
    clip = tcfg.get("grad_clip_norm")

    print(f"\n--- M1, seed {seed} ---")
    print(f"  {n_params:,} parameters over {Xtr.shape[1]} features   device: {device}")
    print(f"  train {len(ytr):,} / val {len(yva):,} galaxies, "
          f"head initialised at the training median {offset:.4f} mag")

    g = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, bad = float("inf"), 0, None, 0
    history = []
    started = time.time()
    Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)
    Xva_d, yva_d = Xva.to(device), yva.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr_d), generator=g).to(device)
        total = 0.0
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            loss = (model(Xtr_d[idx]) - ytr_d[idx]).abs().mean()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            opt.step()
            total += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val = float((model(Xva_d) - yva_d).abs().mean())
        history.append({"epoch": epoch, "train_loss": total / len(perm),
                        "val_loss": val, "lr": sched.get_last_lr()[0]})

        marker = ""
        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            bad += 1
        if epoch % 5 == 0 or marker:
            print(f"  epoch {epoch:>3}/{epochs}  train L1 {total / len(perm):.5f}"
                  f"  val L1 {val:.5f}{marker}")
        if bad >= patience:
            print(f"  early stop at epoch {epoch} (no improvement for {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).cpu().numpy().astype(np.float64)

    took = time.time() - started
    print(f"  best val L1 {best:.5f} at epoch {best_epoch}, {took / 60:.1f} min")

    metrics = magnitude_metrics(
        pred, data["test"]["y"],
        float(cfg.raw.get("photometry", {}).get("outlier_dmag_threshold", 0.5)))

    ckpt = cfg.path("checkpoints_dir") / f"{tag}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed,
                "n_parameters": n_params, "columns": data["train"]["columns"],
                "standardiser": stats, "target_offset": offset}, ckpt)

    np.savez(cfg.path("results_dir") / f"arrays_{tag}_seed{seed}_test.npz",
             y_pred_mag=pred, y_true_mag=data["test"]["y"],
             object_id=data["test"]["object_id"],
             redshift=data["test"]["redshift"])

    return {"seed": seed, "n_parameters": n_params, "best_val_loss": best,
            "best_epoch": best_epoch, "epochs_run": len(history),
            "minutes": took / 60.0, "history": history, "metrics": metrics,
            "checkpoint": str(ckpt)}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    block = cfg.model_config("m1")
    mcfg, tcfg = block["model"], block["training"]
    if args.lr is not None:
        tcfg["lr"] = args.lr
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    label = args.label or "M1 (catalogue MLP)"

    missing = cfg.missing_processed_files()
    if missing:
        print("Processed files are missing:")
        for p in missing:
            print(f"  {p}")
        return 2

    print("=" * 74)
    print(f"{label} -- catalogue metadata only, no pixels")
    print("=" * 74)

    cols = feature_columns(raw_path_for(cfg, "train"))
    data = {s: load_split(raw_path_for(cfg, s), cfg.processed_file(s), cols)
            for s in ("train", "val", "test")}
    for s, d in data.items():
        r = sanity_report(d)
        print(f"  {s:<6} {r['n_galaxies']:>7,} galaxies x {r['n_features']} "
              f"features   non-finite {r['nonfinite_values']}   "
              f"no y_ columns: {r['leakage_check_no_y_columns']}")
    if not all(sanity_report(d)["leakage_check_no_y_columns"]
               for d in data.values()):
        print("\nLEAKAGE: a y_* column reached the feature set. Refusing to run.")
        return 1

    stats = fit_standardiser(data["train"]["X"])

    # The zero-parameter reference M1 has to beat.
    zi = cols.index("z_cmodel_mag")
    median_colour = float(np.median(data["train"]["X"][:, zi]
                                    - data["train"]["y"]))
    base_pred = catalogue_baseline(data["test"]["X"][:, zi], median_colour)
    base = magnitude_metrics(base_pred, data["test"]["y"])
    print(f"\n  catalogue baseline (y = z_mag - {median_colour:.4f}): "
          f"NMAD {base['delta_mag_nmad']:.4f}, "
          f"outliers {base['outlier_percent']:.2f}%")

    seeds = [args.seed] if args.seed is not None else cfg.seeds
    per_seed = [train_one_seed(cfg, data, stats, s, mcfg, tcfg, args.tag)
                for s in seeds]

    keys = list(per_seed[0]["metrics"])
    summary = {k: {"mean": float(np.mean([r["metrics"][k] for r in per_seed])),
                   "std": float(np.std([r["metrics"][k] for r in per_seed]))}
               for k in keys}

    out = {"model": "M1_metadata_nn", "label": label, "tag": args.tag,
           "n_parameters": per_seed[0]["n_parameters"],
           "n_features": len(cols), "features": cols,
           "seeds": [r["seed"] for r in per_seed],
           "hyperparameters": {**tcfg, "hidden": mcfg["hidden"],
                               "dropout": mcfg["dropout"]},
           "catalogue_baseline": {"median_z_minus_y": median_colour, **base},
           "metrics_over_seeds": summary, "per_seed": per_seed}
    path = cfg.path("results_dir") / f"{args.tag}_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "-" * 74)
    print(f"{label} TEST METRICS, mean +/- std over seeds")
    print("-" * 74)
    print(f"  parameters        : {out['n_parameters']:,} over {len(cols)} features")
    print(f"  seeds             : {out['seeds']}")
    s = summary
    print(f"  delta mag median  : {s['delta_mag_median']['mean']:+.4f}")
    print(f"  delta mag NMAD    : {s['delta_mag_nmad']['mean']:.4f} +/- "
          f"{s['delta_mag_nmad']['std']:.4f}")
    print(f"  delta mag MAE     : {s['delta_mag_mae']['mean']:.4f}")
    print(f"  outliers |dm|>0.5 : {s['outlier_percent']['mean']:.2f} %")
    print(f"\n  catalogue baseline NMAD {base['delta_mag_nmad']:.4f}  ->  "
          f"M1 {s['delta_mag_nmad']['mean']:.4f}   "
          f"({100 * (1 - s['delta_mag_nmad']['mean'] / base['delta_mag_nmad']):+.1f}%)")
    print("""
  Compare with the image models' sigma(mag) with care: both are magnitude
  errors on the same 17,072 test galaxies, but M1 predicts a catalogue
  magnitude directly while the image models do aperture photometry on a
  predicted image. A few per cent between them is estimator, not skill.""")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
