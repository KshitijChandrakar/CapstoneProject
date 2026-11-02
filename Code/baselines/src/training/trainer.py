"""Training loop with validation, checkpointing and early stopping.

Shared by B1, B2 and B3. The defaults are B1's original behaviour -- masked
L1 on a model called as ``model(x)`` -- so adding B2 and B3 did not change how
B1 trains. B3 supplies ``loss_fn`` (which also receives the epoch, for the NLL
warmup) and ``conditioning_key`` (redshift, for FiLM).
"""

from __future__ import annotations

import copy
import inspect
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.b1_linear import masked_l1_loss
from src.training.logger import ExperimentLogger


def _default_loss(pred, target, mask):
    """Masked L1 -- B1's and B2's objective."""
    return masked_l1_loss(pred, target, mask)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """Trains a model with masked L1, AdamW and a cosine schedule.

    Args:
        model: the module to train.
        train_loader / val_loader: data.
        lr, weight_decay, epochs: optimisation settings.
        device: where to run.
        logger: receives per-epoch rows.
        early_stopping_patience: stop after this many epochs without a val
            improvement. ``None`` disables it.
        loss_fn: ``f(model_out, target, mask)``, or ``f(..., epoch)`` for an
            objective that changes during the run. Defaults to masked L1,
            which is what B1 and B2 use.
        conditioning_key: batch key passed to the model as a second argument,
            e.g. ``"redshift"`` for B3's FiLM. ``None`` calls ``model(x)``.
        grad_clip_norm: clip the global gradient norm to this value. ``None``
            disables it. B2 and B3 use it: with 2M parameters and a variance
            head, one bad batch can otherwise push NaN through every weight.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        device: str = "cpu",
        logger: ExperimentLogger | None = None,
        early_stopping_patience: int | None = None,
        loss_fn: Callable | None = None,
        conditioning_key: str | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.logger = logger
        self.patience = early_stopping_patience
        self.loss_fn = loss_fn if loss_fn is not None else _default_loss
        self.conditioning_key = conditioning_key
        self.grad_clip_norm = grad_clip_norm
        # Only pass the epoch to a loss that asks for it, so a plain
        # f(pred, target, mask) stays valid.
        self._loss_takes_epoch = "epoch" in inspect.signature(self.loss_fn).parameters
        self.epoch = 0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # A switching loss (B3) exposes warmup_epochs; this is the first
        # epoch on the new objective. None means it never changes.
        warmup = getattr(self.loss_fn, "warmup_epochs", None)
        self.objective_changes_at_epoch = (
            int(warmup) + 1 if warmup not in (None, 0) else None
        )

        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.history: list[dict] = []
        self.stopped_early = False

    # -- one pass ----------------------------------------------------------

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n_batches = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                x = batch["input"].to(self.device, non_blocking=True)
                y = batch["target"].to(self.device, non_blocking=True)
                m = batch["mask"].to(self.device, non_blocking=True)

                if self.conditioning_key is None:
                    out = self.model(x)
                else:
                    cond = batch[self.conditioning_key].to(
                        self.device, non_blocking=True
                    )
                    out = self.model(x, cond)

                loss = (
                    self.loss_fn(out, y, m, epoch=self.epoch)
                    if self._loss_takes_epoch else self.loss_fn(out, y, m)
                )
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()
                total += float(loss.detach())
                n_batches += 1
        return total / max(n_batches, 1)

    # -- full run ----------------------------------------------------------

    def fit(self) -> dict:
        """Train for `epochs`, keeping the best-validation weights."""
        since_improved = 0

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # An objective that changes mid-run (B3: L1 warmup, then NLL)
            # produces two incomparable scales. Comparing across the boundary
            # would freeze the best checkpoint inside the warmup, so the
            # record is reset on the epoch the objective changes.
            if epoch == self.objective_changes_at_epoch:
                self.best_val = float("inf")
                since_improved = 0
                if self.logger is not None:
                    self.logger.info(
                        f"objective changed at epoch {epoch}; best-val record "
                        "reset (the two losses are not comparable)"
                    )

            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                since_improved = 0
            else:
                since_improved += 1

            row = {
                "epoch": epoch, "total_epochs": self.epochs,
                "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
            }
            self.history.append(row)
            if self.logger is not None:
                self.logger.log_epoch(row, is_best=is_best)

            if self.patience is not None and since_improved >= self.patience:
                self.stopped_early = True
                if self.logger is not None:
                    self.logger.info(
                        f"early stop at epoch {epoch}: no val improvement for "
                        f"{self.patience} epochs"
                    )
                break

        # Always finish on the best weights, not the last ones.
        self.model.load_state_dict(self.best_state)
        return {
            "best_val_loss": self.best_val,
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.history),
            "stopped_early": self.stopped_early,
            "objective_changed_at_epoch": self.objective_changes_at_epoch,
            "history": self.history,
        }

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> Path:
        """Write the best weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.best_state,
                "best_val_loss": self.best_val,
                "best_epoch": self.best_epoch,
                **(extra or {}),
            },
            path,
        )
        return path
