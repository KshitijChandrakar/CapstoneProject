"""Training loop for B1, with validation, checkpointing and early stopping."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from src.models.b1_linear import masked_l1_loss
from src.training.logger import ExperimentLogger


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
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.logger = logger
        self.patience = early_stopping_patience

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
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

                loss = masked_l1_loss(self.model(x), y, m)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                total += float(loss.detach())
                n_batches += 1
        return total / max(n_batches, 1)

    # -- full run ----------------------------------------------------------

    def fit(self) -> dict:
        """Train for `epochs`, keeping the best-validation weights."""
        since_improved = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

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
