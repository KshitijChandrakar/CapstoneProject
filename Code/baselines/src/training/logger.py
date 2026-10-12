"""Experiment logging to the console and a CSV file.

Deliberately dependency-free. A WandB backend can be added later by giving it
the same `log_epoch` / `close` interface.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

#: Make +/- and arrows printable on a cp1252 Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ExperimentLogger:
    """Console + CSV logger for one training run."""

    def __init__(
        self,
        run_name: str,
        out_dir: Path,
        fields: tuple[str, ...] = ("epoch", "train_loss", "val_loss", "lr", "seconds"),
        verbose: bool = True,
        log_every: int = 5,
    ) -> None:
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / f"{run_name}_history.csv"
        self.fields = fields
        self.verbose = verbose
        self.log_every = log_every
        self.history: list[dict[str, Any]] = []
        self._start = time.time()

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)

    def log_epoch(self, row: dict[str, Any], is_best: bool = False) -> None:
        """Record one epoch and append it to the CSV immediately."""
        row = {**row, "seconds": round(time.time() - self._start, 2)}
        self.history.append(row)

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])

        if not self.verbose:
            return
        epoch = row.get("epoch", 0)
        total = row.get("total_epochs", "?")
        if is_best or epoch == 1 or epoch == total or epoch % self.log_every == 0:
            marker = "  <- best" if is_best else ""
            print(f"  epoch {epoch:>3}/{total}  "
                  f"train L1 {row.get('train_loss', float('nan')):.5f}  "
                  f"val L1 {row.get('val_loss', float('nan')):.5f}{marker}")

    def log_dict(self, name: str, payload: dict) -> Path:
        """Write a JSON artefact next to the history."""
        path = self.out_dir / f"{self.run_name}_{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def info(self, message: str) -> None:
        if self.verbose:
            print(f"  {message}")

    def close(self) -> dict:
        elapsed = time.time() - self._start
        if self.verbose:
            print(f"  run '{self.run_name}' finished in {elapsed:.1f} s "
                  f"-> {self.csv_path.name}")
        return {"run_name": self.run_name, "seconds": elapsed,
                "history_csv": str(self.csv_path)}
