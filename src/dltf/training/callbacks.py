"""Training callbacks: early stopping, LR monitoring, CSV logging, Optuna pruning."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from dltf.training.trainer import Trainer


class Callback:
    """Base callback. Return ``True`` from :meth:`on_epoch_end` to stop training."""

    def on_train_start(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: dict[str, Any]) -> bool | None:
        """Called after each epoch with that epoch's metrics."""
        return None

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after training finishes or stops early."""


class EarlyStopping(Callback):
    """Stop when ``monitor`` stops improving for ``patience`` epochs."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if patience < 0:
            raise ValueError("patience must be >= 0")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_epochs = 0
        self.stopped_epoch: int | None = None

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: dict[str, Any]) -> bool | None:
        """Track the monitored metric and signal a stop when patience runs out."""
        if self.monitor not in record:
            raise KeyError(f"early stopping monitor {self.monitor!r} missing from metrics")
        value = float(record[self.monitor])
        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.patience and self.bad_epochs >= self.patience:
                self.stopped_epoch = epoch
                trainer.log.info(
                    "early stopping at epoch %d (%s did not improve from %.5f)",
                    epoch,
                    self.monitor,
                    self.best,
                )
                return True
        return None


class LRMonitor(Callback):
    """Record the learning rate seen at the end of every epoch."""

    def __init__(self) -> None:
        self.learning_rates: list[float] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: dict[str, Any]) -> None:
        """Append the current LR."""
        self.learning_rates.append(float(record.get("lr", float("nan"))))


class CSVLogger(Callback):
    """Append one row per epoch to ``<run_dir>/metrics.csv``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._columns: list[str] | None = None

    def on_train_start(self, trainer: Trainer) -> None:
        """Create the file and its header if this is a fresh run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: dict[str, Any]) -> None:
        """Write the epoch record, keeping column order stable."""
        if self._columns is None:
            existing = self.path.read_text(encoding="utf-8").strip().splitlines()
            self._columns = existing[0].split(",") if existing else list(record)
        write_header = self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: record.get(k) for k in self._columns})
