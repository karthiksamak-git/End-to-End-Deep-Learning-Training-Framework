"""The training engine: train/validation loops, AMP, clipping, scheduling, resume."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dltf.config import Config, OptimConfig
from dltf.device import DeviceInfo
from dltf.models.registry import model_summary
from dltf.seed import rng_state, set_rng_state
from dltf.training.callbacks import Callback, CSVLogger, EarlyStopping
from dltf.training.checkpoint import load_checkpoint, save_checkpoint
from dltf.training.metrics import compute_classification_metrics

logger = logging.getLogger("dltf.trainer")


def monitor_mode(metric: str) -> str:
    """``min`` for loss-like metrics, ``max`` otherwise."""
    return "min" if "loss" in metric else "max"


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Create the configured optimizer."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.momentum > 0,
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    *,
    epochs: int,
    steps_per_epoch: int,
) -> tuple[Any | None, str]:
    """Return ``(scheduler, interval)`` where interval is ``epoch`` or ``step``."""
    if cfg.scheduler == "none":
        return None, "epoch"
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs)), "epoch"
    if cfg.scheduler == "onecycle":
        total_steps = max(1, epochs * max(1, steps_per_epoch))
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.3
            ),
            "step",
        )
    if cfg.scheduler == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg.gamma, patience=max(1, cfg.step_size // 5)
            ),
            "epoch",
        )
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


class Trainer:
    """Trains a classifier with per-epoch validation, checkpointing and resume.

    Args:
        model: Any ``nn.Module`` returning logits.
        config: Full experiment config (optimizer/train/paths are read).
        device_info: Resolved device from :func:`dltf.device.get_device`.
        train_loader: Training loader.
        val_loader: Validation loader used for monitoring.
        num_classes: Output dimensionality.
        class_names: Optional human-readable labels for metadata.
        run_dir: Output directory; created if missing.
        callbacks: Extra callbacks (early stopping and CSV logging are added).
        generator: The seeded generator driving train-loader shuffling, so resume
            can restore the exact sample order.
        progress: Show a tqdm bar (off in tests/CI).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        config: Config,
        device_info: DeviceInfo,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        class_names: list[str] | None = None,
        run_dir: str | Path | None = None,
        callbacks: list[Callback] | None = None,
        generator: torch.Generator | None = None,
        progress: bool = True,
        save_checkpoints: bool = True,
    ) -> None:
        self.config = config
        self.device_info = device_info
        self.device = device_info.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = int(num_classes)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.generator = generator
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(self.model, config.optim)
        self.lr_scheduler, self.scheduler_interval = build_scheduler(
            self.optimizer,
            config.optim,
            epochs=config.train.epochs,
            steps_per_epoch=len(train_loader),
        )
        self.use_amp = bool(config.train.amp and device_info.supports_amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else None
        )
        self.epoch = 0
        self.history: list[dict[str, Any]] = []
        self.monitor = config.train.monitor
        self.mode = monitor_mode(self.monitor)
        self.best: dict[str, Any] = {"epoch": 0, "score": None, "metric": self.monitor}
        self.progress = progress
        self.save_checkpoints = save_checkpoints
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else Path(config.paths.runs_dir)
            / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{config.name}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.total_seconds = 0.0
        self._first_epoch_time: float | None = None

        self.callbacks: list[Callback] = list(callbacks or [])
        if config.train.early_stopping_patience > 0 and not any(
            isinstance(cb, EarlyStopping) for cb in self.callbacks
        ):
            self.callbacks.append(
                EarlyStopping(
                    monitor=self.monitor,
                    patience=config.train.early_stopping_patience,
                    min_delta=config.train.early_stopping_min_delta,
                    mode=self.mode,
                )
            )
        self.callbacks.append(CSVLogger(self.run_dir / "metrics.csv"))

        logger.info(
            "trainer ready | model=%s params=%d device=%s amp=%s epochs=%d",
            type(model).__name__,
            model_summary(model)["parameters_total"],
            self.device,
            self.use_amp,
            config.train.epochs,
        )

    # ------------------------------------------------------------------ helpers
    @property
    def current_lr(self) -> float:
        """Learning rate currently used by the optimizer."""
        return float(self.optimizer.param_groups[0]["lr"])

    def _autocast(self) -> Any:
        if self.use_amp:
            return torch.autocast(device_type=self.device.type, dtype=self.device_info.amp_dtype)
        return contextlib.nullcontext()

    def _rng_payload(self) -> dict[str, Any]:
        state = rng_state()
        if self.generator is not None:
            state["generator"] = self.generator.get_state()
        return state

    def _checkpoint_meta(self) -> dict[str, Any]:
        return {
            "run_name": self.config.name,
            "seed": self.config.seed,
            "device": self.device_info.as_dict(),
            "class_names": self.class_names,
            "model": model_summary(self.model),
            "train_size": len(self.train_loader.dataset),
            "val_size": len(self.val_loader.dataset),
            "use_amp": self.use_amp,
        }

    def _save(self, filename: str) -> Path | None:
        if not self.save_checkpoints:
            return None
        return save_checkpoint(
            self.checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            scaler=self.scaler,
            epoch=self.epoch,
            metrics=self.history[-1] if self.history else {},
            history=self.history,
            config=self.config.to_dict(),
            meta=self._checkpoint_meta(),
            best=self.best,
            rng=self._rng_payload(),
        )

    def _is_improvement(self, value: float) -> bool:
        score = self.best.get("score")
        if score is None or not math.isfinite(float(score)):
            return math.isfinite(value)
        return value < score if self.mode == "min" else value > score

    def resume_from(self, path: str | Path) -> None:
        """Restore model/optimizer/scheduler/scaler/epoch/RNG state from a checkpoint.

        After this the next ``fit`` call continues exactly where the saved run
        stopped, producing the same metrics an uninterrupted run would.
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        rng = ckpt.get("rng") or {}
        set_rng_state({k: v for k, v in rng.items() if k != "generator"})
        if self.generator is not None and "generator" in rng:
            self.generator.set_state(rng["generator"])
        self.history = list(ckpt.get("history", []))
        self.best = dict(ckpt.get("best") or self.best)
        self.epoch = int(ckpt.get("epoch", 0))
        logger.info("resumed from %s at epoch %d", path, self.epoch)

    # ------------------------------------------------------------------- loops
    def _train_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        seen = 0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for xb, yb in self.train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.optim.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optim.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.optim.grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optim.grad_clip
                    )
                self.optimizer.step()
            if self.scheduler_interval == "step" and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            batch_size = xb.size(0)
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size
            all_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

        from sklearn.metrics import accuracy_score, f1_score

        y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
        average = "binary" if self.num_classes == 2 else "macro"
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
            "f1_macro": float(f1_score(y_true, y_pred, average=average, zero_division=0))
            if y_true.size
            else 0.0,
        }
        return total_loss / max(1, seen), metrics

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(probabilities, targets)`` for a loader."""
        self.model.eval()
        probs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(xb)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(yb.numpy())
        if not probs:
            return np.empty((0, self.num_classes), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.concatenate(probs).astype(np.float32), np.concatenate(targets).astype(np.int64)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Full metric suite (including loss) for a loader."""
        probs, targets = self.predict(loader)
        metrics = compute_classification_metrics(targets, probs, class_names=self.class_names)
        loss = self._loss_from_probs(probs, targets)
        return {"loss": loss, **metrics}

    def _loss_from_probs(self, probs: np.ndarray, targets: np.ndarray) -> float:
        if targets.size == 0:
            return float("nan")
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        picked = log_probs[np.arange(targets.size), targets]
        return float(-picked.mean())

    def fit(
        self,
        epochs: int | None = None,
        *,
        resume: str | Path | None = None,
        write_outputs: bool = True,
    ) -> dict[str, Any]:
        """Run training and return the run summary.

        Args:
            epochs: Total epochs (defaults to ``config.train.epochs``).
            resume: Optional checkpoint to continue from.
            write_outputs: Write ``metrics.json`` when done.
        """
        total_epochs = int(epochs if epochs is not None else self.config.train.epochs)
        if total_epochs < 1:
            raise ValueError("epochs must be >= 1")
        if resume is not None:
            self.resume_from(resume)

        for cb in self.callbacks:
            cb.on_train_start(self)
        started = time.time()
        stopped = False
        remaining = range(self.epoch + 1, total_epochs + 1)
        bar = tqdm(remaining, desc=self.config.name, unit="epoch", disable=not self.progress)
        for epoch in bar:
            self.epoch = epoch
            t0 = time.time()
            train_loss, train_metrics = self._train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics.pop("loss")
            elapsed = time.time() - t0
            if self._first_epoch_time is None:
                self._first_epoch_time = elapsed

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "train_f1": round(train_metrics["f1_macro"], 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_f1": round(val_metrics["f1_macro"], 6),
                "val_roc_auc": round(val_metrics["roc_auc"], 6)
                if math.isfinite(val_metrics["roc_auc"])
                else None,
                "lr": self.current_lr,
                "seconds": round(elapsed, 3),
            }
            self.history.append(record)

            # Step the scheduler *before* checkpointing so a resumed run restores
            # the exact LR state the uninterrupted run would have at this point.
            if self.scheduler_interval == "epoch" and self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            score = record.get(self.monitor)
            if score is not None and self._is_improvement(float(score)):
                self.best = {"epoch": epoch, "score": float(score), "metric": self.monitor}
                self._save("best.pt")
            self._save("last.pt")

            if not self.progress:
                logger.info(
                    "epoch %d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                    epoch,
                    record["train_loss"],
                    record["val_loss"],
                    record["val_accuracy"],
                    record["val_f1"],
                )
            bar.set_postfix(
                train=f"{record['train_loss']:.3f}",
                val=f"{record['val_loss']:.3f}",
                acc=f"{record['val_accuracy']:.3f}",
            )

            for cb in self.callbacks:
                if cb.on_epoch_end(self, epoch, record) is True:
                    stopped = True
            if stopped:
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        self.total_seconds = time.time() - started
        summary = {
            "run_dir": str(self.run_dir),
            "run_name": self.config.name,
            "epochs_run": len(self.history),
            "requested_epochs": total_epochs,
            "early_stopped": stopped,
            "duration_sec": round(self.total_seconds, 2),
            "seconds_per_epoch": round(
                self.total_seconds / max(1, len(self.history)), 3
            ),
            "best": self.best,
            "best_record": self.history[self.best["epoch"] - 1] if self.best["epoch"] else None,
            "final": self.history[-1] if self.history else None,
            "device": self.device_info.as_dict(),
            "model": model_summary(self.model),
            "use_amp": self.use_amp,
            "history": self.history,
        }
        if write_outputs:
            (self.run_dir / "metrics.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            logger.info(
                "finished %d epochs in %.1fs | best %s=%.4f at epoch %d",
                len(self.history),
                self.total_seconds,
                self.monitor,
                float(self.best.get("score") or float("nan")),
                int(self.best.get("epoch") or 0),
            )
        return summary
