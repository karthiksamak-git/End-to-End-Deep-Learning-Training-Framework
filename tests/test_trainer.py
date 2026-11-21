"""Trainer: overfit sanity check, checkpoint round-trip and resume parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dltf.device import get_device
from dltf.models.mlp import MLP
from dltf.seed import set_seed
from dltf.training.checkpoint import load_checkpoint, resolve_checkpoint, save_checkpoint
from dltf.training.trainer import Trainer
from tests.conftest import fast_config, synthetic_tabular


def _trainer(seed: int, run_dir: Path, epochs: int, *, dropout: float = 0.0) -> Trainer:
    set_seed(seed)
    train_loader, val_loader, generator = synthetic_tabular(seed=seed)
    cfg = fast_config(epochs=epochs)
    cfg.model.dropout = dropout
    model = MLP(in_features=8, hidden_sizes=[32], num_classes=3, dropout=dropout, batchnorm=False)
    return Trainer(
        model,
        config=cfg,
        device_info=get_device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=3,
        run_dir=run_dir,
        generator=generator,
        progress=False,
    )


def test_trainer_overfits_tiny_batch(tmp_path: Path) -> None:
    """A 32-sample separable set must be memorised to 100% accuracy."""
    set_seed(0)
    train_loader, val_loader, generator = synthetic_tabular(n=32, classes=3, batch_size=8, seed=0)
    cfg = fast_config(epochs=80)
    cfg.optim.lr = 0.05
    cfg.model.dropout = 0.0
    model = MLP(in_features=8, hidden_sizes=[32], num_classes=3, dropout=0.0, batchnorm=False)
    trainer = Trainer(
        model,
        config=cfg,
        device_info=get_device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=3,
        run_dir=tmp_path / "overfit",
        generator=generator,
        progress=False,
    )
    summary = trainer.fit()
    metrics = trainer.evaluate(val_loader)
    assert len(summary["history"]) == 80
    assert metrics["accuracy"] >= 0.99, metrics
    # Outputs must be written for the dashboard/CLI to consume.
    assert (tmp_path / "overfit" / "metrics.json").exists()
    assert (tmp_path / "overfit" / "metrics.csv").exists()
    best = load_checkpoint(tmp_path / "overfit" / "checkpoints" / "best.pt")
    assert best["epoch"] >= 1
    assert best["meta"]["seed"] == 7
    assert best["history"]


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    set_seed(3)
    model = MLP(in_features=8, hidden_sizes=[16], num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=4,
        metrics={"val_loss": 0.5},
        history=[{"epoch": 4, "val_loss": 0.5}],
        config={"name": "roundtrip"},
        meta={"seed": 3},
        best={"epoch": 4, "score": 0.5},
        rng={},
    )
    payload = load_checkpoint(path)
    assert payload["epoch"] == 4
    assert payload["config"]["name"] == "roundtrip"
    assert payload["metrics"]["val_loss"] == pytest.approx(0.5)
    assert resolve_checkpoint(tmp_path, "last") == path
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file"


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    """Continuing from a checkpoint must reproduce the uninterrupted metrics."""
    total_epochs = 4
    reference_dir = tmp_path / "reference"
    interrupted_dir = tmp_path / "interrupted"

    reference = _trainer(1, reference_dir, total_epochs, dropout=0.3)
    reference_summary = reference.fit()
    assert len(reference_summary["history"]) == total_epochs

    first = _trainer(1, interrupted_dir, total_epochs, dropout=0.3)
    first.fit(epochs=2)
    checkpoint = interrupted_dir / "checkpoints" / "last.pt"

    resumed = _trainer(1, interrupted_dir, total_epochs, dropout=0.3)
    resumed_summary = resumed.fit(epochs=total_epochs, resume=checkpoint)

    assert resumed_summary["epochs_run"] == total_epochs
    for reference_row, resumed_row in zip(
        reference_summary["history"][2:], resumed_summary["history"][2:], strict=True
    ):
        assert resumed_row["val_loss"] == pytest.approx(reference_row["val_loss"], abs=1e-6)
        assert resumed_row["val_f1"] == pytest.approx(reference_row["val_f1"], abs=1e-6)
    json.loads((interrupted_dir / "metrics.json").read_text())
