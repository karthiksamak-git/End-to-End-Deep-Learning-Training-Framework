"""Shared fixtures and fast factories for the test-suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dltf.config import Config, DataConfig, ModelConfig, OptimConfig, TrainConfig
from dltf.device import DeviceInfo, get_device
from dltf.seed import make_generator


def fast_config(*, task: str = "tabular", epochs: int = 3, **overrides) -> Config:
    """A tiny config: one hidden layer, no early stopping, deterministic."""
    cfg = Config(
        name="pytest",
        task=task,
        seed=7,
        data=DataConfig(batch_size=8, eval_batch_size=16, val_fraction=0.2, test_fraction=0.0),
        model=ModelConfig(
            name="mlp" if task == "tabular" else "cnn",
            hidden_sizes=[16],
            dropout=0.2,
            batchnorm=True,
            width=8,
            depth=1,
        ),
        optim=OptimConfig(optimizer="adamw", lr=0.05, scheduler="cosine", grad_clip=1.0),
        train=TrainConfig(
            epochs=epochs,
            amp=False,
            deterministic=True,
            early_stopping_patience=0,
            monitor="val_loss",
        ),
    )
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if section:
            setattr(getattr(cfg, section), field, value)
        else:
            setattr(cfg, key, value)
    return cfg


def synthetic_tabular(
    *, n: int = 32, features: int = 8, classes: int = 3, seed: int = 0, batch_size: int = 8
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    """Linearly separable synthetic classification data on CPU."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0.0, 3.0, size=(classes, features))
    y = np.arange(n) % classes
    X = centroids[y] + rng.normal(0.0, 0.25, size=(n, features))
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    generator = make_generator(seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader, generator


@pytest.fixture(scope="session")
def device_info() -> DeviceInfo:
    return get_device("cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")
