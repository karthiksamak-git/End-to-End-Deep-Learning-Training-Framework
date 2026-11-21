"""Data pipeline: preprocessing, splits and DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataBundle:
    """Everything the trainer/evaluator needs from a dataset.

    Attributes:
        train_loader: Shuffling loader used for optimization.
        val_loader: Validation loader (monitoring, early stopping).
        test_loader: Held-out test loader (final metrics only).
        input_shape: Per-sample feature shape, e.g. ``(30,)`` or ``(1, 28, 28)``.
        num_classes: Number of target classes.
        class_names: Human-readable class labels, index-aligned.
        task: ``tabular`` or ``image``.
        test_dataset: Underlying dataset, used for error analysis.
        generator: Seeded generator that drives train-loader shuffling, so a
            resumed run can restore the exact sample order.
        meta: Extra JSON-friendly facts (EDA summary, normalisation constants).
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_shape: tuple[int, ...]
    num_classes: int
    class_names: list[str]
    task: str
    test_dataset: Dataset
    generator: torch.Generator
    sizes: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = ["DataBundle"]
