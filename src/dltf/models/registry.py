"""Name -> model factory registry, shared by every entry point."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dltf.config import ModelConfig
from dltf.models.cnn import ResNetSmall
from dltf.models.mlp import MLP

MODEL_NAMES = ("mlp", "cnn")


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count model parameters (trainable by default)."""
    params = model.parameters()
    return int(sum(p.numel() for p in params if p.requires_grad or not trainable_only))


def build_model(
    cfg: ModelConfig,
    *,
    input_shape: tuple[int, ...],
    num_classes: int,
) -> nn.Module:
    """Instantiate a model from config.

    Args:
        cfg: Model settings (``name``, widths, dropout, ...).
        input_shape: Per-sample shape, ``(features,)`` or ``(C, H, W)``.
        num_classes: Output dimensionality.

    Raises:
        ValueError: for unknown model names or shapes incompatible with the model.
    """
    name = cfg.name.lower()
    if name == "mlp":
        in_features = 1
        for dim in input_shape:
            in_features *= dim
        return MLP(
            in_features=in_features,
            hidden_sizes=cfg.hidden_sizes,
            num_classes=num_classes,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
            activation=cfg.activation,
        )
    if name == "cnn":
        if len(input_shape) != 3:
            raise ValueError(
                f"cnn expects a (C, H, W) input shape, got {input_shape}; "
                "use the image task config"
            )
        return ResNetSmall(
            in_channels=input_shape[0],
            num_classes=num_classes,
            width=cfg.width,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown model {cfg.name!r}; available: {MODEL_NAMES}")


def model_summary(model: nn.Module) -> dict[str, Any]:
    """JSON-friendly architecture summary for logs and the model card."""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model)
    return {
        "class": type(model).__name__,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "size_mb_fp32": round(total * 4 / (1024**2), 3),
    }


def assert_gradients_are_finite(model: nn.Module, input_shape: tuple[int, ...] | None = None) -> None:
    """Raise when any parameter gradient is NaN/Inf after a backward pass."""
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")
