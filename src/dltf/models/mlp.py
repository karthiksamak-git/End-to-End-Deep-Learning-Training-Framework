"""A small, fully configurable multilayer perceptron."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dltf.config import ACTIVATIONS


def resolve_activation(name: str) -> nn.Module:
    """Map an activation name to a fresh module instance."""
    builders: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in builders:
        raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")
    return builders[name]()


class MLP(nn.Module):
    """Multilayer perceptron with optional BatchNorm and Dropout.

    Args:
        in_features: Flattened input size.
        hidden_sizes: Width of each hidden layer; empty means a linear classifier.
        num_classes: Output logits.
        dropout: Dropout probability applied after each hidden activation.
        batchnorm: Insert BatchNorm1d before each activation.
        activation: One of :data:`dltf.config.ACTIVATIONS`.

    Shape:
        - Input: ``(N, in_features)`` or ``(N, ...)`` — extra dims are flattened.
        - Output: ``(N, num_classes)`` logits (no softmax).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (64, 32),
        num_classes: int = 2,
        dropout: float = 0.2,
        batchnorm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_features < 1 or num_classes < 2:
            raise ValueError("in_features must be >= 1 and num_classes >= 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self._activation_name = activation
        self._dropout = float(dropout)
        self._batchnorm = bool(batchnorm)

        layers: list[nn.Module] = []
        prev = self.in_features
        for width in self.hidden_sizes:
            if width < 1:
                raise ValueError("hidden sizes must be >= 1")
            layers.append(nn.Linear(prev, width))
            if self._batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(resolve_activation(activation))
            if self._dropout > 0:
                layers.append(nn.Dropout(self._dropout))
            prev = width
        layers.append(nn.Linear(prev, self.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch."""
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
