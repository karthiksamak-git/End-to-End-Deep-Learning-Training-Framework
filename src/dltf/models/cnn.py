"""A compact residual CNN for 28x28 grayscale images.

Three stages (28 -> 14 -> 7 pixels) of pre-activation residual blocks, global
average pooling instead of a huge fully-connected head, and a dropout classifier.
Small enough to export comfortably under 10 MB and to train on CPU in minutes.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity shortcut and pre-activation BN."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual transform."""
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class ResNetSmall(nn.Module):
    """Residual CNN with a global-average-pooling classifier.

    Args:
        in_channels: Input image channels.
        num_classes: Number of output logits.
        width: Base channel count; stages use ``width``, ``2*width``, ``4*width``.
        depth: Residual blocks per stage.
        dropout: Spatial dropout inside blocks and before the classifier.

    Shape:
        - Input: ``(N, in_channels, 28, 28)``.
        - Output: ``(N, num_classes)`` logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        width: int = 32,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 4 or depth < 1 or num_classes < 2:
            raise ValueError("width >= 4, depth >= 1 and num_classes >= 2 are required")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.width = int(width)
        self.depth = int(depth)

        channels = [width, width * 2, width * 4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        prev = width
        for i, ch in enumerate(channels):
            if i > 0:
                stages.append(
                    nn.Sequential(
                        nn.Conv2d(prev, ch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                )
                prev = ch
            stages.append(nn.Sequential(*[ResidualBlock(ch, dropout) for _ in range(depth)]))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        return self.head(self.stages(self.stem(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled embedding (useful for analysis)."""
        return self.head[:2](self.stages(self.stem(x)))
