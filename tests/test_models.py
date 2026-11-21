"""Architecture shapes, gradients and the model registry."""

from __future__ import annotations

import pytest
import torch

from dltf.config import ModelConfig
from dltf.models import MLP, ResNetSmall, build_model, count_parameters, model_summary


def test_mlp_forward_shape() -> None:
    model = MLP(in_features=30, hidden_sizes=[64, 32], num_classes=2, dropout=0.1)
    out = model(torch.randn(5, 30))
    assert out.shape == (5, 2)
    assert count_parameters(model) > 0


def test_mlp_accepts_higher_rank_input() -> None:
    model = MLP(in_features=28, hidden_sizes=[8], num_classes=10)
    assert model(torch.randn(3, 1, 28)).shape == (3, 10)


def test_cnn_forward_shape() -> None:
    model = ResNetSmall(in_channels=1, num_classes=10, width=16, depth=2)
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    assert model.features(torch.randn(2, 1, 28, 28)).shape == (2, 64)


def test_backward_gradients_are_finite() -> None:
    model = ResNetSmall(in_channels=1, num_classes=10, width=8, depth=1)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 1, 28, 28)), torch.tensor([1, 3]))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_registry_builds_both_models() -> None:
    mlp = build_model(ModelConfig(name="mlp", hidden_sizes=[8]), input_shape=(30,), num_classes=2)
    cnn = build_model(ModelConfig(name="cnn", width=8, depth=1), input_shape=(1, 28, 28), num_classes=10)
    assert isinstance(mlp, MLP)
    assert isinstance(cnn, ResNetSmall)
    summary = model_summary(cnn)
    assert summary["parameters_total"] == summary["parameters_trainable"]
    assert summary["class"] == "ResNetSmall"


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelConfig(name="transformer"), input_shape=(10,), num_classes=2)


def test_cnn_requires_image_shape() -> None:
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        build_model(ModelConfig(name="cnn"), input_shape=(30,), num_classes=2)


def test_mlp_validation() -> None:
    with pytest.raises(ValueError):
        MLP(in_features=4, hidden_sizes=[4], num_classes=1)
    with pytest.raises(ValueError):
        MLP(in_features=4, hidden_sizes=[4], num_classes=2, activation="swish")
