"""ONNX export, dynamic batch axis and numerical parity with PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from dltf.export.onnx_export import export_onnx, verify_onnx_parity
from dltf.models.cnn import ResNetSmall
from dltf.models.mlp import MLP


def _cnn() -> ResNetSmall:
    torch.manual_seed(0)
    return ResNetSmall(in_channels=1, num_classes=10, width=8, depth=1, dropout=0.0)


def test_onnx_export_and_parity(tmp_path: Path) -> None:
    model = _cnn().eval()
    path = tmp_path / "model.onnx"
    info = export_onnx(model, input_shape=(1, 28, 28), output_path=path, opset=17)
    assert path.exists()
    assert info["opset"] == 17
    assert info["under_10mb"], info["size_mb"]
    assert info["input_name"] == "input"

    report = verify_onnx_parity(model, path, input_shape=(1, 28, 28), atol=1e-4)
    assert report["passed"], report
    assert report["max_abs_diff"] < 1e-4


def test_onnx_dynamic_batch_axis(tmp_path: Path) -> None:
    model = _cnn().eval()
    path = export_onnx(model, input_shape=(1, 28, 28), output_path=tmp_path / "m.onnx")["path"]
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    batch = np.random.default_rng(0).normal(size=(3, 1, 28, 28)).astype(np.float32)
    out = session.run(None, {"input": batch})[0]
    assert out.shape == (3, 10)


def test_parity_with_real_preprocessed_batch(tmp_path: Path) -> None:
    model = _cnn().eval()
    path = export_onnx(model, input_shape=(1, 28, 28), output_path=tmp_path / "m.onnx")["path"]
    sample = torch.randn(5, 1, 28, 28)
    report = verify_onnx_parity(model, path, input_shape=(1, 28, 28), sample_input=sample)
    assert report["passed"]
    assert report["inputs_checked"] == 9  # 4 random + 5 real


def test_mlp_export(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = MLP(in_features=30, hidden_sizes=[8], num_classes=2, dropout=0.0, batchnorm=False)
    info = export_onnx(model, input_shape=(30,), output_path=tmp_path / "mlp.onnx")
    report = verify_onnx_parity(model, info["path"], input_shape=(30,))
    assert report["passed"], report


def test_missing_onnx_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_onnx_parity(_cnn(), tmp_path / "nope.onnx", input_shape=(1, 28, 28))


def test_low_opset_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="opset"):
        export_onnx(_cnn(), input_shape=(1, 28, 28), output_path=tmp_path / "x.onnx", opset=11)
