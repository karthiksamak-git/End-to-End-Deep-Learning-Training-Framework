"""Export a trained model to ONNX and prove numerical parity with PyTorch.

The browser playground depends on the ONNX graph having a dynamic batch axis and
raw logits as output (softmax happens in JavaScript), and on the exported graph
reproducing PyTorch within ``1e-4``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

logger = logging.getLogger("dltf.export")

MAX_MODEL_BYTES = 10 * 1024 * 1024
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def export_onnx(
    model: nn.Module,
    *,
    input_shape: tuple[int, ...],
    output_path: str | Path,
    opset: int = 17,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Export ``model`` to ``output_path`` with a dynamic batch axis.

    Returns:
        Metadata about the exported artifact (path, size, opset, IO names).
    """
    if opset < 17:
        raise ValueError("opset must be >= 17")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to(device)
    dummy = torch.randn(1, *input_shape, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_bytes = path.stat().st_size
    info = {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024**2, 3),
        "under_10mb": size_bytes < MAX_MODEL_BYTES,
        "opset": opset,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_shape": list(input_shape),
        "dynamic_batch": True,
    }
    if not info["under_10mb"]:
        logger.warning("exported model is %.2f MB (budget is 10 MB)", info["size_mb"])
    return info


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: str | Path,
    *,
    input_shape: tuple[int, ...],
    sample_input: torch.Tensor | None = None,
    atol: float = 1e-4,
    n_random: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare PyTorch logits with onnxruntime logits.

    Uses the supplied ``sample_input`` (real preprocessed images) plus a few seeded
    random batches so the parity claim covers more than one input.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    model = model.eval()
    generator = torch.Generator().manual_seed(seed)
    batches = [torch.randn(n_random, *input_shape, generator=generator)]
    if sample_input is not None:
        batches.append(sample_input.detach().cpu().float())

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs = 0.0
    checked = 0
    sample_logits: list[float] = []
    for batch in batches:
        batch = batch.cpu().float()
        with torch.no_grad():
            reference = model(batch).cpu().numpy()
        result = session.run(None, {INPUT_NAME: batch.numpy()})[0]
        if result.shape != reference.shape:
            raise ValueError(
                f"ONNX output shape {result.shape} != PyTorch output shape {reference.shape}"
            )
        max_abs = max(max_abs, float(np.abs(result - reference).max()))
        checked += int(batch.shape[0])
        if not sample_logits:
            sample_logits = [float(v) for v in result[0][:5]]
    return {
        "max_abs_diff": max_abs,
        "atol": atol,
        "passed": max_abs < atol,
        "inputs_checked": checked,
        "sample_logits": sample_logits,
        "providers": session.get_providers(),
    }


def write_export_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the export + parity report as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
