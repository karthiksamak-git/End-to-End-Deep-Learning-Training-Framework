"""Atomic checkpointing with self-describing metadata.

Every checkpoint carries the config, the epoch, the metric history, the RNG state
and provenance (git hash, seed, device, library versions) so a run can be audited
or resumed months later.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT = 1


def git_commit_hash(short: bool = True) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def environment_info() -> dict[str, str]:
    """Versions of the tools that affect numerical results."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_hash": git_commit_hash(),
    }


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` to ``path`` via a temp file + ``os.replace``.

    A crash mid-write can therefore never leave a truncated checkpoint behind.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    meta: dict[str, Any],
    best: dict[str, Any] | None = None,
    rng: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the checkpoint dictionary (plain Python + tensors only)."""
    return {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "history": history,
        "best": best or {},
        "config": config,
        "meta": {**environment_info(), **meta},
        "rng": rng or {},
    }


def save_checkpoint(
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Build and atomically write a checkpoint. See :func:`build_checkpoint`."""
    return atomic_torch_save(build_checkpoint(**kwargs), path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint and validate its format.

    Raises:
        FileNotFoundError: when the path does not exist.
        ValueError: when the file is not a dltf checkpoint.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint not found: {target}")
    payload = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{target} is not a dltf checkpoint")
    if payload.get("format", 0) > CHECKPOINT_FORMAT:
        raise ValueError(
            f"{target} was written by a newer dltf (format {payload['format']}); "
            "upgrade the package"
        )
    return payload


def resolve_checkpoint(run_dir: str | Path, which: str = "best") -> Path:
    """Return ``run_dir/checkpoints/{which}.pt``, falling back to ``last.pt``."""
    checks = Path(run_dir) / "checkpoints"
    best = checks / f"{which}.pt"
    if best.exists():
        return best
    last = checks / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"no checkpoint in {checks}")
