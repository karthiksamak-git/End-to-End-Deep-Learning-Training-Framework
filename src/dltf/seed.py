"""Global reproducibility helpers.

Seeding covers Python ``random``, NumPy, PyTorch (CPU + all CUDA devices), the
DataLoader worker RNG and, optionally, deterministic kernels.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG used by the framework and return ``seed``.

    Args:
        seed: Non-negative integer seed.
        deterministic: When True, request deterministic kernels.  This can make
            some ops unsupported on a given device, so callers running exotic
            layers may set it to False.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - very old torch
            torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that gives each DataLoader worker its own seed.

    Derived from the *trainer* generator so the stream is reproducible run to run.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded :class:`torch.Generator` for DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def rng_state() -> dict[str, Any]:
    """Snapshot RNG state so a resumed run continues the exact same stream."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`rng_state`."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
