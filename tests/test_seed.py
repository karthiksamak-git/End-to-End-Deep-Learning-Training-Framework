"""Seeding determinism, RNG round-trips and device resolution."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dltf.device import get_device
from dltf.seed import make_generator, rng_state, set_rng_state, set_seed


def test_same_seed_repeats_exactly() -> None:
    set_seed(1234, deterministic=False)
    first = (torch.randn(4), np.random.rand(4))
    set_seed(1234, deterministic=False)
    second = (torch.randn(4), np.random.rand(4))
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = torch.randn(3)
    set_seed(2)
    b = torch.randn(3)
    assert not torch.equal(a, b)


def test_rng_state_roundtrip() -> None:
    set_seed(11)
    state = rng_state()
    expected = torch.randn(2)
    set_rng_state(state)
    assert torch.equal(torch.randn(2), expected)


def test_device_resolution_cpu() -> None:
    info = get_device("cpu")
    assert isinstance(info.device, torch.device)
    assert info.kind == "cpu"
    assert isinstance(info.as_dict(), dict)


def test_auto_device_is_valid() -> None:
    info = get_device("auto")
    assert info.kind in ("cpu", "cuda", "mps")
    if not info.supports_amp:
        assert info.amp_dtype is None


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        get_device("tpu")


def test_generator_state_changes_after_seed() -> None:
    gen = make_generator(5)
    before = gen.get_state().clone()
    gen.manual_seed(9)
    assert not torch.equal(gen.get_state(), before)
