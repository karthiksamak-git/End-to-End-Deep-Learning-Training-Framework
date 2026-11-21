"""Data pipeline: stratified splits, leakage-free preprocessing, image shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dltf.config import DataConfig
from dltf.data.image import make_image_bundle
from dltf.data.splits import assert_disjoint, stratified_kfold_indices, stratified_split
from dltf.data.tabular import (
    build_tabular_loaders,
    load_tabular_frame,
    make_tabular_bundle,
)


def test_stratified_split_is_disjoint_and_keeps_ratios() -> None:
    y = np.repeat([0, 1, 2], 50)
    split = stratified_split(y, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert_disjoint(split["train"], split["val"], split["test"])
    assert split["train"].size + split["val"].size + split["test"].size == y.size
    for name in ("train", "val", "test"):
        fractions = np.bincount(y[split[name]], minlength=3) / split[name].size
        assert np.allclose(fractions, fractions[0], atol=0.02)


def test_kfold_indices_cover_everything_once() -> None:
    y = np.repeat([0, 1, 2, 3], 25)
    seen: list[int] = []
    folds = list(stratified_kfold_indices(y, n_splits=4, seed=0))
    assert len(folds) == 4
    for fit_idx, eval_idx in folds:
        assert_disjoint(fit_idx, eval_idx)
        assert set(eval_idx).isdisjoint(seen)
        seen.extend(eval_idx.tolist())
    assert sorted(seen) == list(range(y.size))


def test_preprocessor_fits_on_train_fold_only() -> None:
    frame = load_tabular_frame("breast_cancer")
    rng = np.random.default_rng(0)
    indices = rng.permutation(frame.X.shape[0])
    train_idx, val_idx, test_idx = indices[:300], indices[300:400], indices[400:]
    loaders, pipeline, meta = build_tabular_loaders(
        frame, train_idx, val_idx, test_idx, batch_size=32, num_workers=0
    )
    scaler = pipeline.named_steps["scale"]
    assert np.allclose(scaler.mean_, frame.X[train_idx].mean(axis=0), atol=1e-5)
    assert not np.allclose(scaler.mean_, frame.X.mean(axis=0), atol=1e-5), "leakage!"
    assert meta["split_sizes"] == {"train": 300, "val": 100, "test": 169}
    features, targets = next(iter(loaders["train"]))
    assert features.shape[1] == frame.X.shape[1]
    assert targets.dtype == torch.int64


def test_tabular_bundle_splits_and_shapes(tmp_path: Path) -> None:
    cfg = DataConfig(name="breast_cancer", batch_size=32, subset_train=200, subset_val=80)
    bundle = make_tabular_bundle(cfg, seed=1, eda_dir=tmp_path / "eda")
    assert bundle.task == "tabular"
    assert bundle.num_classes == 2
    assert bundle.sizes == {"train": 200, "val": 80, "test": 86}
    assert (tmp_path / "eda" / "eda_summary.json").exists()
    xb, yb = next(iter(bundle.train_loader))
    assert xb.shape == (32, 30)
    assert yb.max() < bundle.num_classes


def test_image_bundle_shapes_and_subsets() -> None:
    cfg = DataConfig(
        name="fashion_mnist",
        batch_size=16,
        eval_batch_size=64,
        val_fraction=0.2,
        subset_train=160,
        subset_val=64,
        subset_test=64,
        num_workers=0,
    )
    try:
        bundle = make_image_bundle(cfg, data_dir="data", seed=0)
    except (RuntimeError, OSError) as exc:  # pragma: no cover - offline environments
        pytest.skip(f"Fashion-MNIST unavailable: {exc}")
    assert bundle.input_shape == (1, 28, 28)
    assert bundle.sizes == {"train": 160, "val": 64, "test": 64}
    assert len(bundle.class_names) == 10
    xb, yb = next(iter(bundle.train_loader))
    assert xb.shape == (16, 1, 28, 28)
    assert torch.isfinite(xb).all()
    # Normalised training tensors should be roughly zero-centred.
    assert abs(float(xb.mean())) < 2.0
    assert bundle.meta["image_mean"] == pytest.approx(0.286, abs=0.05)
