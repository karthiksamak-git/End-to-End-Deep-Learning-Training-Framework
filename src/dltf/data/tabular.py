"""Tabular pipeline: sklearn dataset -> Pandas EDA -> leak-free preprocessing.

The preprocessing ``Pipeline`` (median imputation + standardisation) is fitted on
the *training indices only*; validation and test rows are merely transformed. That
is what keeps the reported metrics free of leakage — and it is proven by a test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dltf.config import DataConfig
from dltf.data import DataBundle
from dltf.data.splits import stratified_split, stratified_subsample
from dltf.seed import make_generator, seed_worker

DATASETS = {
    "breast_cancer": (load_breast_cancer, ["malignant", "benign"]),
    "wine": (load_wine, None),
}


@dataclass
class TabularFrame:
    """Raw tabular data before any preprocessing."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    dataset_name: str

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.y).size)

    def as_frame(self) -> pd.DataFrame:
        """Return the features plus a ``target`` column as a DataFrame."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["target"] = self.y
        return df


def load_tabular_frame(name: str = "breast_cancer") -> TabularFrame:
    """Load a bundled scikit-learn dataset into a typed container."""
    if name not in DATASETS:
        raise ValueError(f"unknown tabular dataset {name!r}; available: {sorted(DATASETS)}")
    loader, names = DATASETS[name]
    bunch = loader()
    X = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target, dtype=np.int64)
    if y.min() != 0:  # wine targets are 0-based already, but be safe
        y = y - y.min()
    class_names = list(names) if names else [str(c) for c in bunch.target_names]
    class_names = [str(n) for n in class_names][: int(y.max()) + 1]
    return TabularFrame(
        X=X,
        y=y,
        feature_names=[str(f) for f in bunch.feature_names],
        class_names=class_names,
        dataset_name=name,
    )


def build_preprocessor() -> Pipeline:
    """Median imputation followed by standardisation."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def build_eda(frame: TabularFrame, out_dir: str | Path) -> dict[str, Any]:
    """Write Pandas EDA (JSON + PNG figures) and return the summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = frame.as_frame()
    features = df.drop(columns=["target"])

    class_counts = df["target"].value_counts().sort_index()
    balance = {
        frame.class_names[int(k)]: {
            "count": int(v),
            "fraction": float(v / len(df)),
        }
        for k, v in class_counts.items()
    }
    corr = features.corr()
    target_corr = features.corrwith(df["target"]).sort_values(key=np.abs, ascending=False)
    missing = features.isna().sum()
    summary: dict[str, Any] = {
        "dataset": frame.dataset_name,
        "n_samples": int(len(df)),
        "n_features": int(features.shape[1]),
        "dtypes": {c: str(t) for c, t in features.dtypes.items()},
        "missing_values": {c: int(v) for c, v in missing.items() if v},
        "n_missing_total": int(missing.sum()),
        "class_balance": balance,
        "feature_means": {c: float(v) for c, v in features.mean().items()},
        "feature_stds": {c: float(v) for c, v in features.std().items()},
        "top_correlated_with_target": {
            str(k): float(v) for k, v in target_corr.head(10).items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(
        [frame.class_names[int(k)] for k in class_counts.index],
        class_counts.to_numpy(),
        color=["#e45756", "#4c78a8"][: len(class_counts)],
    )
    ax.bar_label(bars)
    ax.set_title("Class balance")
    ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    ax.set_title("Feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    target_corr.head(12).iloc[::-1].plot.barh(ax=ax, color="#4c78a8")
    ax.set_title("Top |correlation| with target")
    ax.set_xlabel("Pearson r")
    fig.tight_layout()
    fig.savefig(out / "target_correlation.png", dpi=130)
    plt.close(fig)
    return summary


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    generator: torch.Generator | None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def build_tabular_loaders(
    frame: TabularFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[dict[str, DataLoader], Pipeline, dict[str, Any]]:
    """Fit the preprocessing pipeline on ``train_idx`` and build three loaders.

    Returns the loaders, the fitted pipeline and metadata about the split.
    """
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(frame.X[train_idx])
    X_val = preprocessor.transform(frame.X[val_idx]) if val_idx.size else np.empty((0, frame.X.shape[1]))
    X_test = preprocessor.transform(frame.X[test_idx]) if test_idx.size else np.empty((0, frame.X.shape[1]))

    generator = make_generator(seed)
    loaders = {
        "train": _make_loader(
            X_train,
            frame.y[train_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            generator=generator,
        ),
        "val": _make_loader(
            X_val,
            frame.y[val_idx],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            generator=None,
        ),
        "test": _make_loader(
            X_test,
            frame.y[test_idx],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            generator=None,
        ),
    }
    scaler: StandardScaler = preprocessor.named_steps["scale"]
    meta = {
        "preprocessing": "SimpleImputer(median) -> StandardScaler, fitted on train only",
        "feature_means": [float(v) for v in scaler.mean_],
        "feature_scales": [float(v) for v in scaler.scale_],
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    return loaders, preprocessor, meta


def make_tabular_bundle(
    cfg: DataConfig,
    *,
    seed: int = 42,
    eda_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> DataBundle:
    """Build the full tabular :class:`DataBundle` from a :class:`DataConfig`."""
    frame = load_tabular_frame(cfg.name)
    split = stratified_split(
        frame.y,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=seed,
    )
    train_idx = stratified_subsample(
        split["train"], frame.y, max_samples=cfg.subset_train, seed=seed
    )
    val_idx = stratified_subsample(split["val"], frame.y, max_samples=cfg.subset_val, seed=seed)
    test_idx = stratified_subsample(
        split["test"], frame.y, max_samples=cfg.subset_test, seed=seed
    )
    loaders, _preprocessor, meta = build_tabular_loaders(
        frame,
        train_idx,
        val_idx,
        test_idx,
        batch_size=batch_size or cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=seed,
    )
    eda = build_eda(frame, eda_dir) if eda_dir else {}
    meta.update(
        {
            "dataset": cfg.name,
            "features": frame.feature_names,
            "num_classes": frame.num_classes,
            "class_names": frame.class_names,
            "eda": {k: v for k, v in eda.items() if k != "feature_means"},
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
        }
    )
    return DataBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        input_shape=(frame.X.shape[1],),
        num_classes=frame.num_classes,
        class_names=frame.class_names,
        task="tabular",
        test_dataset=loaders["test"].dataset,
        generator=loaders["train"].generator,
        sizes={"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        meta=meta,
    )
