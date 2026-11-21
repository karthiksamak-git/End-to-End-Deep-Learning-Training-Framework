"""Stratified train/val/test splitting and k-fold utilities.

All helpers return integer index arrays so callers can slice raw arrays or
``torch.utils.data.Subset`` objects without copying data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def assert_disjoint(*index_sets: np.ndarray, names: Sequence[str] | None = None) -> None:
    """Raise ``AssertionError`` when two index sets share an element."""
    labels = list(names) if names else [f"set{i}" for i in range(len(index_sets))]
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            overlap = np.intersect1d(index_sets[i], index_sets[j])
            if overlap.size:
                raise AssertionError(
                    f"{labels[i]} and {labels[j]} overlap on {overlap.size} indices "
                    f"(e.g. {overlap[:5].tolist()})"
                )


def _as_targets(y: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("targets must be 1-D")
    return arr


def stratified_split(
    y: Sequence[int] | np.ndarray,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split indices into stratified ``train``/``val``/``test`` sets.

    The split is hierarchical (train+val vs test first, then train vs val) so each
    class keeps its proportion everywhere.  Class counts that are too small for the
    requested fractions fall back to ``train_test_split``'s best effort, and the
    result is validated for disjointness and non-empty train.
    """
    targets = _as_targets(y)
    n = targets.size
    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    indices = np.arange(n)
    # Guard against classes with a single member: stratifying on them raises.
    _, counts = np.unique(targets, return_counts=True)
    stratify = targets if counts.min() >= 2 else None

    if test_fraction > 0:
        rest_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=seed, stratify=stratify
        )
    else:
        rest_idx, test_idx = indices, np.array([], dtype=indices.dtype)

    if val_fraction > 0:
        rest_targets = targets[rest_idx]
        _, rest_counts = np.unique(rest_targets, return_counts=True)
        stratify_rest = rest_targets if rest_counts.min() >= 2 else None
        relative_val = val_fraction / (1.0 - test_fraction)
        train_idx, val_idx = train_test_split(
            rest_idx,
            test_size=relative_val,
            random_state=seed,
            stratify=stratify_rest,
        )
    else:
        train_idx, val_idx = rest_idx, np.array([], dtype=indices.dtype)

    if train_idx.size == 0:
        raise ValueError("stratified_split produced an empty training set")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{name} indices out of range")
    assert_disjoint(
        train_idx,
        val_idx,
        test_idx,
        names=("train", "val", "test"),
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def stratified_subsample(
    indices: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    max_samples: int | None,
    seed: int = 42,
) -> np.ndarray:
    """Take at most ``max_samples`` stratified indices from ``indices``."""
    if max_samples is None or max_samples >= indices.size:
        return np.asarray(indices)
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1 or None")
    targets = np.asarray(targets)
    subset_targets = targets[indices]
    picked, _ = train_test_split(
        np.asarray(indices),
        train_size=int(max_samples),
        random_state=seed,
        stratify=subset_targets if np.unique(subset_targets).size > 1 else None,
    )
    return np.sort(picked)


def stratified_kfold_indices(
    y: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(fit_idx, eval_idx)`` pairs from a seeded StratifiedKFold."""
    targets = _as_targets(y)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for fit_idx, eval_idx in splitter.split(np.zeros((targets.size, 1)), targets):
        assert_disjoint(fit_idx, eval_idx, names=("fold-train", "fold-val"))
        yield np.sort(fit_idx), np.sort(eval_idx)
