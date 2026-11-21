"""Classification metrics built on scikit-learn, plus calibration helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def to_probabilities(outputs: np.ndarray) -> np.ndarray:
    """Convert logits (or already-normalised probabilities) to probabilities."""
    arr = np.asarray(outputs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D outputs (N, C), got shape {arr.shape}")
    row_sums = arr.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-4) and arr.min() >= 0:
        # Renormalise: float32 softmax rows often sum to 1 -/+ 1e-7, which makes
        # scikit-learn's log_loss complain about non-probabilities.
        return arr / row_sums[:, None]
    return _softmax(arr)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute accuracy, macro precision/recall/F1, ROC-AUC and log loss.

    Args:
        y_true: Integer class labels.
        y_prob: ``(N, C)`` probabilities (logits are accepted and softmaxed).
        class_names: Optional labels used only for error messages.

    Returns:
        Dict of metric name -> float. ROC-AUC is ``nan`` when a split contains a
        single class, which is possible for tiny test fixtures.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    probs = to_probabilities(np.asarray(y_prob))
    if probs.shape[0] != y_true.size:
        raise ValueError(f"got {probs.shape[0]} probability rows for {y_true.size} labels")
    if probs.shape[1] <= y_true.max():
        raise ValueError("probability matrix has fewer columns than classes present in y_true")
    preds = probs.argmax(axis=1)

    n_classes = probs.shape[1]
    average = "binary" if n_classes == 2 else "macro"

    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # A degenerate split (single class present) has no defined ROC-AUC.
        auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average=average, zero_division=0)),
        "precision_macro": float(precision_score(y_true, preds, average=average, zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average=average, zero_division=0)),
        "roc_auc": auc,
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=list(range(n_classes))))
    except ValueError:
        metrics["log_loss"] = float("nan")
    if class_names is not None and len(class_names) < n_classes:
        raise ValueError("class_names has fewer entries than model outputs")
    return metrics


def confusion_matrix_array(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Confusion matrix with a fixed ``num_classes`` shape (rows = true)."""
    return confusion_matrix(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        labels=list(range(num_classes)),
    )


def per_class_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Per-class precision/recall/F1/support as a DataFrame."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows = {
        name: {k: report[name][k] for k in ("precision", "recall", "f1-score", "support")}
        for name in class_names
        if name in report
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.rename(columns={"f1-score": "f1"}).astype(float)


def reliability_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Confidence-vs-accuracy bins and expected calibration error.

    Returns:
        ``(bins_df, ece)`` where ``bins_df`` has one row per non-empty bin.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    probs = to_probabilities(np.asarray(y_prob))
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += count / len(y_true) * abs(avg_acc - avg_conf)
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": avg_acc,
                "gap": avg_acc - avg_conf,
            }
        )
    return pd.DataFrame(rows), float(ece)
