"""Metrics, calibration and error-analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dltf.evaluation.error_analysis import build_error_analysis, write_error_analysis
from dltf.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_array,
    reliability_curve,
    to_probabilities,
)


def test_perfect_predictions() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    y = np.array([0, 1, 0, 1])
    metrics = compute_classification_metrics(y, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.5


def test_logits_are_softmaxed() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    probs = to_probabilities(logits)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > probs[0, 1]


def test_confusion_matrix_shape_and_counts() -> None:
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 2])
    matrix = confusion_matrix_array(y_true, y_pred, num_classes=3)
    assert matrix.shape == (3, 3)
    assert matrix[1, 2] == 1 and matrix[2, 2] == 2
    assert matrix.sum() == 4


def test_calibration_bins_and_ece() -> None:
    # Perfectly confident and always right -> ECE == 0.
    probs = np.tile(np.array([[1.0, 0.0, 0.0]]), (10, 1))
    y = np.zeros(10, dtype=int)
    bins, ece = reliability_curve(y, probs, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert bins["count"].sum() == 10

    # Always right but only 60% confident -> ECE ~ 0.4.
    over_confident = np.tile(np.array([[0.6, 0.4]]), (10, 1))
    _, ece2 = reliability_curve(np.zeros(10, dtype=int), over_confident, n_bins=5)
    assert ece2 == pytest.approx(0.4, abs=1e-6)


def _synthetic_probs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(3), 10)
    logits = rng.normal(0, 1, size=(30, 3))
    logits[np.arange(30), y] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # Force a few confident mistakes.
    probs[0] = [0.02, 0.97, 0.01]
    y[0] = 0
    probs[1] = [0.01, 0.02, 0.97]
    y[1] = 1
    return probs, y


def test_error_analysis_payload() -> None:
    probs, y = _synthetic_probs()
    analysis = build_error_analysis(
        probs, y, class_names=["a", "b", "c"], task="tabular", max_misclassified=10
    )
    assert analysis["num_samples"] == 30
    assert analysis["num_misclassified"] == int((probs.argmax(1) != y).sum())
    assert len(analysis["confusion_matrix"]["matrix"]) == 3
    assert len(analysis["per_class"]) == 3
    assert analysis["top_confusions"]
    assert analysis["calibration"]["bins"]
    top_item = analysis["misclassified"]["items"][0]
    assert top_item["confidence"] >= 0.9
    assert top_item["top3"][0]["probability"] == pytest.approx(top_item["confidence"])
    assert analysis["findings_markdown"].startswith("- ")
    assert "accuracy" in analysis["findings_markdown"]


def test_error_analysis_files(tmp_path: Path) -> None:
    probs, y = _synthetic_probs()
    analysis = build_error_analysis(
        probs, y, class_names=["a", "b", "c"], max_misclassified=5
    )
    paths = write_error_analysis(analysis, out_dir=tmp_path / "out", web_dir=tmp_path / "web")
    for key in ("confusion_matrix", "per_class", "misclassified", "calibration", "full"):
        assert paths[key].exists(), key
        json.loads(paths[key].read_text())
    assert (tmp_path / "out" / "findings.md").stat().st_size > 100
    assert (tmp_path / "out" / "confusion_matrix.png").stat().st_size > 1000
    assert (tmp_path / "web" / "confusion_matrix.json").exists()
    assert (tmp_path / "web" / "misclassified_samples.json").exists()
