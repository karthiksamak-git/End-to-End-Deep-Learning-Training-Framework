"""Cross-validation and hyper-parameter search, using tiny configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dltf.config import Config, CVConfig, DataConfig, ModelConfig, OptimConfig, TrainConfig
from dltf.evaluation.cross_validation import run_cross_validation
from dltf.hpo.search import run_hpo


def _cv_config() -> Config:
    return Config(
        name="cv-test",
        task="tabular",
        seed=0,
        data=DataConfig(name="breast_cancer", batch_size=64, eval_batch_size=256),
        model=ModelConfig(name="mlp", hidden_sizes=[8], dropout=0.0),
        optim=OptimConfig(optimizer="adamw", lr=0.01, scheduler="cosine"),
        train=TrainConfig(epochs=2, amp=False, early_stopping_patience=0, monitor="val_loss"),
        cv=CVConfig(n_splits=3, epochs=2, shuffle=True),
    )


def test_cross_validation_writes_results(tmp_path: Path) -> None:
    cfg = _cv_config()
    result = run_cross_validation(
        cfg,
        out_dir=tmp_path / "artifacts",
        web_dir=tmp_path / "web",
        runs_dir=tmp_path / "runs",
        progress=False,
    )
    assert result["n_splits"] == 3
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert fold["test_size"] > 0
        assert fold["fold"] in (0, 1, 2)
    summary = result["summary"]
    assert set(summary) >= {"accuracy", "f1_macro", "roc_auc"}
    assert summary["accuracy"]["std"] >= 0
    assert (tmp_path / "artifacts" / "cv_results.csv").exists()
    payload = json.loads((tmp_path / "artifacts" / "cv_results.json").read_text())
    assert payload["summary"]["accuracy"]["mean"] == pytest.approx(summary["accuracy"]["mean"])
    assert (tmp_path / "web" / "cv_results.json").exists()


def test_hpo_finds_best_params(tmp_path: Path) -> None:
    from dltf.config import HPConfig

    cfg = Config(
        name="hpo-test",
        task="image",
        seed=0,
        data=DataConfig(
            name="fashion_mnist",
            batch_size=32,
            eval_batch_size=64,
            val_fraction=0.2,
            subset_train=64,
            subset_val=32,
            subset_test=None,
            num_workers=0,
        ),
        model=ModelConfig(name="cnn", width=8, depth=1, dropout=0.0),
        optim=OptimConfig(optimizer="adamw", lr=0.001, scheduler="onecycle"),
        train=TrainConfig(epochs=1, amp=False, early_stopping_patience=0),
        hpo=HPConfig(
            n_trials=2,
            study_name="pytest-hpo",
            metric="val_f1",
            subset_train=64,
            subset_val=32,
            epochs=1,
            pruner_startup_trials=1,
            search_space={
                "lr": [0.001, 0.002],
                "weight_decay": [0.0, 0.0],
                "dropout": [0.0, 0.0],
                "width": [8, 16],
                "batch_size": [32],
                "optimizer": ["adamw"],
            },
        ),
    )
    try:
        result = run_hpo(
            cfg,
            out_dir=tmp_path / "artifacts",
            runs_dir=tmp_path / "runs",
            data_dir="data",
            progress=False,
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - offline environments
        pytest.skip(f"Fashion-MNIST unavailable: {exc}")
    assert result["n_trials"] == 2
    assert result["n_completed"] >= 1
    assert "lr" in result["best_params"]
    assert result["best_value"] is not None
    saved = json.loads((tmp_path / "artifacts" / "hpo_results.json").read_text())
    assert saved["best_params"] == result["best_params"]
    best = json.loads((tmp_path / "artifacts" / "hpo_best_params.json").read_text())
    assert best["best_params"] == result["best_params"]
    assert len(saved["trials"]) == 2
