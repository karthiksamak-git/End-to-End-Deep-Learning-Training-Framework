"""Turn run outputs into the static JSON the Next.js dashboard reads.

Only summary data is copied into ``web/public/data`` — checkpoints, datasets and
logs stay in gitignored directories.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("dltf.web")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_experiments(runs_dir: str | Path = "runs") -> list[dict[str, Any]]:
    """Collect every completed run under ``runs_dir`` that has a ``summary.json``."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    experiments: list[dict[str, Any]] = []
    for run in sorted(root.iterdir()):
        summary_path = run / "summary.json"
        if not run.is_dir() or not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        metrics_path = run / "metrics.json"
        history = summary.get("history") or (
            _read_json(metrics_path).get("history", []) if metrics_path.exists() else []
        )
        experiments.append(
            {
                "id": run.name,
                "name": summary.get("name", run.name),
                "task": summary.get("task"),
                "created_at": summary.get("created_at"),
                "duration_sec": summary.get("train", {}).get("duration_sec"),
                "epochs_run": summary.get("train", {}).get("epochs_run"),
                "seed": summary.get("seed"),
                "device": summary.get("device", {}),
                "git_hash": summary.get("git_hash"),
                "model": summary.get("model", {}),
                "config": summary.get("config", {}),
                "history": history,
                "best": summary.get("train", {}).get("best", {}),
                "best_record": summary.get("train", {}).get("best_record"),
                "final": summary.get("train", {}).get("final"),
                "test_metrics": summary.get("test_metrics", {}),
                "sizes": summary.get("sizes", {}),
            }
        )
    return experiments


def build_model_card(
    *,
    config: dict[str, Any],
    experiment: dict[str, Any],
    test_metrics: dict[str, Any],
    onnx_info: dict[str, Any],
    parity: dict[str, Any],
    dataset_meta: dict[str, Any],
    git_hash: str,
) -> dict[str, Any]:
    """Assemble the model card shown on the overview page."""
    return {
        "name": config.get("name", "model"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "seed": config.get("seed"),
        "device": experiment.get("device", {}),
        "task": config.get("task"),
        "dataset": {
            "name": dataset_meta.get("dataset", config.get("data", {}).get("name")),
            "train_size": experiment.get("sizes", {}).get("train"),
            "val_size": experiment.get("sizes", {}).get("val"),
            "test_size": experiment.get("sizes", {}).get("test"),
            "image_mean": dataset_meta.get("image_mean"),
            "image_std": dataset_meta.get("image_std"),
            "preprocessing": dataset_meta.get("preprocessing"),
        },
        "architecture": {
            "class": experiment.get("model", {}).get("class", config.get("model", {}).get("name")),
            "parameters_total": experiment.get("model", {}).get("parameters_total"),
            "size_mb_fp32": experiment.get("model", {}).get("size_mb_fp32"),
            "config": config.get("model", {}),
        },
        "training": {
            "epochs_run": experiment.get("epochs_run"),
            "duration_sec": experiment.get("duration_sec"),
            "optimizer": config.get("optim", {}).get("optimizer"),
            "lr": config.get("optim", {}).get("lr"),
            "weight_decay": config.get("optim", {}).get("weight_decay"),
            "scheduler": config.get("optim", {}).get("scheduler"),
            "batch_size": config.get("data", {}).get("batch_size"),
            "amp": experiment.get("device", {}).get("amp_dtype"),
        },
        "metrics": test_metrics,
        "onnx": onnx_info,
        "onnx_parity": parity,
        "class_names": dataset_meta.get("class_names"),
        "limitations": [
            "Demo runs use subsets of Fashion-MNIST so the whole pipeline finishes in "
            "minutes on CPU; metrics are therefore not state-of-the-art.",
            "The browser playground runs the ONNX model with WASM on the client: no "
            "server-side inference, no uploads.",
        ],
    }


def write_web_artifacts(
    *,
    out_dir: str | Path = "web/public/data",
    experiments: list[dict[str, Any]],
    cv_results: dict[str, Any] | None = None,
    hpo_results: dict[str, Any] | None = None,
    error_analysis: dict[str, Any] | None = None,
    model_card: dict[str, Any] | None = None,
    playground_samples: dict[str, Any] | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Write every JSON file the dashboard imports.

    Returns a manifest of the written files and the headline metrics shown on the
    overview page.  Individual error-analysis files (confusion matrix, per-class
    metrics, misclassified samples, calibration) are copied from ``artifacts_dir``
    when present.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = Path(artifacts_dir)
    written: dict[str, str] = {}

    def dump(name: str, payload: Any) -> None:
        target = out / name
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written[name] = str(target)

    dump(
        "experiments.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": experiments,
        },
    )
    if cv_results is not None:
        dump("cv_results.json", cv_results)
    if hpo_results is not None:
        dump("hpo_trials.json", hpo_results)
    if error_analysis is not None:
        dump("confusion_matrix.json", error_analysis["confusion_matrix"])
        dump(
            "per_class_metrics.json",
            {"classes": error_analysis["per_class"], "overall": error_analysis["overall"]},
        )
        dump("misclassified_samples.json", error_analysis["misclassified"])
        dump("calibration.json", error_analysis["calibration"])
        dump("error_analysis.json", error_analysis)
        findings = artifacts / "error_analysis" / "findings.md"
        if findings.exists():
            shutil.copyfile(findings, out / "findings.md")
            written["findings.md"] = str(out / "findings.md")
    elif (artifacts / "error_analysis" / "error_analysis.json").exists():
        dump(
            "error_analysis.json",
            _read_json(artifacts / "error_analysis" / "error_analysis.json"),
        )
    if model_card is not None:
        dump("model_card.json", model_card)
    if playground_samples is not None:
        dump("playground_samples.json", playground_samples)

    headline = {
        "test_accuracy": (model_card or {}).get("metrics", {}).get("accuracy"),
        "test_f1_macro": (model_card or {}).get("metrics", {}).get("f1_macro"),
        "cv_accuracy_mean": (
            (cv_results or {}).get("summary", {}).get("accuracy", {}).get("mean")
        ),
        "cv_accuracy_std": ((cv_results or {}).get("summary", {}).get("accuracy", {}).get("std")),
        "hpo_best_value": (hpo_results or {}).get("best_value"),
        "hpo_metric": (hpo_results or {}).get("metric"),
        "n_runs": len(experiments),
    }
    dump("headline.json", headline)
    logger.info("wrote %d web artifacts to %s", len(written), out)
    return {"written": written, "headline": headline}


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
