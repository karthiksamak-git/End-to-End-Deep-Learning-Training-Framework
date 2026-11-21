"""Error analysis: what the model gets wrong, how confidently, and why it matters.

Produces a :class:`Trainer`-free, pure-data analysis (so it is unit-testable with
synthetic probabilities) plus a runner that collects predictions from a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dltf.data import DataBundle
from dltf.device import DeviceInfo
from dltf.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_array,
    per_class_report,
    reliability_curve,
)

logger = logging.getLogger("dltf.analysis")


def build_error_analysis(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    class_names: list[str],
    task: str = "tabular",
    max_misclassified: int = 30,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute confusion matrix, per-class metrics, top mistakes and calibration.

    Args:
        probabilities: ``(N, C)`` predicted probabilities.
        targets: ``(N,)`` integer labels.
        class_names: Index-aligned class labels.
        task: ``tabular`` or ``image`` (included in the payload for the UI).
        max_misclassified: How many of the most confident errors to keep.
        n_bins: Calibration bins.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(targets, dtype=np.int64).ravel()
    if probs.ndim != 2 or probs.shape[0] != y_true.size:
        raise ValueError("probabilities must be (N, C) and match targets")
    num_classes = probs.shape[1]
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    matrix = confusion_matrix_array(y_true, y_pred, num_classes=num_classes)
    per_class_df = per_class_report(y_true, y_pred, class_names=class_names)
    calibration_df, ece = reliability_curve(y_true, probs, n_bins=n_bins)
    overall = compute_classification_metrics(y_true, probs, class_names=class_names)

    wrong = np.flatnonzero(y_pred != y_true)
    wrong = wrong[np.argsort(-confidence[wrong])][:max_misclassified]
    misclassified = []
    for idx in wrong:
        order = np.argsort(-probs[idx])[:3]
        misclassified.append(
            {
                "sample_index": int(idx),
                "true_index": int(y_true[idx]),
                "pred_index": int(y_pred[idx]),
                "true_label": class_names[int(y_true[idx])],
                "pred_label": class_names[int(y_pred[idx])],
                "confidence": round(float(confidence[idx]), 4),
                "top3": [
                    {"label": class_names[int(j)], "probability": round(float(probs[idx, j]), 4)}
                    for j in order
                ],
            }
        )

    off_diagonal = [
        {
            "true_label": class_names[i],
            "pred_label": class_names[j],
            "count": int(matrix[i, j]),
        }
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j and matrix[i, j] > 0
    ]
    off_diagonal.sort(key=lambda row: row["count"], reverse=True)

    per_class = [
        {
            "class": name,
            "precision": round(float(per_class_df.loc[name, "precision"]), 4),
            "recall": round(float(per_class_df.loc[name, "recall"]), 4),
            "f1": round(float(per_class_df.loc[name, "f1"]), 4),
            "support": int(per_class_df.loc[name, "support"]),
        }
        for name in class_names
        if name in per_class_df.index
    ]

    total = int(y_true.size)
    n_wrong = int((y_pred != y_true).sum())
    weakest = min(per_class, key=lambda row: row["recall"]) if per_class else None
    top_pair = off_diagonal[0] if off_diagonal else None
    findings = [
        f"**{total}** held-out samples, **{n_wrong}** misclassified "
        f"({(n_wrong / total * 100) if total else 0:.2f}% error rate).",
        f"Overall accuracy **{overall['accuracy']:.4f}**, macro F1 **{overall['f1_macro']:.4f}**, "
        f"ROC-AUC (macro OVR) **{overall['roc_auc']:.4f}**.",
    ]
    if weakest:
        findings.append(
            f"Weakest class by recall: **{weakest['class']}** "
            f"(recall {weakest['recall']:.3f}, support {weakest['support']})."
        )
    if top_pair:
        findings.append(
            f"Most frequent confusion: **{top_pair['true_label']} -> "
            f"{top_pair['pred_label']}** ({top_pair['count']} cases)."
        )
    findings.append(
        f"Expected calibration error **{ece:.4f}** over {n_bins} confidence bins "
        f"(mean confidence {float(confidence.mean()):.3f} vs accuracy {overall['accuracy']:.3f})."
    )
    if misclassified:
        findings.append(
            f"Most confident error: true **{misclassified[0]['true_label']}** predicted "
            f"**{misclassified[0]['pred_label']}** with confidence "
            f"{misclassified[0]['confidence']:.3f}."
        )

    return {
        "task": task,
        "num_samples": total,
        "num_misclassified": n_wrong,
        "class_names": list(class_names),
        "overall": {k: (round(v, 6) if np.isfinite(v) else None) for k, v in overall.items()},
        "confusion_matrix": {
            "labels": list(class_names),
            "matrix": matrix.tolist(),
        },
        "per_class": per_class,
        "misclassified": {
            "total": n_wrong,
            "items": misclassified,
        },
        "top_confusions": off_diagonal[:10],
        "calibration": {
            "ece": round(ece, 6),
            "bins": [
                {
                    "bin": int(row.bin),
                    "lower": round(float(row.lower), 3),
                    "upper": round(float(row.upper), 3),
                    "count": int(row.count),
                    "avg_confidence": round(float(row.avg_confidence), 4),
                    "accuracy": round(float(row.accuracy), 4),
                }
                for row in calibration_df.itertuples()
            ],
        },
        "findings_markdown": "\n".join(f"- {line}" for line in findings),
    }


def _plot_confusion(analysis: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(analysis["confusion_matrix"]["matrix"])
    labels = analysis["confusion_matrix"]["labels"]
    size = max(4.0, 0.45 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_calibration(analysis: dict[str, Any], path: Path) -> None:
    bins = analysis["calibration"]["bins"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    if bins:
        ax.plot(
            [b["avg_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color="#4c78a8",
            label="model",
        )
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={analysis['calibration']['ece']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_error_analysis(
    analysis: dict[str, Any],
    *,
    out_dir: str | Path,
    web_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Persist the analysis as JSON, Markdown and PNG files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "confusion_matrix": out / "confusion_matrix.json",
        "per_class": out / "per_class_metrics.json",
        "misclassified": out / "misclassified_samples.json",
        "calibration": out / "calibration.json",
        "full": out / "error_analysis.json",
        "findings": out / "findings.md",
        "confusion_png": out / "confusion_matrix.png",
        "calibration_png": out / "calibration.png",
    }
    (out / "confusion_matrix.json").write_text(
        json.dumps(analysis["confusion_matrix"], indent=2), encoding="utf-8"
    )
    (out / "per_class_metrics.json").write_text(
        json.dumps({"classes": analysis["per_class"], "overall": analysis["overall"]}, indent=2),
        encoding="utf-8",
    )
    (out / "misclassified_samples.json").write_text(
        json.dumps(analysis["misclassified"], indent=2), encoding="utf-8"
    )
    (out / "calibration.json").write_text(
        json.dumps(analysis["calibration"], indent=2), encoding="utf-8"
    )
    (out / "error_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (out / "findings.md").write_text(
        "# Error analysis\n\n"
        f"Task: `{analysis['task']}` · samples: {analysis['num_samples']} · "
        f"misclassified: {analysis['num_misclassified']}\n\n"
        f"{analysis['findings_markdown']}\n",
        encoding="utf-8",
    )
    _plot_confusion(analysis, paths["confusion_png"])
    _plot_calibration(analysis, paths["calibration_png"])

    if web_dir is not None:
        web = Path(web_dir)
        web.mkdir(parents=True, exist_ok=True)
        for key, name in (
            ("confusion_matrix", "confusion_matrix.json"),
            ("per_class", "per_class_metrics.json"),
            ("misclassified", "misclassified_samples.json"),
            ("calibration", "calibration.json"),
            ("full", "error_analysis.json"),
        ):
            (web / name).write_text(paths[key].read_text(encoding="utf-8"), encoding="utf-8")
        (web / "findings.md").write_text(
            (out / "findings.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
    return paths


def _predict(model: Any, loader: Any, device_info: DeviceInfo) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device_info.device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(np.asarray(yb))
    return np.concatenate(probs), np.concatenate(targets)


def run_error_analysis(
    model: Any,
    bundle: DataBundle,
    *,
    device_info: DeviceInfo,
    out_dir: str | Path = "artifacts/error_analysis",
    web_dir: str | Path | None = "web/public/data",
    max_misclassified: int = 30,
    split: str = "test",
) -> dict[str, Any]:
    """Collect predictions for ``split``, analyse them and write the artifacts."""
    loader = getattr(bundle, f"{split}_loader")
    probs, targets = _predict(model, loader, device_info)
    analysis = build_error_analysis(
        probs,
        targets,
        class_names=bundle.class_names,
        task=bundle.task,
        max_misclassified=max_misclassified,
    )

    if bundle.task == "image":
        import torch

        mean = float(bundle.meta["image_mean"])
        std = float(bundle.meta["image_std"])
        for item in analysis["misclassified"]["items"]:
            tensor, _ = bundle.test_dataset[item["sample_index"]]
            arr = tensor.squeeze(0).float()
            pixels = ((arr * std + mean).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            item["pixels"] = pixels.reshape(-1).tolist()

    paths = write_error_analysis(analysis, out_dir=out_dir, web_dir=web_dir)
    logger.info(
        "error analysis | %d/%d misclassified | ECE %.4f | written to %s",
        analysis["num_misclassified"],
        analysis["num_samples"],
        analysis["calibration"]["ece"],
        paths["findings"].parent,
    )
    return analysis
