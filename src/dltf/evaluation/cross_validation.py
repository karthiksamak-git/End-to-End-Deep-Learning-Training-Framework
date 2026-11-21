"""Stratified k-fold cross-validation.

Each fold trains on a stratified inner split of the fold's training portion
(so early stopping never sees the evaluation fold) and reports metrics on the
untouched evaluation fold.  Preprocessing is refitted per fold — no leakage.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dltf.config import Config
from dltf.data.splits import stratified_kfold_indices
from dltf.data.tabular import build_tabular_loaders, load_tabular_frame
from dltf.device import get_device
from dltf.models.registry import build_model
from dltf.seed import set_seed
from dltf.training.trainer import Trainer

logger = logging.getLogger("dltf.cv")

METRIC_COLUMNS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "log_loss"]


def run_cross_validation(
    config: Config,
    *,
    out_dir: str | Path = "artifacts",
    web_dir: str | Path | None = "web/public/data",
    runs_dir: str | Path = "runs/cv",
    n_splits: int | None = None,
    epochs: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run stratified k-fold CV on the tabular task and persist the results.

    Returns a dict with per-fold rows, mean/std summary and the output paths.
    """
    if config.task != "tabular":
        raise ValueError("cross-validation is implemented for the tabular task")
    n_splits = int(n_splits or config.cv.n_splits)
    epochs = int(epochs or config.cv.epochs)
    device_info = get_device(config.device)
    frame = load_tabular_frame(config.data.name)

    rows: list[dict[str, Any]] = []
    for fold, (fit_idx, eval_idx) in enumerate(
        stratified_kfold_indices(
            frame.y, n_splits=n_splits, shuffle=config.cv.shuffle, seed=config.seed
        )
    ):
        fold_seed = config.seed + fold
        # Inner validation split of the fold's training data, for early stopping.
        train_idx, val_idx = train_test_split(
            fit_idx,
            test_size=0.15,
            random_state=fold_seed,
            stratify=frame.y[fit_idx],
        )
        set_seed(fold_seed, deterministic=config.train.deterministic)
        loaders, _, _ = build_tabular_loaders(
            frame,
            np.sort(train_idx),
            np.sort(val_idx),
            eval_idx,
            batch_size=config.data.batch_size,
            eval_batch_size=config.data.eval_batch_size,
            num_workers=config.data.num_workers,
            seed=fold_seed,
        )
        fold_cfg = copy.deepcopy(config)
        fold_cfg.train.epochs = epochs
        model = build_model(
            config.model,
            input_shape=(frame.X.shape[1],),
            num_classes=frame.num_classes,
        )
        trainer = Trainer(
            model,
            config=fold_cfg,
            device_info=device_info,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            num_classes=frame.num_classes,
            class_names=frame.class_names,
            run_dir=Path(runs_dir) / f"fold_{fold}",
            generator=loaders["train"].generator,
            progress=progress,
            save_checkpoints=False,
        )
        summary = trainer.fit()
        test_metrics = trainer.evaluate(loaders["test"])
        row: dict[str, Any] = {
            "fold": fold,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(eval_idx.size),
            "best_epoch": int(summary["best"]["epoch"]),
            "epochs_run": int(summary["epochs_run"]),
        }
        for column in METRIC_COLUMNS:
            row[column] = float(test_metrics[column])
        rows.append(row)
        logger.info(
            "fold %d/%d | acc=%.4f f1=%.4f auc=%.4f",
            fold + 1,
            n_splits,
            row["accuracy"],
            row["f1_macro"],
            row["roc_auc"],
        )

    df = pd.DataFrame(rows)
    summary_stats = {
        column: {
            "mean": float(df[column].mean()),
            "std": float(df[column].std(ddof=0)),
        }
        for column in METRIC_COLUMNS
    }
    result: dict[str, Any] = {
        "task": "tabular",
        "dataset": config.data.name,
        "model": config.model.name,
        "n_splits": n_splits,
        "epochs": epochs,
        "seed": config.seed,
        "device": device_info.as_dict(),
        "folds": rows,
        "summary": summary_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cv_results.csv", index=False)
    (out / "cv_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if web_dir is not None:
        web = Path(web_dir)
        web.mkdir(parents=True, exist_ok=True)
        (web / "cv_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "cv done | accuracy %.4f ± %.4f | f1 %.4f ± %.4f | auc %.4f ± %.4f",
        summary_stats["accuracy"]["mean"],
        summary_stats["accuracy"]["std"],
        summary_stats["f1_macro"]["mean"],
        summary_stats["f1_macro"]["std"],
        summary_stats["roc_auc"]["mean"],
        summary_stats["roc_auc"]["std"],
    )
    return result
