"""Optuna hyper-parameter search (TPE sampler + median pruning).

Search spaces are declared in the YAML config (``hpo.search_space``) as
``name: [low, high]`` (or a list of choices for categorical parameters), which keeps
the experiment definition next to everything else.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from dltf.config import Config
from dltf.data.image import make_image_bundle
from dltf.device import get_device
from dltf.models.registry import build_model
from dltf.seed import set_seed
from dltf.training.callbacks import Callback
from dltf.training.trainer import Trainer

logger = logging.getLogger("dltf.hpo")

CATEGORICAL_PARAMS = {"optimizer", "activation", "scheduler"}


class OptunaPruningCallback(Callback):
    """Report each epoch to Optuna and raise ``TrialPruned`` when told to stop."""

    def __init__(self, trial: optuna.Trial, metric: str = "val_f1") -> None:
        self.trial = trial
        self.metric = metric

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: dict[str, Any]) -> None:
        """Report the intermediate value for pruning."""
        value = record.get(self.metric)
        if value is None:
            return
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {epoch}")


def suggest_trial_config(base: Config, trial: optuna.Trial) -> Config:
    """Materialise a trial's hyper-parameters into a full :class:`Config`."""
    cfg = copy.deepcopy(base)
    space = base.hpo.search_space
    cfg.name = f"hpo_trial_{trial.number}"
    cfg.train.epochs = base.hpo.epochs
    cfg.train.early_stopping_patience = 0
    cfg.data.subset_train = base.hpo.subset_train
    cfg.data.subset_val = base.hpo.subset_val
    cfg.data.subset_test = None

    if "lr" in space:
        cfg.optim.lr = trial.suggest_float("lr", float(space["lr"][0]), float(space["lr"][1]), log=True)
    if "weight_decay" in space:
        cfg.optim.weight_decay = trial.suggest_float(
            "weight_decay", float(space["weight_decay"][0]), float(space["weight_decay"][1])
        )
    if "dropout" in space:
        cfg.model.dropout = trial.suggest_float(
            "dropout", float(space["dropout"][0]), float(space["dropout"][1])
        )
    if "width" in space:
        cfg.model.width = trial.suggest_int(
            "width", int(space["width"][0]), int(space["width"][1]), step=8
        )
    if "depth" in space:
        cfg.model.depth = trial.suggest_int("depth", int(space["depth"][0]), int(space["depth"][1]))
    if "batch_size" in space:
        cfg.data.batch_size = trial.suggest_categorical(
            "batch_size", [int(v) for v in space["batch_size"]]
        )
    if "optimizer" in space:
        cfg.optim.optimizer = trial.suggest_categorical("optimizer", list(space["optimizer"]))
    cfg.validate()
    return cfg


def run_hpo(
    config: Config,
    *,
    out_dir: str | Path = "artifacts",
    runs_dir: str | Path = "runs/hpo",
    data_dir: str | Path = "data",
    n_trials: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the search and persist ``hpo_results.json`` + ``hpo_best_params.json``.

    Every trial is fully seeded (``seed + trial.number``) so a search repeats
    exactly.  The study itself uses a seeded TPE sampler and a median pruner.
    """
    set_seed(config.seed, deterministic=config.train.deterministic)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials_dir = Path(runs_dir)
    trials_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=config.hpo.study_name,
        direction=config.hpo.direction,
        sampler=TPESampler(seed=config.seed, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=config.hpo.pruner_startup_trials,
            n_warmup_steps=1,
        ),
    )
    device_info = get_device(config.device)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = suggest_trial_config(config, trial)
        set_seed(config.seed + trial.number, deterministic=True)
        bundle = make_image_bundle(trial_cfg.data, data_dir=data_dir, seed=config.seed)
        model = build_model(
            trial_cfg.model,
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
        )
        trainer = Trainer(
            model,
            config=trial_cfg,
            device_info=device_info,
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            num_classes=bundle.num_classes,
            class_names=bundle.class_names,
            run_dir=trials_dir / f"trial_{trial.number:03d}",
            callbacks=[OptunaPruningCallback(trial, config.hpo.metric)],
            generator=bundle.generator,
            progress=False,
            save_checkpoints=False,
        )
        summary = trainer.fit(write_outputs=True)
        record = summary["best_record"] or summary["final"]
        value = record.get(config.hpo.metric) if record else None
        if value is None:
            raise optuna.TrialPruned("metric unavailable")
        return float(value)

    study.optimize(
        objective,
        n_trials=n_trials or config.hpo.n_trials,
        timeout=config.hpo.timeout_seconds,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_payload = [
        {
            "number": t.number,
            "state": t.state.name.lower(),
            "value": t.value,
            "params": t.params,
            "duration_sec": round((t.duration.total_seconds() if t.duration else 0.0), 2),
            "started_at": t.datetime_start.isoformat() if t.datetime_start else None,
        }
        for t in study.trials
    ]
    try:
        importance = {
            k: float(v) for k, v in optuna.importance.get_param_importances(study).items()
        }
    except (ValueError, RuntimeError):
        importance = {}
    best_params = study.best_trial.params if completed else {}
    result: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": config.hpo.direction,
        "metric": config.hpo.metric,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "best_value": float(study.best_value) if completed else None,
        "best_params": best_params,
        "best_trial": study.best_trial.number if completed else None,
        "param_importance": importance,
        "search_space": config.hpo.search_space,
        "epochs_per_trial": config.hpo.epochs,
        "subset_train": config.hpo.subset_train,
        "subset_val": config.hpo.subset_val,
        "seed": config.seed,
        "trials": trials_payload,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "hpo_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "hpo_best_params.json").write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_value": result["best_value"],
                "metric": config.hpo.metric,
                "epochs_per_trial": config.hpo.epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "hpo done | %d trials (%d completed, %d pruned) | best %s=%.4f",
        result["n_trials"],
        result["n_completed"],
        result["n_pruned"],
        config.hpo.metric,
        result["best_value"] or float("nan"),
    )
    if progress:
        print(json.dumps(best_params, indent=2))
    return result
