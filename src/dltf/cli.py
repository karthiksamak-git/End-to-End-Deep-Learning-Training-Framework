"""Command line interface: ``dltf train|cv|hpo|analyze|export|all``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dltf.config import Config, config_from_dict, load_config
from dltf.data import DataBundle
from dltf.data.image import make_image_bundle
from dltf.data.tabular import make_tabular_bundle
from dltf.device import DeviceInfo, get_device
from dltf.evaluation.cross_validation import run_cross_validation
from dltf.evaluation.error_analysis import run_error_analysis
from dltf.export.onnx_export import export_onnx, verify_onnx_parity, write_export_report
from dltf.export.web_artifacts import (
    build_model_card,
    gather_experiments,
    write_web_artifacts,
)
from dltf.hpo.search import run_hpo
from dltf.models.registry import build_model, model_summary
from dltf.seed import set_seed
from dltf.training.checkpoint import git_commit_hash, load_checkpoint, resolve_checkpoint
from dltf.training.metrics import compute_classification_metrics
from dltf.training.trainer import Trainer

logger = logging.getLogger("dltf")

DEFAULT_TABULAR_CONFIG = "configs/tabular_mlp.yaml"
DEFAULT_IMAGE_CONFIG = "configs/image_cnn.yaml"
DEFAULT_HPO_CONFIG = "configs/hpo_image_cnn.yaml"


# --------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_bundle(cfg: Config, *, eda_dir: str | Path | None = None) -> DataBundle:
    """Build the dataset bundle implied by ``cfg``."""
    if cfg.task == "tabular":
        return make_tabular_bundle(cfg.data, seed=cfg.seed, eda_dir=eda_dir)
    return make_image_bundle(cfg.data, data_dir=cfg.paths.data_dir, seed=cfg.seed)


def _build_model(cfg: Config, bundle: DataBundle) -> nn.Module:
    return build_model(cfg.model, input_shape=bundle.input_shape, num_classes=bundle.num_classes)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device_info: DeviceInfo,
    *,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Run the model over ``loader`` and return the full metric suite."""
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for xb, yb in loader:
        logits = model(xb.to(device_info.device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.append(yb)
    if not probs:
        return {}
    return compute_classification_metrics(
        torch.cat(targets).numpy(),
        torch.cat(probs).numpy(),
        class_names=class_names,
    )


def build_playground_samples(
    bundle: DataBundle, *, count: int = 12, seed: int = 0
) -> dict[str, Any]:
    """Pick real test images for the browser playground (denormalised pixels)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    total = len(bundle.test_dataset)
    positions = rng.choice(total, size=min(count, total), replace=False)
    mean = float(bundle.meta.get("image_mean", 0.0))
    std = float(bundle.meta.get("image_std", 1.0))
    samples = []
    for position in positions:
        tensor, label = bundle.test_dataset[int(position)]
        pixels = ((tensor.squeeze(0).float() * std + mean).clamp(0, 1) * 255).round()
        samples.append(
            {
                "index": int(position),
                "label": int(label),
                "label_name": bundle.class_names[int(label)],
                "pixels": pixels.to(torch.uint8).numpy().reshape(-1).tolist(),
            }
        )
    return {"generated_at": _now(), "samples": samples}


def find_latest_run(task: str, cfg: Config) -> Path:
    """Return the most recent completed run directory for ``task``."""
    root = Path(cfg.paths.runs_dir)
    candidates: list[tuple[float, Path]] = []
    for run in root.glob("*"):
        summary = run / "summary.json"
        if not summary.exists():
            continue
        try:
            payload = _read_json(summary)
        except json.JSONDecodeError:
            continue
        if payload.get("task") == task:
            candidates.append((summary.stat().st_mtime, run))
    if not candidates:
        raise FileNotFoundError(
            f"no completed {task} run under {root / '*'}; run 'dltf train' first"
        )
    return sorted(candidates)[-1][1]


def load_run(run_dir: str | Path) -> tuple[Config, nn.Module, DataBundle, DeviceInfo, dict]:
    """Rebuild the trained model from a run's best checkpoint.

    The config is restored from the checkpoint (not the YAML file) so the
    architecture, subset sizes and preprocessing always match the weights.
    """
    run_path = Path(run_dir)
    ckpt = load_checkpoint(resolve_checkpoint(run_path, "best"))
    cfg = config_from_dict(ckpt["config"])
    device_info = get_device(cfg.device)
    bundle = build_bundle(cfg)
    model = _build_model(cfg, bundle)
    model.load_state_dict(ckpt["model_state"])
    model.to(device_info.device)
    model.eval()
    logger.info(
        "loaded %s (epoch %s, %s=%.4f)",
        run_path,
        ckpt.get("epoch"),
        ckpt.get("best", {}).get("metric"),
        float(ckpt.get("best", {}).get("score") or float("nan")),
    )
    return cfg, model, bundle, device_info, ckpt


# ----------------------------------------------------------------- commands
def train_command(
    cfg: Config,
    *,
    resume: str | None = None,
    epochs: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one model, evaluate on the held-out split and write the run summary."""
    set_seed(cfg.seed, deterministic=cfg.train.deterministic)
    device_info = get_device(cfg.device, amp=cfg.train.amp)
    eda_dir = (
        Path(cfg.paths.artifacts_dir) / "eda" if cfg.task == "tabular" else None
    )
    bundle = build_bundle(cfg, eda_dir=eda_dir)
    model = _build_model(cfg, bundle)
    if epochs is not None:
        cfg.train.epochs = int(epochs)
    trainer = Trainer(
        model,
        config=cfg,
        device_info=device_info,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        generator=bundle.generator,
        progress=progress,
    )
    summary = trainer.fit(resume=resume)
    test_metrics = trainer.evaluate(bundle.test_loader)
    cfg.to_yaml(trainer.run_dir / "config.yaml")

    payload = {
        "name": cfg.name,
        "task": cfg.task,
        "created_at": _now(),
        "git_hash": git_commit_hash(),
        "seed": cfg.seed,
        "device": device_info.as_dict(),
        "model": model_summary(model),
        "config": cfg.to_dict(),
        "sizes": bundle.sizes,
        "dataset_meta": {
            k: bundle.meta.get(k)
            for k in ("dataset", "image_mean", "image_std", "preprocessing")
            if k in bundle.meta
        },
        "class_names": bundle.class_names,
        "input_shape": list(bundle.input_shape),
        "train": summary,
        "history": summary["history"],
        "test_metrics": test_metrics,
        "checkpoints": {
            "best": str(trainer.checkpoint_dir / "best.pt"),
            "last": str(trainer.checkpoint_dir / "last.pt"),
        },
    }
    (trainer.run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "run complete | dir=%s | test accuracy=%.4f f1=%.4f",
        trainer.run_dir,
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
    )
    return payload


def cv_command(cfg: Config, *, folds: int | None = None, epochs: int | None = None) -> dict[str, Any]:
    """Cross-validate the tabular model."""
    return run_cross_validation(
        cfg,
        out_dir=cfg.paths.artifacts_dir,
        web_dir=cfg.paths.web_data_dir,
        n_splits=folds,
        epochs=epochs,
    )


def hpo_command(cfg: Config, *, trials: int | None = None) -> dict[str, Any]:
    """Run the Optuna search."""
    return run_hpo(
        cfg,
        out_dir=cfg.paths.artifacts_dir,
        runs_dir=Path(cfg.paths.runs_dir) / "hpo",
        data_dir=cfg.paths.data_dir,
        n_trials=trials,
    )


def analyze_command(
    cfg: Config,
    *,
    run_dir: str | Path | None = None,
    task: str | None = None,
    max_misclassified: int = 30,
) -> dict[str, Any]:
    """Error-analyse the best model of a run."""
    task = task or cfg.task
    run_path = Path(run_dir) if run_dir else find_latest_run(task, cfg)
    run_cfg, model, bundle, device_info, ckpt = load_run(run_path)
    analysis = run_error_analysis(
        model,
        bundle,
        device_info=device_info,
        out_dir=Path(cfg.paths.artifacts_dir) / "error_analysis",
        web_dir=cfg.paths.web_data_dir,
        max_misclassified=max_misclassified,
    )
    payload = {
        "run_dir": str(run_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_metrics": {
            "accuracy": analysis["overall"]["accuracy"],
            "f1_macro": analysis["overall"]["f1_macro"],
            "roc_auc": analysis["overall"]["roc_auc"],
            "n_samples": analysis["num_samples"],
            "n_misclassified": analysis["num_misclassified"],
        },
        "findings_markdown": analysis["findings_markdown"],
    }
    (Path(cfg.paths.artifacts_dir) / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def export_command(cfg: Config, *, run_dir: str | Path | None = None) -> dict[str, Any]:
    """Export the image CNN to ONNX, verify parity and write the web artifacts."""
    run_path = Path(run_dir) if run_dir else find_latest_run("image", cfg)
    run_cfg, model, bundle, device_info, ckpt = load_run(run_path)

    onnx_path = Path(cfg.paths.web_model_dir) / "model.onnx"
    onnx_info = export_onnx(
        model,
        input_shape=bundle.input_shape,
        output_path=onnx_path,
        opset=17,
        device="cpu",
    )
    sample_input, _ = next(iter(bundle.test_loader))
    parity = verify_onnx_parity(
        model,
        onnx_path,
        input_shape=bundle.input_shape,
        sample_input=sample_input[:8],
    )
    if not parity["passed"]:
        raise RuntimeError(
            f"ONNX parity failed: max abs diff {parity['max_abs_diff']:.3e} >= {parity['atol']:.0e}"
        )
    logger.info(
        "exported %s (%.2f MB) | parity max|diff|=%.2e over %d inputs",
        onnx_path.name,
        onnx_info["size_mb"],
        parity["max_abs_diff"],
        parity["inputs_checked"],
    )

    test_metrics = evaluate_model(
        model, bundle.test_loader, device_info, class_names=bundle.class_names
    )
    history = ckpt.get("history", [])
    experiment = {
        "device": ckpt.get("meta", {}).get("device", device_info.as_dict()),
        "model": model_summary(model),
        "sizes": bundle.sizes,
        "epochs_run": len(history),
        "duration_sec": round(sum(float(r.get("seconds", 0)) for r in history), 2),
    }
    model_card = build_model_card(
        config=run_cfg.to_dict(),
        experiment=experiment,
        test_metrics=test_metrics,
        onnx_info=onnx_info,
        parity=parity,
        dataset_meta={**bundle.meta, "class_names": bundle.class_names},
        git_hash=git_commit_hash(),
    )
    artifacts = Path(cfg.paths.artifacts_dir)
    cv_results = _read_json(artifacts / "cv_results.json") if (artifacts / "cv_results.json").exists() else None
    hpo_results = _read_json(artifacts / "hpo_results.json") if (artifacts / "hpo_results.json").exists() else None
    analysis_path = artifacts / "error_analysis" / "error_analysis.json"
    analysis = _read_json(analysis_path) if analysis_path.exists() else None
    experiments = gather_experiments(cfg.paths.runs_dir)
    playground_samples = build_playground_samples(bundle)

    manifest = write_web_artifacts(
        out_dir=cfg.paths.web_data_dir,
        experiments=experiments,
        cv_results=cv_results,
        hpo_results=hpo_results,
        error_analysis=analysis,
        model_card=model_card,
        playground_samples=playground_samples,
        artifacts_dir=artifacts,
    )
    report = {
        "generated_at": _now(),
        "run_dir": str(run_path),
        "onnx": onnx_info,
        "parity": parity,
        "test_metrics": test_metrics,
        "web": manifest,
    }
    write_export_report(report, artifacts / "export_report.json")
    return {
        "onnx": onnx_info,
        "parity": parity,
        "test_metrics": test_metrics,
        "web_artifacts": sorted(manifest["written"]),
    }


def _best_param_overrides(best: dict[str, Any]) -> list[str]:
    mapping = {
        "lr": "optim.lr",
        "weight_decay": "optim.weight_decay",
        "dropout": "model.dropout",
        "width": "model.width",
        "depth": "model.depth",
        "batch_size": "data.batch_size",
        "optimizer": "optim.optimizer",
    }
    return [f"{mapping[k]}={v}" for k, v in best.items() if k in mapping]


def all_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full pipeline: tabular train + CV, HPO, image train, analyze, export."""
    tabular_cfg = load_config(args.tabular_config, args.override)
    image_cfg = load_config(args.image_config, [])
    hpo_cfg = load_config(args.hpo_config, [])

    tabular_run = train_command(tabular_cfg, epochs=args.tabular_epochs, progress=not args.quiet)
    cv_result = cv_command(tabular_cfg, folds=args.folds, epochs=args.cv_epochs)
    hpo_result = hpo_command(hpo_cfg, trials=args.trials)

    overrides = _best_param_overrides(hpo_result.get("best_params") or {})
    image_cfg = load_config(args.image_config, overrides)
    if args.image_epochs:
        image_cfg.train.epochs = int(args.image_epochs)
    image_run = train_command(image_cfg, progress=not args.quiet)
    analysis = analyze_command(image_cfg, run_dir=None)
    export = export_command(image_cfg, run_dir=None)
    return {
        "tabular": {
            "run_dir": tabular_run["checkpoints"]["best"],
            "test_metrics": tabular_run["test_metrics"],
        },
        "cv": cv_result["summary"],
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
        },
        "image": {
            "test_metrics": image_run["test_metrics"],
            "onnx": export["onnx"],
            "parity": export["parity"],
        },
        "error_analysis": analysis["test_metrics"],
    }


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``dltf`` console script."""
    parser = argparse.ArgumentParser(
        prog="dltf",
        description="End-to-end deep learning training framework",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, default_config: str) -> None:
        sp.add_argument("--config", default=default_config, help="YAML config path")
        sp.add_argument(
            "--override",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="dotted config override, e.g. --override model.dropout=0.3",
        )
        sp.add_argument("--name", default=None, help="override the run name")

    sp = sub.add_parser("train", help="train a model")
    add_common(sp, DEFAULT_TABULAR_CONFIG)
    sp.add_argument("--resume", default=None, help="checkpoint to resume from")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--quiet", action="store_true")

    sp = sub.add_parser("cv", help="stratified k-fold cross-validation")
    add_common(sp, DEFAULT_TABULAR_CONFIG)
    sp.add_argument("--folds", type=int, default=None)
    sp.add_argument("--epochs", type=int, default=None)

    sp = sub.add_parser("hpo", help="Optuna hyper-parameter search")
    add_common(sp, DEFAULT_HPO_CONFIG)
    sp.add_argument("--trials", type=int, default=None)

    sp = sub.add_parser("analyze", help="error analysis for a trained model")
    add_common(sp, DEFAULT_IMAGE_CONFIG)
    sp.add_argument("--run", default=None, help="run directory (defaults to latest)")
    sp.add_argument("--task", default=None, choices=["tabular", "image"])
    sp.add_argument("--max-misclassified", type=int, default=30)

    sp = sub.add_parser("export", help="export ONNX + write web artifacts")
    add_common(sp, DEFAULT_IMAGE_CONFIG)
    sp.add_argument("--run", default=None, help="run directory (defaults to latest image run)")

    sp = sub.add_parser("all", help="run the whole pipeline end to end")
    sp.add_argument("--tabular-config", default=DEFAULT_TABULAR_CONFIG)
    sp.add_argument("--image-config", default=DEFAULT_IMAGE_CONFIG)
    sp.add_argument("--hpo-config", default=DEFAULT_HPO_CONFIG)
    sp.add_argument("--override", action="append", default=[])
    sp.add_argument("--trials", type=int, default=None)
    sp.add_argument("--folds", type=int, default=None)
    sp.add_argument("--tabular-epochs", type=int, default=None)
    sp.add_argument("--cv-epochs", type=int, default=None)
    sp.add_argument("--image-epochs", type=int, default=None)
    sp.add_argument("--quiet", action="store_true")
    return parser


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, args.override)
    if getattr(args, "name", None):
        cfg.name = str(args.name)
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``dltf`` command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        if args.command == "train":
            cfg = _load(args)
            result = train_command(
                cfg, resume=args.resume, epochs=args.epochs, progress=not args.quiet
            )
            printable = {"run_dir": result["train"]["run_dir"], "test_metrics": result["test_metrics"]}
        elif args.command == "cv":
            result = cv_command(_load(args), folds=args.folds, epochs=args.epochs)
            printable = {"summary": result["summary"], "n_splits": result["n_splits"]}
        elif args.command == "hpo":
            result = hpo_command(_load(args), trials=args.trials)
            printable = {
                "best_params": result["best_params"],
                "best_value": result["best_value"],
                "n_trials": result["n_trials"],
                "n_pruned": result["n_pruned"],
            }
        elif args.command == "analyze":
            result = analyze_command(
                _load(args),
                run_dir=args.run,
                task=args.task,
                max_misclassified=args.max_misclassified,
            )
            printable = result
        elif args.command == "export":
            printable = export_command(_load(args), run_dir=args.run)
        elif args.command == "all":
            printable = all_command(args)
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"unknown command {args.command}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
