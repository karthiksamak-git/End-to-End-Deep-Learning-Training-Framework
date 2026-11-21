"""Typed experiment configuration.

Every run is fully described by one YAML file.  :func:`load_config` parses it into
nested dataclasses, rejects unknown keys (a typo should never be silently ignored)
and applies dotted CLI overrides such as ``model.dropout=0.3``.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS = ("tabular", "image")
OPTIMIZERS = ("adamw", "adam", "sgd")
SCHEDULERS = ("none", "cosine", "onecycle", "plateau")
ACTIVATIONS = ("relu", "gelu", "elu", "tanh", "sigmoid", "leaky_relu")


@dataclass
class DataConfig:
    """Dataset and DataLoader settings."""

    name: str = "breast_cancer"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 0
    # Optional caps used to keep CPU demo runs fast (None = use everything).
    subset_train: int | None = None
    subset_val: int | None = None
    subset_test: int | None = None
    # Image-only options.
    augmentation: bool = True
    rotation_degrees: float = 10.0
    translate: float = 0.1
    download: bool = True


@dataclass
class ModelConfig:
    """Model factory settings (see :mod:`dltf.models.registry`)."""

    name: str = "mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout: float = 0.2
    batchnorm: bool = True
    # CNN-only options.
    width: int = 32
    depth: int = 2


@dataclass
class OptimConfig:
    """Optimizer and learning-rate scheduler settings."""

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"
    step_size: int = 10
    gamma: float = 0.5
    grad_clip: float | None = 1.0


@dataclass
class TrainConfig:
    """Training-loop settings."""

    epochs: int = 10
    amp: bool = True
    deterministic: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    monitor: str = "val_loss"
    log_every_n_steps: int = 0
    resume: str | None = None


@dataclass
class HPConfig:
    """Hyper-parameter search settings."""

    n_trials: int = 8
    timeout_seconds: int | None = None
    study_name: str = "cnn_hpo"
    direction: str = "maximize"
    metric: str = "val_f1"
    subset_train: int = 3000
    subset_val: int = 1000
    epochs: int = 2
    pruner_startup_trials: int = 2
    search_space: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "lr": [1e-4, 3e-3],
            "weight_decay": [0.0, 1e-2],
            "dropout": [0.0, 0.5],
            "width": [16, 32],
            "batch_size": [32, 64],
            "optimizer": ["adamw", "sgd"],
        }
    )


@dataclass
class CVConfig:
    """Cross-validation settings."""

    n_splits: int = 5
    epochs: int = 12
    shuffle: bool = True


@dataclass
class PathsConfig:
    """Filesystem locations (relative to the repository root)."""

    runs_dir: str = "runs"
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    web_data_dir: str = "web/public/data"
    web_model_dir: str = "web/public/models"


@dataclass
class Config:
    """Root experiment configuration."""

    name: str = "experiment"
    task: str = "tabular"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hpo: HPConfig = field(default_factory=HPConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        """Return a plain-JSON-serialisable dict of the whole config."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the config to ``path`` (created if needed)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return out

    def validate(self) -> None:
        """Raise ``ValueError`` on inconsistent settings."""
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.model.name not in ("mlp", "cnn"):
            raise ValueError(f"model.name must be 'mlp' or 'cnn', got {self.model.name!r}")
        if self.model.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.model.activation!r}")
        if self.optim.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optim.optimizer!r}")
        if self.optim.scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.optim.scheduler!r}")
        if not 0.0 <= self.data.val_fraction < 1.0:
            raise ValueError("data.val_fraction must be in [0, 1)")
        if not 0.0 <= self.data.test_fraction < 1.0:
            raise ValueError("data.test_fraction must be in [0, 1)")
        if self.data.val_fraction + self.data.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if self.task == "tabular" and self.model.name != "mlp":
            raise ValueError("tabular task requires model.name='mlp'")
        if self.task == "image" and self.model.name != "cnn":
            raise ValueError("image task requires model.name='cnn'")
        if self.cv.n_splits < 2:
            raise ValueError("cv.n_splits must be >= 2")


def _construct(cls: type, payload: dict[str, Any] | None, path: str) -> Any:
    """Build ``cls`` from a dict, recursing into nested dataclasses."""
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise ValueError(f"config section {path or 'root'!r} must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown config key(s) in {path or 'root'}: {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        target = hints.get(name)
        if isinstance(target, type) and dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _construct(target, value, f"{path}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(payload: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a nested dict."""
    cfg = _construct(Config, payload, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file and apply ``key.subkey=value`` overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    for item in overrides or []:
        _apply_override(raw, item)
    return config_from_dict(raw)


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot parse override value {text!r}: {exc}") from exc


def _apply_override(raw: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like key.subkey=value, got {item!r}")
    key, _, value = item.partition("=")
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"empty override key in {item!r}")
    node: dict[str, Any] = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {item!r} traverses a non-mapping value")
    node[parts[-1]] = _parse_scalar(value)
