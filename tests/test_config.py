"""Config loading, dotted overrides and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dltf.config import config_from_dict, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tabular_mlp.yaml"


def test_loads_yaml_config() -> None:
    cfg = load_config(CONFIG)
    assert cfg.task == "tabular"
    assert cfg.model.name == "mlp"
    assert cfg.model.hidden_sizes == [64, 32]
    assert cfg.data.name == "breast_cancer"


def test_dotted_overrides_coerce_types() -> None:
    cfg = load_config(
        CONFIG,
        ["model.dropout=0.33", "train.epochs=3", "model.hidden_sizes=[8,4]", "data.download=false"],
    )
    assert cfg.model.dropout == pytest.approx(0.33)
    assert cfg.train.epochs == 3
    assert cfg.model.hidden_sizes == [8, 4]
    assert cfg.data.download is False


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(CONFIG, ["model.nope=1"])


def test_validation_rejects_inconsistent_task() -> None:
    payload = yaml.safe_load(CONFIG.read_text())
    payload["model"]["name"] = "cnn"
    with pytest.raises(ValueError, match="tabular task requires"):
        config_from_dict(payload)


def test_roundtrip_to_yaml(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, ["train.epochs=2"])
    out = cfg.to_yaml(tmp_path / "nested" / "config.yaml")
    again = load_config(out)
    assert again.to_dict() == cfg.to_dict()


def test_bad_override_syntax() -> None:
    with pytest.raises(ValueError, match="key.subkey=value"):
        load_config(CONFIG, ["model.dropout"])
