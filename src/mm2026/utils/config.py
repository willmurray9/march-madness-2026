from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    cfg_dir = Path(config_dir)
    return {
        "data": load_yaml(cfg_dir / "data.yaml"),
        "features": load_yaml(cfg_dir / "features.yaml"),
        "models": load_yaml(cfg_dir / "models.yaml"),
        "train": load_yaml(cfg_dir / "train.yaml"),
    }
