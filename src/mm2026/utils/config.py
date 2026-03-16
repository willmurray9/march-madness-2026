from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
FEATURE_FAMILY_KEYS = ["advanced_rates", "sos_adjusted", "volatility", "trend", "elo_upgrades"]


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


def resolve_feature_families(feature_cfg: dict[str, Any], gender: str | None = None) -> dict[str, bool]:
    shared = feature_cfg.get("feature_families", {})
    families = {key: bool(shared.get(key, False)) for key in FEATURE_FAMILY_KEYS}

    by_gender = feature_cfg.get("feature_families_by_gender", {})
    if gender is not None and isinstance(by_gender, dict):
        overrides = by_gender.get(gender, {})
        if isinstance(overrides, dict):
            for key in FEATURE_FAMILY_KEYS:
                if key in overrides:
                    families[key] = bool(overrides[key])
    return families


def resolve_data_seasons(data_cfg: dict[str, Any]) -> dict[str, int]:
    seasons_cfg = data_cfg.get("seasons", {})
    min_train = int(seasons_cfg.get("min_train_season", 2003))
    max_train = int(seasons_cfg.get("max_train_season", 2025))
    predict = int(seasons_cfg.get("predict_season", max_train + 1))
    if min_train > max_train:
        raise ValueError(f"Invalid season bounds: min_train_season={min_train} > max_train_season={max_train}")
    return {
        "min_train_season": min_train,
        "max_train_season": max_train,
        "predict_season": predict,
    }
