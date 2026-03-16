from mm2026.utils.config import resolve_data_seasons, resolve_feature_families


def test_resolve_feature_families_applies_gender_overrides() -> None:
    feature_cfg = {
        "feature_families": {
            "advanced_rates": False,
            "sos_adjusted": False,
            "volatility": False,
            "trend": False,
            "elo_upgrades": False,
        },
        "feature_families_by_gender": {
            "M": {"trend": True},
            "W": {"volatility": True, "elo_upgrades": True},
        },
    }

    assert resolve_feature_families(feature_cfg, "M") == {
        "advanced_rates": False,
        "sos_adjusted": False,
        "volatility": False,
        "trend": True,
        "elo_upgrades": False,
    }
    assert resolve_feature_families(feature_cfg, "W") == {
        "advanced_rates": False,
        "sos_adjusted": False,
        "volatility": True,
        "trend": False,
        "elo_upgrades": True,
    }


def test_resolve_data_seasons_validates_bounds() -> None:
    data_cfg = {
        "seasons": {
            "min_train_season": 2005,
            "max_train_season": 2024,
            "predict_season": 2026,
        }
    }
    assert resolve_data_seasons(data_cfg) == {
        "min_train_season": 2005,
        "max_train_season": 2024,
        "predict_season": 2026,
    }
