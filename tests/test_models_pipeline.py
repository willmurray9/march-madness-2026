import pandas as pd
import pytest

from mm2026.models.pipeline import train_gender


def test_train_gender_requires_xgb_and_catboost_enabled() -> None:
    train_df = pd.DataFrame(
        {
            "Season": [2020, 2021],
            "target": [0, 1],
            "diff_elo_rating": [10.0, -5.0],
        }
    )
    model_cfg = {
        "seed": 42,
        "ensemble": {"enabled_base_models": ["logistic", "hgb", "elo"]},
        "base_models": {
            "elo": {"scale": 400.0},
            "logistic_regression": {"C": 1.0, "max_iter": 100},
            "hist_gradient_boosting": {"learning_rate": 0.1, "max_depth": 3, "max_iter": 50, "min_samples_leaf": 2},
            "xgboost": {},
            "catboost": {},
        },
        "stacking": {"use_meta_model": False, "regularization_C": 1.0},
        "calibration": {"candidates": ["none"]},
    }
    train_cfg = {"holdout_seasons": [2021]}

    with pytest.raises(ValueError, match="Missing required base models"):
        train_gender(train_df=train_df, model_cfg=model_cfg, train_cfg=train_cfg)
