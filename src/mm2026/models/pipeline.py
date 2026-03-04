from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mm2026.utils.metrics import brier_score


@dataclass
class ModelBundle:
    feature_cols: list[str]
    base_models: dict[str, Any]
    meta_model: LogisticRegression | None
    calibration: str
    calibrator: Any


FEATURE_PREFIX = "diff_"


def _fit_base_models(X: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any]) -> dict[str, Any]:
    lr_cfg = model_cfg["base_models"]["logistic_regression"]
    gb_cfg = model_cfg["base_models"]["hist_gradient_boosting"]

    lr = LogisticRegression(C=lr_cfg["C"], max_iter=lr_cfg["max_iter"], random_state=model_cfg["seed"])
    lr.fit(X, y)

    gb = HistGradientBoostingClassifier(
        learning_rate=gb_cfg["learning_rate"],
        max_depth=gb_cfg["max_depth"],
        max_iter=gb_cfg["max_iter"],
        min_samples_leaf=gb_cfg["min_samples_leaf"],
        random_state=model_cfg["seed"],
    )
    gb.fit(X, y)

    return {"logistic": lr, "hgb": gb}


def _predict_base(models: dict[str, Any], X: pd.DataFrame, elo_diff: np.ndarray, elo_scale: float) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["p_logistic"] = models["logistic"].predict_proba(X)[:, 1]
    out["p_hgb"] = models["hgb"].predict_proba(X)[:, 1]
    out["p_elo"] = 1.0 / (1.0 + np.power(10.0, -elo_diff / elo_scale))
    return out


def _fit_meta(oof_base: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any]) -> LogisticRegression:
    meta = LogisticRegression(
        C=model_cfg["stacking"]["regularization_C"],
        max_iter=2000,
        random_state=model_cfg["seed"],
    )
    meta.fit(oof_base, y)
    return meta


def _fit_calibrator(
    raw_probs: np.ndarray,
    y: np.ndarray,
    candidates: list[str],
) -> tuple[str, Any, dict[str, float]]:
    raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    scores: dict[str, float] = {}

    scores["none"] = brier_score(y, raw_probs)

    platt = LogisticRegression(C=1.0, max_iter=2000)
    platt.fit(raw_probs.reshape(-1, 1), y)
    p_platt = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    scores["platt"] = brier_score(y, p_platt)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y)
    p_iso = iso.predict(raw_probs)
    scores["isotonic"] = brier_score(y, p_iso)

    allowed = [c for c in candidates if c in scores]
    best = min(allowed, key=lambda c: scores[c])

    if best == "none":
        return best, None, scores
    if best == "platt":
        return best, platt, scores
    return best, iso, scores


def _apply_calibration(raw_probs: np.ndarray, method: str, calibrator: Any) -> np.ndarray:
    raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    if method == "none" or calibrator is None:
        return raw_probs
    if method == "platt":
        return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    return calibrator.predict(raw_probs)


def train_gender(
    train_df: pd.DataFrame,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[ModelBundle, dict[str, Any]]:
    feature_cols = sorted([c for c in train_df.columns if c.startswith(FEATURE_PREFIX)])
    train_df = train_df.dropna(subset=["target", "Season"]).copy()
    X = train_df[feature_cols].fillna(0.0)
    y = train_df["target"].astype(int)

    holdouts = train_cfg["holdout_seasons"]
    elo_scale = model_cfg["base_models"]["elo"]["scale"]

    oof_parts = []
    season_metrics: dict[int, dict[str, float]] = {}

    for season in holdouts:
        tr_idx = train_df["Season"] < season
        va_idx = train_df["Season"] == season
        if tr_idx.sum() < 100 or va_idx.sum() == 0:
            continue

        X_tr = X.loc[tr_idx]
        y_tr = y.loc[tr_idx]
        X_va = X.loc[va_idx]
        y_va = y.loc[va_idx]

        base_models = _fit_base_models(X_tr, y_tr, model_cfg)
        elo_va = X_va["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X_va.columns else np.zeros(len(X_va))
        base_pred = _predict_base(base_models, X_va, elo_va, elo_scale)

        fold = base_pred.copy()
        fold["target"] = y_va.to_numpy()
        fold["Season"] = season
        fold.index = X_va.index
        oof_parts.append(fold)

        season_metrics[season] = {
            "brier_logistic": brier_score(y_va.to_numpy(), base_pred["p_logistic"].to_numpy()),
            "brier_hgb": brier_score(y_va.to_numpy(), base_pred["p_hgb"].to_numpy()),
            "brier_elo": brier_score(y_va.to_numpy(), base_pred["p_elo"].to_numpy()),
            "brier_blend_equal": brier_score(
                y_va.to_numpy(), base_pred[["p_logistic", "p_hgb", "p_elo"]].mean(axis=1).to_numpy()
            ),
        }

    if not oof_parts:
        raise ValueError("Not enough data to produce rolling OOF predictions for training.")

    oof = pd.concat(oof_parts).sort_index()
    meta_cols = ["p_logistic", "p_hgb", "p_elo"]
    meta = _fit_meta(oof[meta_cols], oof["target"], model_cfg)
    raw_meta_oof = meta.predict_proba(oof[meta_cols])[:, 1]

    cal_method, calibrator, cal_scores = _fit_calibrator(
        raw_probs=raw_meta_oof,
        y=oof["target"].to_numpy(),
        candidates=model_cfg["calibration"]["candidates"],
    )

    final_base = _fit_base_models(X, y, model_cfg)
    bundle = ModelBundle(
        feature_cols=feature_cols,
        base_models=final_base,
        meta_model=meta,
        calibration=cal_method,
        calibrator=calibrator,
    )

    meta_oof_cal = _apply_calibration(raw_meta_oof, cal_method, calibrator)
    report = {
        "rows": int(len(train_df)),
        "feature_count": int(len(feature_cols)),
        "holdout_seasons": holdouts,
        "season_metrics": season_metrics,
        "oof_brier_meta_raw": brier_score(oof["target"].to_numpy(), raw_meta_oof),
        "oof_brier_meta_calibrated": brier_score(oof["target"].to_numpy(), meta_oof_cal),
        "calibration": cal_method,
        "calibration_candidates": cal_scores,
    }
    return bundle, report


def predict_gender(bundle: ModelBundle, features_df: pd.DataFrame, model_cfg: dict[str, Any]) -> np.ndarray:
    X = features_df.reindex(columns=bundle.feature_cols).fillna(0.0)
    elo_scale = model_cfg["base_models"]["elo"]["scale"]
    elo_diff = X["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X.columns else np.zeros(len(X))

    base_pred = _predict_base(bundle.base_models, X, elo_diff, elo_scale)
    meta_cols = ["p_logistic", "p_hgb", "p_elo"]
    raw_meta = bundle.meta_model.predict_proba(base_pred[meta_cols])[:, 1]
    pred = _apply_calibration(raw_meta, bundle.calibration, bundle.calibrator)
    return np.clip(pred, 0.0, 1.0)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    payload = {
        "feature_cols": bundle.feature_cols,
        "base_models": bundle.base_models,
        "meta_model": bundle.meta_model,
        "calibration": bundle.calibration,
        "calibrator": bundle.calibrator,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, p)


def load_bundle(path: str | Path) -> ModelBundle:
    payload = joblib.load(path)
    return ModelBundle(
        feature_cols=payload["feature_cols"],
        base_models=payload["base_models"],
        meta_model=payload["meta_model"],
        calibration=payload["calibration"],
        calibrator=payload["calibrator"],
    )
