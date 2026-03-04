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
from sklearn.ensemble import RandomForestClassifier

from mm2026.utils.metrics import brier_score

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


@dataclass
class ModelBundle:
    feature_cols: list[str]
    base_models: dict[str, Any]
    base_prob_cols: list[str]
    blend_weights: dict[str, float]
    meta_model: LogisticRegression | None
    champion_raw_model: str
    calibration: str
    calibrator: Any


FEATURE_PREFIX = "diff_"


def _enabled_model_names(model_cfg: dict[str, Any]) -> list[str]:
    cfg = model_cfg.get("ensemble", {})
    names = list(cfg.get("enabled_base_models", ["logistic", "hgb", "elo"]))
    out = []
    for name in names:
        if name == "xgb" and XGBClassifier is None:
            continue
        if name == "catboost" and CatBoostClassifier is None:
            continue
        out.append(name)
    return out


def _fit_base_models(
    X: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any], enabled_models: list[str]
) -> dict[str, Any]:
    lr_cfg = model_cfg["base_models"]["logistic_regression"]
    gb_cfg = model_cfg["base_models"]["hist_gradient_boosting"]
    rf_cfg = model_cfg["base_models"].get("random_forest", {})
    xgb_cfg = model_cfg["base_models"].get("xgboost", {})
    cb_cfg = model_cfg["base_models"].get("catboost", {})

    out: dict[str, Any] = {}
    if "logistic" in enabled_models:
        lr = LogisticRegression(C=lr_cfg["C"], max_iter=lr_cfg["max_iter"], random_state=model_cfg["seed"])
        lr.fit(X, y)
        out["logistic"] = lr

    if "hgb" in enabled_models:
        gb = HistGradientBoostingClassifier(
            learning_rate=gb_cfg["learning_rate"],
            max_depth=gb_cfg["max_depth"],
            max_iter=gb_cfg["max_iter"],
            min_samples_leaf=gb_cfg["min_samples_leaf"],
            random_state=model_cfg["seed"],
        )
        gb.fit(X, y)
        out["hgb"] = gb

    if "rf" in enabled_models:
        rf = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 400)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            random_state=model_cfg["seed"],
            n_jobs=-1,
        )
        rf.fit(X, y)
        out["rf"] = rf

    if "xgb" in enabled_models and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 400)),
            max_depth=int(xgb_cfg.get("max_depth", 5)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
            subsample=float(xgb_cfg.get("subsample", 0.85)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.85)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
            min_child_weight=float(xgb_cfg.get("min_child_weight", 1.0)),
            random_state=model_cfg["seed"],
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        xgb.fit(X, y)
        out["xgb"] = xgb

    if "catboost" in enabled_models and CatBoostClassifier is not None:
        cb = CatBoostClassifier(
            iterations=int(cb_cfg.get("iterations", 600)),
            depth=int(cb_cfg.get("depth", 6)),
            learning_rate=float(cb_cfg.get("learning_rate", 0.03)),
            l2_leaf_reg=float(cb_cfg.get("l2_leaf_reg", 3.0)),
            random_seed=int(model_cfg["seed"]),
            loss_function="Logloss",
            verbose=False,
        )
        cb.fit(X, y)
        out["catboost"] = cb

    return out


def _predict_base(
    models: dict[str, Any],
    enabled_models: list[str],
    X: pd.DataFrame,
    elo_diff: np.ndarray,
    elo_scale: float,
) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for name in enabled_models:
        if name == "elo":
            out["p_elo"] = 1.0 / (1.0 + np.power(10.0, -elo_diff / elo_scale))
            continue
        if name not in models:
            continue
        out[f"p_{name}"] = models[name].predict_proba(X)[:, 1]
    return out


def _fit_meta(oof_base: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any]) -> LogisticRegression:
    meta = LogisticRegression(
        C=model_cfg["stacking"]["regularization_C"],
        max_iter=2000,
        random_state=model_cfg["seed"],
    )
    meta.fit(oof_base, y)
    return meta


def _blend_predictions(oof_base: pd.DataFrame, weights: dict[str, float], cols: list[str]) -> np.ndarray:
    raw = np.zeros(len(oof_base))
    for col in cols:
        raw += float(weights.get(col, 0.0)) * oof_base[col].to_numpy()
    return np.clip(raw, 1e-6, 1 - 1e-6)


def _score_weights_by_season(
    oof_base: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    cols: list[str],
    weights: dict[str, float],
) -> float:
    pred = _blend_predictions(oof_base, weights, cols)
    scores = []
    season_arr = seasons.to_numpy()
    y_arr = y.to_numpy()
    for season in np.unique(season_arr):
        mask = season_arr == season
        scores.append(brier_score(y_arr[mask], pred[mask]))
    return float(np.mean(scores))


def _tune_blend_weights(
    oof_base: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    cols: list[str],
    model_cfg: dict[str, Any],
) -> dict[str, float]:
    if len(cols) == 1:
        return {cols[0]: 1.0}

    rng = np.random.default_rng(int(model_cfg["seed"]))
    samples = int(model_cfg.get("ensemble", {}).get("blend_search_samples", 4000))
    best_w = {c: 1.0 / len(cols) for c in cols}
    best_score = _score_weights_by_season(oof_base, y, seasons, cols, best_w)

    # Random simplex search over 2021-2025 OOF seasons for direct leaderboard-aligned tuning.
    for _ in range(samples):
        proposal = rng.dirichlet(np.ones(len(cols)))
        weights = {c: float(w) for c, w in zip(cols, proposal)}
        score = _score_weights_by_season(oof_base, y, seasons, cols, weights)
        if score < best_score:
            best_score = score
            best_w = weights

    return best_w


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
    enabled_models = _enabled_model_names(model_cfg)
    if "elo" not in enabled_models:
        enabled_models.append("elo")

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

        base_models = _fit_base_models(X_tr, y_tr, model_cfg, enabled_models=enabled_models)
        elo_va = X_va["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X_va.columns else np.zeros(len(X_va))
        base_pred = _predict_base(base_models, enabled_models=enabled_models, X=X_va, elo_diff=elo_va, elo_scale=elo_scale)

        fold = base_pred.copy()
        fold["target"] = y_va.to_numpy()
        fold["Season"] = season
        fold.index = X_va.index
        oof_parts.append(fold)

        season_metrics[season] = {}
        for col in [c for c in base_pred.columns if c.startswith("p_")]:
            season_metrics[season][f"brier_{col}"] = brier_score(y_va.to_numpy(), base_pred[col].to_numpy())
        season_metrics[season]["brier_blend_equal"] = brier_score(
            y_va.to_numpy(), base_pred[[c for c in base_pred.columns if c.startswith("p_")]].mean(axis=1).to_numpy()
        )

    if not oof_parts:
        raise ValueError("Not enough data to produce rolling OOF predictions for training.")

    oof = pd.concat(oof_parts).sort_index()
    base_prob_cols = [c for c in oof.columns if c.startswith("p_")]
    blend_weights = _tune_blend_weights(
        oof_base=oof[base_prob_cols],
        y=oof["target"],
        seasons=oof["Season"],
        cols=base_prob_cols,
        model_cfg=model_cfg,
    )
    raw_blend_oof = _blend_predictions(oof[base_prob_cols], blend_weights, base_prob_cols)

    meta = None
    raw_meta_oof = None
    if bool(model_cfg["stacking"].get("use_meta_model", True)):
        meta = _fit_meta(oof[base_prob_cols], oof["target"], model_cfg)
        raw_meta_oof = meta.predict_proba(oof[base_prob_cols])[:, 1]

    champion_raw_model = "blend"
    champion_raw = raw_blend_oof
    if raw_meta_oof is not None and brier_score(oof["target"].to_numpy(), raw_meta_oof) < brier_score(
        oof["target"].to_numpy(), raw_blend_oof
    ):
        champion_raw_model = "meta"
        champion_raw = raw_meta_oof

    cal_method, calibrator, cal_scores = _fit_calibrator(
        raw_probs=champion_raw,
        y=oof["target"].to_numpy(),
        candidates=model_cfg["calibration"]["candidates"],
    )

    final_base = _fit_base_models(X, y, model_cfg, enabled_models=enabled_models)
    final_meta = None
    if meta is not None:
        final_base_pred = _predict_base(
            final_base,
            enabled_models=enabled_models,
            X=X,
            elo_diff=X["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X.columns else np.zeros(len(X)),
            elo_scale=elo_scale,
        )
        final_meta = _fit_meta(final_base_pred[base_prob_cols], y, model_cfg)

    bundle = ModelBundle(
        feature_cols=feature_cols,
        base_models=final_base,
        base_prob_cols=base_prob_cols,
        blend_weights=blend_weights,
        meta_model=final_meta,
        champion_raw_model=champion_raw_model,
        calibration=cal_method,
        calibrator=calibrator,
    )

    champion_oof_cal = _apply_calibration(champion_raw, cal_method, calibrator)
    report = {
        "rows": int(len(train_df)),
        "feature_count": int(len(feature_cols)),
        "enabled_models": enabled_models,
        "base_prob_cols": base_prob_cols,
        "blend_weights": blend_weights,
        "champion_raw_model": champion_raw_model,
        "holdout_seasons": holdouts,
        "season_metrics": season_metrics,
        "oof_brier_blend_raw": brier_score(oof["target"].to_numpy(), raw_blend_oof),
        "oof_brier_meta_raw": brier_score(oof["target"].to_numpy(), raw_meta_oof) if raw_meta_oof is not None else None,
        "oof_brier_champion_raw": brier_score(oof["target"].to_numpy(), champion_raw),
        "oof_brier_champion_calibrated": brier_score(oof["target"].to_numpy(), champion_oof_cal),
        "calibration": cal_method,
        "calibration_candidates": cal_scores,
    }
    for season in holdouts:
        key = int(season)
        if key not in season_metrics:
            continue
        season_mask = oof["Season"] == key
        y_season = oof.loc[season_mask, "target"].to_numpy()
        season_pred = _blend_predictions(oof.loc[season_mask, base_prob_cols], blend_weights, base_prob_cols)
        season_metrics[key]["brier_blend_tuned"] = brier_score(y_season, season_pred)
        if raw_meta_oof is not None:
            season_metrics[key]["brier_meta_raw"] = brier_score(y_season, raw_meta_oof[season_mask.to_numpy()])
        season_raw = champion_raw[season_mask.to_numpy()]
        season_cal = champion_oof_cal[season_mask.to_numpy()]
        season_metrics[key]["brier_champion_raw"] = brier_score(y_season, season_raw)
        season_metrics[key]["brier_champion_calibrated"] = brier_score(y_season, season_cal)
    return bundle, report


def predict_gender(bundle: ModelBundle, features_df: pd.DataFrame, model_cfg: dict[str, Any]) -> np.ndarray:
    X = features_df.reindex(columns=bundle.feature_cols).fillna(0.0)
    elo_scale = model_cfg["base_models"]["elo"]["scale"]
    elo_diff = X["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X.columns else np.zeros(len(X))

    enabled_models = [col.replace("p_", "") for col in bundle.base_prob_cols]
    base_pred = _predict_base(bundle.base_models, enabled_models=enabled_models, X=X, elo_diff=elo_diff, elo_scale=elo_scale)
    raw_blend = _blend_predictions(base_pred, bundle.blend_weights, bundle.base_prob_cols)

    if bundle.champion_raw_model == "meta" and bundle.meta_model is not None:
        raw = bundle.meta_model.predict_proba(base_pred[bundle.base_prob_cols])[:, 1]
    else:
        raw = raw_blend
    pred = _apply_calibration(raw, bundle.calibration, bundle.calibrator)
    return np.clip(pred, 0.0, 1.0)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    payload = {
        "feature_cols": bundle.feature_cols,
        "base_models": bundle.base_models,
        "base_prob_cols": bundle.base_prob_cols,
        "blend_weights": bundle.blend_weights,
        "meta_model": bundle.meta_model,
        "champion_raw_model": bundle.champion_raw_model,
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
        base_prob_cols=payload.get("base_prob_cols", ["p_logistic", "p_hgb", "p_elo"]),
        blend_weights=payload.get("blend_weights", {"p_logistic": 1 / 3, "p_hgb": 1 / 3, "p_elo": 1 / 3}),
        meta_model=payload["meta_model"],
        champion_raw_model=payload.get("champion_raw_model", "meta"),
        calibration=payload["calibration"],
        calibrator=payload["calibrator"],
    )
