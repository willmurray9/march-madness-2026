from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mm2026.models.pipeline import (
    _apply_calibration,
    _blend_predictions,
    _enabled_model_names,
    _fit_base_models,
    _fit_meta,
    _predict_base,
    load_bundle,
)
from mm2026.utils.io import ensure_dir, read_csv, write_csv, write_json
from mm2026.utils.metrics import brier_score


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _shap_values_to_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Binary classification outputs can have shape (n, d, 2); use positive class.
        arr = arr[:, :, -1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")
    return arr


def _champion_probs(
    base_pred: pd.DataFrame,
    base_prob_cols: list[str],
    blend_weights: dict[str, float],
    champion_raw_model: str,
    meta_model: Any,
    calibration: str,
    calibrator: Any,
) -> np.ndarray:
    raw_blend = _blend_predictions(base_pred[base_prob_cols], blend_weights, base_prob_cols)
    if champion_raw_model == "meta":
        if meta_model is None:
            raise ValueError("Champion model is meta but no meta model is available.")
        raw = meta_model.predict_proba(base_pred[base_prob_cols])[:, 1]
    else:
        raw = raw_blend
    return _apply_calibration(raw, calibration, calibrator)


def _fit_holdout_folds(
    train_df: pd.DataFrame,
    model_cfg: dict[str, Any],
    holdouts: list[int],
    feature_cols: list[str],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_df = train_df.dropna(subset=["target", "Season"]).copy()
    X_all = train_df[feature_cols].fillna(0.0)
    y_all = train_df["target"].astype(int)
    elo_scale = float(model_cfg["base_models"]["elo"]["scale"])
    enabled_models = _enabled_model_names(model_cfg)
    if "elo" not in enabled_models:
        enabled_models.append("elo")

    folds: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    for season in holdouts:
        tr_idx = train_df["Season"] < season
        va_idx = train_df["Season"] == season
        if tr_idx.sum() < 100 or va_idx.sum() == 0:
            continue
        X_tr = X_all.loc[tr_idx]
        y_tr = y_all.loc[tr_idx]
        X_va = X_all.loc[va_idx]
        y_va = y_all.loc[va_idx]

        base_models = _fit_base_models(X_tr, y_tr, model_cfg, enabled_models=enabled_models)
        elo_va = X_va["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X_va.columns else np.zeros(len(X_va))
        base_pred = _predict_base(base_models, enabled_models=enabled_models, X=X_va, elo_diff=elo_va, elo_scale=elo_scale)
        base_pred = base_pred.reindex(columns=[c for c in base_pred.columns if c.startswith("p_")])

        fold = {
            "season": int(season),
            "X_va": X_va,
            "y_va": y_va.to_numpy(),
            "base_models": base_models,
            "base_pred": base_pred,
            "enabled_models": enabled_models,
            "elo_scale": elo_scale,
        }
        folds.append(fold)

        part = base_pred.copy()
        part["target"] = y_va.to_numpy()
        part["Season"] = int(season)
        oof_parts.append(part)

    if not oof_parts:
        return [], pd.DataFrame()
    oof = pd.concat(oof_parts, ignore_index=True)
    return folds, oof


def _permutation_importance(
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    base_prob_cols: list[str],
    blend_weights: dict[str, float],
    champion_raw_model: str,
    meta_model: Any,
    calibration: str,
    calibrator: Any,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    for fold in folds:
        baseline_pred = _champion_probs(
            base_pred=fold["base_pred"],
            base_prob_cols=base_prob_cols,
            blend_weights=blend_weights,
            champion_raw_model=champion_raw_model,
            meta_model=meta_model,
            calibration=calibration,
            calibrator=calibrator,
        )
        fold["baseline_brier"] = brier_score(fold["y_va"], baseline_pred)

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        deltas: list[float] = []
        for fold in folds:
            X_perm = fold["X_va"].copy()
            X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
            elo_va = X_perm["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X_perm.columns else np.zeros(len(X_perm))
            base_pred_perm = _predict_base(
                fold["base_models"],
                enabled_models=fold["enabled_models"],
                X=X_perm,
                elo_diff=elo_va,
                elo_scale=float(fold["elo_scale"]),
            )
            brier_perm = brier_score(
                fold["y_va"],
                _champion_probs(
                    base_pred=base_pred_perm,
                    base_prob_cols=base_prob_cols,
                    blend_weights=blend_weights,
                    champion_raw_model=champion_raw_model,
                    meta_model=meta_model,
                    calibration=calibration,
                    calibrator=calibrator,
                ),
            )
            deltas.append(float(brier_perm - fold["baseline_brier"]))

        rows.append(
            {
                "feature": feature,
                "mean_brier_delta": float(np.mean(deltas)),
                "std_brier_delta": float(np.std(deltas)),
                "min_brier_delta": float(np.min(deltas)),
                "max_brier_delta": float(np.max(deltas)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_brier_delta", ascending=False).reset_index(drop=True)


def _logistic_importance(bundle: Any) -> pd.DataFrame:
    lr = bundle.base_models.get("logistic")
    if lr is None:
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])
    coef = np.asarray(lr.coef_)[0]
    out = pd.DataFrame({"feature": bundle.feature_cols, "coef": coef})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def _shap_importance_and_plots(
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    plots_dir: Path,
    gender: str,
    sample_rows_per_fold: int,
    seed: int,
) -> dict[str, Any]:
    model_names = ["hgb", "xgb", "catboost"]
    out: dict[str, Any] = {}

    for model_name in model_names:
        all_values: list[np.ndarray] = []
        all_features: list[pd.DataFrame] = []
        for idx, fold in enumerate(folds):
            model = fold["base_models"].get(model_name)
            if model is None or fold["X_va"].empty:
                continue
            n = min(sample_rows_per_fold, len(fold["X_va"]))
            X_s = fold["X_va"].sample(n=n, random_state=seed + idx)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_s)
            values_2d = _shap_values_to_2d(shap_values.values)
            all_values.append(values_2d)
            all_features.append(X_s)

        if not all_values:
            continue

        values = np.vstack(all_values)
        features = pd.concat(all_features, ignore_index=True)
        mean_abs = np.abs(values).mean(axis=0)
        imp_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        csv_path = plots_dir / f"{gender}_{model_name}_shap_importance.csv"
        write_csv(imp_df, csv_path)

        bar_path = plots_dir / f"{gender}_{model_name}_shap_bar.png"
        plt.figure(figsize=(10, 7))
        shap.summary_plot(values, features=features, feature_names=feature_cols, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=140)
        plt.close()

        beeswarm_path = plots_dir / f"{gender}_{model_name}_shap_beeswarm.png"
        plt.figure(figsize=(10, 8))
        shap.summary_plot(values, features=features, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=140)
        plt.close()

        out[model_name] = {
            "rows_used": int(len(features)),
            "importance_csv": str(csv_path),
            "bar_plot": str(bar_path),
            "beeswarm_plot": str(beeswarm_path),
            "top_features": imp_df.head(15).to_dict(orient="records"),
        }
    return out


def _generate_gender_explainability(
    gender: str,
    train_df: pd.DataFrame,
    bundle: Any,
    train_report: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    feature_cols = sorted([c for c in train_df.columns if c.startswith("diff_")])
    holdouts = [int(s) for s in train_cfg.get("holdout_seasons", [])]
    seed = int(model_cfg.get("seed", 42))

    folds, oof = _fit_holdout_folds(train_df, model_cfg=model_cfg, holdouts=holdouts, feature_cols=feature_cols)
    if not folds or oof.empty:
        return {
            "gender": gender,
            "status": "skipped",
            "reason": "No eligible holdout folds for explainability.",
        }

    base_prob_cols = list(train_report.get("base_prob_cols", []))
    blend_weights = {str(k): float(v) for k, v in train_report.get("blend_weights", {}).items()}
    champion_raw_model = str(train_report.get("champion_raw_model", "blend"))

    meta_model = None
    if champion_raw_model == "meta":
        meta_model = _fit_meta(oof[base_prob_cols], oof["target"], model_cfg)

    perm_df = _permutation_importance(
        folds=folds,
        feature_cols=feature_cols,
        base_prob_cols=base_prob_cols,
        blend_weights=blend_weights,
        champion_raw_model=champion_raw_model,
        meta_model=meta_model,
        calibration=bundle.calibration,
        calibrator=bundle.calibrator,
        seed=seed,
    )
    perm_path = out_dir / f"{gender}_permutation_importance.csv"
    write_csv(perm_df, perm_path)

    coef_df = _logistic_importance(bundle)
    coef_path = out_dir / f"{gender}_logistic_coefficients.csv"
    write_csv(coef_df, coef_path)

    plots_dir = ensure_dir(out_dir / "plots")
    shap_payload = _shap_importance_and_plots(
        folds=folds,
        feature_cols=feature_cols,
        plots_dir=plots_dir,
        gender=gender,
        sample_rows_per_fold=400,
        seed=seed,
    )

    return {
        "gender": gender,
        "status": "ok",
        "generated_at_utc": _now_utc_stamp(),
        "holdout_seasons": holdouts,
        "rows_scored": int(len(oof)),
        "feature_count": int(len(feature_cols)),
        "base_prob_cols": base_prob_cols,
        "champion_raw_model": champion_raw_model,
        "calibration": bundle.calibration,
        "artifacts": {
            "permutation_importance_csv": str(perm_path),
            "logistic_coefficients_csv": str(coef_path),
            "shap": shap_payload,
        },
        "top_features": {
            "permutation": perm_df.head(15).to_dict(orient="records"),
            "logistic_abs_coef": coef_df.head(15).to_dict(orient="records"),
            "shap_mean_abs": {k: v.get("top_features", []) for k, v in shap_payload.items()},
        },
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    train_cfg = cfg["train"]
    genders = data_cfg.get("genders", ["M", "W"])

    features_dir = Path(data_cfg["features_dir"])
    artifacts_dir = ensure_dir(data_cfg["artifacts_dir"])
    explain_dir = ensure_dir(Path(artifacts_dir) / "reports" / "explainability")
    models_dir = Path(artifacts_dir) / "models"
    reports_dir = Path(artifacts_dir) / "reports"

    payload: dict[str, Any] = {}
    for gender in genders:
        train_path = features_dir / f"{gender}_train_features.csv"
        train_report_path = reports_dir / f"{gender}_train_report.json"
        bundle_path = models_dir / f"{gender}_bundle.joblib"
        train_df = read_csv(train_path)
        train_report: dict[str, Any] = {}
        if train_report_path.exists():
            with train_report_path.open("r", encoding="utf-8") as f:
                train_report = json.load(f)

        if train_df.empty or not train_report_path.exists() or not bundle_path.exists():
            payload[gender] = {
                "gender": gender,
                "status": "skipped",
                "reason": "Required training artifacts are missing.",
            }
            continue

        bundle = load_bundle(bundle_path)
        result = _generate_gender_explainability(
            gender=gender,
            train_df=train_df,
            bundle=bundle,
            train_report=train_report,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            out_dir=explain_dir,
        )
        out_path = explain_dir / f"{gender}_explainability.json"
        write_json(result, out_path)
        result["path"] = str(out_path)
        payload[gender] = result
    return payload
