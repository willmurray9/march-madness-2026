from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mm2026.backtest.bracket2025 import (
    _actual_winner_map_for_2025,
    _build_matchup_feature_frame,
    _choose_predicted_winner,
    _load_snapshot_for_2025,
    _train_bundle_for_2025,
)
from mm2026.models.pipeline import (
    _apply_calibration,
    _blend_predictions,
    _predict_base,
    load_bundle,
)
from mm2026.utils.config import load_all_configs, resolve_data_seasons
from mm2026.utils.io import read_csv


SOURCE_2026 = "2026_predicted"
SOURCE_2025 = "2025_backtest"

MODEL_LABELS = {
    "p_logistic": "Logistic",
    "p_hgb": "HGB",
    "p_xgb": "XGBoost",
    "p_catboost": "CatBoost",
    "p_elo": "Elo",
}

STAT_SPECS = [
    ("seed_num", "Seed"),
    ("games_played", "Games Played"),
    ("elo_rating", "Elo Rating"),
    ("net_eff_season_mean", "Net Efficiency"),
    ("off_eff_season_mean", "Offensive Efficiency"),
    ("def_eff_season_mean", "Defensive Efficiency"),
    ("score_margin_season_mean", "Average Margin"),
    ("net_eff_roll_short", "Net Efficiency (Short)"),
    ("net_eff_roll_long", "Net Efficiency (Long)"),
    ("score_margin_roll_short", "Margin (Short)"),
    ("score_margin_roll_long", "Margin (Long)"),
    ("net_eff_trend_short_long", "Net Efficiency Trend"),
    ("score_margin_trend_short_long", "Margin Trend"),
    ("off_eff_trend_short_long", "Offensive Trend"),
    ("def_eff_trend_short_long", "Defensive Trend"),
]

EXCLUDED_DIFF_FEATURES = {
    "diff_seed_is_low_better",
    "diff_seed_gap_abs",
    "diff_seed_sum",
    "diff_elo_seed_interaction",
}


def _cfg() -> dict[str, Any]:
    return load_all_configs()


@lru_cache(maxsize=1)
def _model_cfg() -> dict[str, Any]:
    return _cfg()["models"]


@lru_cache(maxsize=1)
def _men_team_names() -> dict[int, str]:
    raw_dir = Path(_cfg()["data"]["raw_snapshot_dir"])
    df = read_csv(raw_dir / "MTeams.csv")
    if df.empty or "TeamID" not in df.columns or "TeamName" not in df.columns:
        return {}
    return {int(row.TeamID): str(row.TeamName) for row in df.itertuples(index=False)}


@lru_cache(maxsize=1)
def _men_current_bundle() -> Any:
    cfg = _cfg()
    return load_bundle(Path(cfg["data"]["artifacts_dir"]) / "models" / "M_bundle.joblib")


@lru_cache(maxsize=1)
def _men_current_inference() -> pd.DataFrame:
    cfg = _cfg()
    return read_csv(Path(cfg["data"]["features_dir"]) / "M_inference_features.csv")


@lru_cache(maxsize=1)
def _men_current_train_features() -> pd.DataFrame:
    cfg = _cfg()
    df = read_csv(Path(cfg["data"]["features_dir"]) / "M_train_features.csv")
    seasons = resolve_data_seasons(cfg["data"])
    if df.empty or "Season" not in df.columns:
        return df
    return df[df["Season"].between(seasons["min_train_season"], seasons["max_train_season"])].copy()


@lru_cache(maxsize=1)
def _men_2025_bundle() -> Any:
    return _train_bundle_for_2025(_cfg(), "M")


@lru_cache(maxsize=1)
def _men_2025_snapshot() -> pd.DataFrame:
    return _load_snapshot_for_2025(_cfg(), "M")


@lru_cache(maxsize=1)
def _men_2025_train_features() -> pd.DataFrame:
    cfg = _cfg()
    df = read_csv(Path(cfg["data"]["features_dir"]) / "M_train_features.csv")
    seasons = resolve_data_seasons(cfg["data"])
    if df.empty or "Season" not in df.columns:
        return df
    return df[df["Season"].between(seasons["min_train_season"], min(seasons["max_train_season"], 2024))].copy()


@lru_cache(maxsize=1)
def _men_2025_actual_winners() -> dict[tuple[int, int], int]:
    raw_dir = Path(_cfg()["data"]["raw_snapshot_dir"])
    return _actual_winner_map_for_2025(raw_dir, "M")


def _format_stat_label(name: str) -> str:
    exact = {key: label for key, label in STAT_SPECS}
    if name in exact:
        return exact[name]
    label = name
    label = label.replace("_season_mean", " season mean")
    label = label.replace("_roll_short", " short window")
    label = label.replace("_roll_mid", " mid window")
    label = label.replace("_roll_long", " long window")
    label = label.replace("_trend_short_long", " trend")
    label = label.replace("_vol_short", " volatility (short)")
    label = label.replace("_vol_mid", " volatility (mid)")
    label = label.replace("_vol_long", " volatility (long)")
    label = label.replace("_", " ")
    return " ".join(word.capitalize() for word in label.split())


def _format_numeric(value: Any, decimals: int = 2) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if float(value).is_integer():
        return int(value)
    return round(float(value), decimals)


def _feature_std_map(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    if train_df.empty:
        return {col: 1.0 for col in feature_cols}
    std = train_df.reindex(columns=feature_cols).fillna(0.0).std(ddof=0)
    out: dict[str, float] = {}
    for col in feature_cols:
        value = float(std.get(col, 0.0))
        out[col] = value if value > 1e-9 else 1.0
    return out


def _matchup_feature_row_2026(team_low_id: int, team_high_id: int) -> pd.DataFrame:
    df = _men_current_inference()
    if df.empty:
        raise ValueError("Missing men inference features.")
    row = df[(df["TeamID_low"] == int(team_low_id)) & (df["TeamID_high"] == int(team_high_id))].copy()
    if row.empty:
        raise ValueError(f"Missing 2026 inference row for matchup {team_low_id} vs {team_high_id}.")
    return row.iloc[[0]].copy()


def _matchup_feature_row_2025(team_low_id: int, team_high_id: int) -> pd.DataFrame:
    _low, _high, feat = _build_matchup_feature_frame(
        snapshot=_men_2025_snapshot(),
        season=2025,
        gender="M",
        team_a=int(team_low_id),
        team_b=int(team_high_id),
    )
    return feat


def _predict_breakdown(bundle: Any, feature_row: pd.DataFrame) -> dict[str, Any]:
    model_cfg = _model_cfg()
    X = feature_row.reindex(columns=bundle.feature_cols).fillna(0.0)
    elo_scale = float(model_cfg["base_models"]["elo"]["scale"])
    elo_diff = X["diff_elo_rating"].to_numpy() if "diff_elo_rating" in X.columns else np.zeros(len(X))
    enabled_models = [col.replace("p_", "") for col in bundle.base_prob_cols]
    base_pred = _predict_base(bundle.base_models, enabled_models=enabled_models, X=X, elo_diff=elo_diff, elo_scale=elo_scale)
    raw_blend = _blend_predictions(base_pred, bundle.blend_weights, bundle.base_prob_cols)
    if bundle.champion_raw_model == "meta" and bundle.meta_model is not None:
        raw_final = bundle.meta_model.predict_proba(base_pred[bundle.base_prob_cols])[:, 1]
    else:
        raw_final = raw_blend
    calibrated = _apply_calibration(raw_final, bundle.calibration, bundle.calibrator)

    agreement_rows: list[dict[str, Any]] = []
    for col in bundle.base_prob_cols:
        p_low = float(base_pred.iloc[0][col])
        agreement_rows.append(
            {
                "Model": MODEL_LABELS.get(col, col),
                "P(Low Wins)": p_low,
                "Pick": "Low" if p_low > 0.5 else ("High" if p_low < 0.5 else "Even"),
            }
        )

    raw_final_label = "Raw Final (Meta)" if bundle.champion_raw_model == "meta" else "Raw Final (Blend)"
    agreement_rows.append(
        {
            "Model": raw_final_label,
            "P(Low Wins)": float(raw_final[0]),
            "Pick": "Low" if float(raw_final[0]) > 0.5 else ("High" if float(raw_final[0]) < 0.5 else "Even"),
        }
    )
    agreement_rows.append(
        {
            "Model": "Calibrated Final",
            "P(Low Wins)": float(calibrated[0]),
            "Pick": "Low" if float(calibrated[0]) > 0.5 else ("High" if float(calibrated[0]) < 0.5 else "Even"),
        }
    )

    return {
        "agreement_rows": agreement_rows,
        "raw_final_low_win_prob": float(raw_final[0]),
        "calibrated_low_win_prob": float(calibrated[0]),
        "champion_raw_model": bundle.champion_raw_model,
        "calibration": bundle.calibration,
    }


def _team_comparison_rows(feature_row: pd.DataFrame) -> list[dict[str, Any]]:
    row = feature_row.iloc[0]
    rows: list[dict[str, Any]] = []
    for stat_key, label in STAT_SPECS:
        low_col = f"low_{stat_key}"
        high_col = f"high_{stat_key}"
        if low_col not in row.index or high_col not in row.index:
            continue
        rows.append(
            {
                "Metric": label,
                "Low Team": _format_numeric(row.get(low_col)),
                "High Team": _format_numeric(row.get(high_col)),
                "Low - High": _format_numeric(row.get(low_col) - row.get(high_col)),
            }
        )
    return rows


def _top_difference_rows(
    feature_row: pd.DataFrame,
    feature_cols: list[str],
    feature_stds: dict[str, float],
    limit: int = 8,
) -> list[dict[str, Any]]:
    row = feature_row.iloc[0]
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        if feature in EXCLUDED_DIFF_FEATURES:
            continue
        if not feature.startswith("diff_"):
            continue
        base = feature.replace("diff_", "", 1)
        low_col = f"low_{base}"
        high_col = f"high_{base}"
        if low_col not in row.index or high_col not in row.index:
            continue
        if pd.isna(row.get(low_col)) or pd.isna(row.get(high_col)):
            continue
        diff_value = float(row.get(feature, row.get(low_col) - row.get(high_col)))
        rows.append(
            {
                "Metric": _format_stat_label(base),
                "Low Team": _format_numeric(row.get(low_col)),
                "High Team": _format_numeric(row.get(high_col)),
                "Low - High": _format_numeric(diff_value),
                "Relative Gap": round(abs(diff_value) / float(feature_stds.get(feature, 1.0)), 2),
            }
        )
    rows = sorted(rows, key=lambda item: (float(item["Relative Gap"]), str(item["Metric"])), reverse=True)
    return rows[:limit]


def _winner_prob(prob_low: float, winner_side: str) -> float:
    return float(prob_low) if winner_side == "low" else float(1.0 - prob_low)


def _summary_text(
    predicted_winner_name: str,
    calibrated_low_win_prob: float,
    raw_low_win_prob: float,
    winner_side: str,
    pick_rule: str,
    top_differences: list[dict[str, Any]],
) -> str:
    winner_prob_cal = _winner_prob(calibrated_low_win_prob, winner_side)
    winner_prob_raw = _winner_prob(raw_low_win_prob, winner_side)
    if pick_rule == "raw_tiebreak":
        lead = (
            f"The bracket advances {predicted_winner_name} because calibration flattened this matchup to 50/50, "
            f"but the raw model still leaned {predicted_winner_name} ({winner_prob_raw:.1%})."
        )
    else:
        lead = f"The model favors {predicted_winner_name} with a calibrated win probability of {winner_prob_cal:.1%}."

    if not top_differences:
        return lead

    top_labels = [str(row["Metric"]) for row in top_differences[:3]]
    if len(top_labels) == 1:
        tail = top_labels[0]
    elif len(top_labels) == 2:
        tail = " and ".join(top_labels)
    else:
        tail = ", ".join(top_labels[:-1]) + f", and {top_labels[-1]}"
    if pick_rule == "raw_tiebreak":
        return lead + f" The largest measured gaps in this matchup are {tail}."
    if abs(winner_prob_raw - winner_prob_cal) >= 0.05:
        return lead + f" Calibration moved the final probability away from the raw score ({winner_prob_raw:.1%}). The largest measured gaps are {tail}."
    return lead + f" The largest measured gaps in this matchup are {tail}."


def build_men_matchup_explanation(source_key: str, team_low_id: int, team_high_id: int) -> dict[str, Any]:
    low = int(team_low_id)
    high = int(team_high_id)
    team_names = _men_team_names()

    if source_key == SOURCE_2026:
        feature_row = _matchup_feature_row_2026(low, high)
        bundle = _men_current_bundle()
        train_df = _men_current_train_features()
        season = 2026
        actual_winner = None
    elif source_key == SOURCE_2025:
        feature_row = _matchup_feature_row_2025(low, high)
        bundle = _men_2025_bundle()
        train_df = _men_2025_train_features()
        season = 2025
        actual_winner = _men_2025_actual_winners().get((low, high))
    else:
        raise ValueError(f"Unsupported matchup explanation source: {source_key}")

    breakdown = _predict_breakdown(bundle=bundle, feature_row=feature_row)
    calibrated_low_win_prob = float(breakdown["calibrated_low_win_prob"])
    raw_low_win_prob = float(breakdown["raw_final_low_win_prob"])
    predicted_winner, pick_rule = _choose_predicted_winner(
        low=low,
        high=high,
        pred=calibrated_low_win_prob,
        raw_pred=raw_low_win_prob,
    )
    winner_side = "low" if int(predicted_winner) == low else "high"
    winner_prob_cal = _winner_prob(calibrated_low_win_prob, winner_side)
    winner_prob_raw = _winner_prob(raw_low_win_prob, winner_side)

    feature_stds = _feature_std_map(train_df, bundle.feature_cols)
    top_differences = _top_difference_rows(feature_row, bundle.feature_cols, feature_stds)
    team_comparison = _team_comparison_rows(feature_row)

    actual_winner_name = team_names.get(int(actual_winner), f"Team {int(actual_winner)}") if actual_winner is not None else None
    actual_low_win = None if actual_winner is None else int(int(actual_winner) == low)
    squared_error = None if actual_low_win is None else float((calibrated_low_win_prob - actual_low_win) ** 2)

    return {
        "status": "ok",
        "source_key": source_key,
        "season": season,
        "team_low_id": low,
        "team_low_name": team_names.get(low, f"Team {low}"),
        "team_high_id": high,
        "team_high_name": team_names.get(high, f"Team {high}"),
        "predicted_winner_team_id": int(predicted_winner),
        "predicted_winner_name": team_names.get(int(predicted_winner), f"Team {int(predicted_winner)}"),
        "winner_prob_calibrated": winner_prob_cal,
        "winner_prob_raw": winner_prob_raw,
        "calibrated_low_win_prob": calibrated_low_win_prob,
        "raw_low_win_prob": raw_low_win_prob,
        "pick_rule": pick_rule,
        "agreement_rows": breakdown["agreement_rows"],
        "team_comparison": team_comparison,
        "top_differences": top_differences,
        "actual_matchup_occurred": actual_winner is not None,
        "actual_winner_team_id": None if actual_winner is None else int(actual_winner),
        "actual_winner_name": actual_winner_name,
        "squared_error": squared_error,
        "summary": _summary_text(
            predicted_winner_name=team_names.get(int(predicted_winner), f"Team {int(predicted_winner)}"),
            calibrated_low_win_prob=calibrated_low_win_prob,
            raw_low_win_prob=raw_low_win_prob,
            winner_side=winner_side,
            pick_rule=pick_rule,
            top_differences=top_differences,
        ),
        "calibration": breakdown["calibration"],
        "champion_raw_model": breakdown["champion_raw_model"],
    }
