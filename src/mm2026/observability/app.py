from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from mm2026.backtest.bracket2025 import run_current_bracket_forecast, run_submission_bracket_forecast
from mm2026.models.pipeline import load_bundle, predict_gender
from mm2026.observability.feature_dictionary import build_feature_dictionary_df
from mm2026.observability.matchup_explainer import (
    SOURCE_2025,
    SOURCE_2026,
    build_men_matchup_explanation,
)
from mm2026.utils.config import load_all_configs
from mm2026.utils.ids import parse_matchup_id


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _load_latest_snapshot(reports_dir: Path) -> dict[str, Any]:
    latest = reports_dir / "observability_latest.json"
    if latest.exists():
        return _load_json(latest)
    candidates = sorted(reports_dir.glob("observability_snapshot_*.json"))
    if not candidates:
        return {}
    return _load_json(candidates[-1])


def _train_rows(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    promotion = snapshot.get("promotion", {}).get("by_gender", {})
    for gender, report in snapshot.get("train_reports", {}).items():
        if not report:
            continue
        promo = promotion.get(gender, {}) if isinstance(promotion, dict) else {}
        rows.append(
            {
                "gender": gender,
                "rows": report.get("rows"),
                "feature_count": report.get("feature_count"),
                "oof_brier_champion_calibrated": report.get("oof_brier_champion_calibrated"),
                "oof_brier_champion_raw": report.get("oof_brier_champion_raw"),
                "oof_brier_blend_raw": report.get("oof_brier_blend_raw"),
                "oof_brier_meta_raw": report.get("oof_brier_meta_raw"),
                "champion_raw_model": report.get("champion_raw_model"),
                "calibration": report.get("calibration"),
                "lift_vs_baseline": promo.get("lift"),
                "promoted": promo.get("promoted"),
                "baseline_run_id": promo.get("baseline_run_id"),
            }
        )
    return pd.DataFrame(rows)


def _season_rows(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gender, report in snapshot.get("train_reports", {}).items():
        season_metrics = report.get("season_metrics", {}) if isinstance(report, dict) else {}
        for season, metrics in season_metrics.items():
            row = {"gender": gender, "season": int(season)}
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_summary(snapshot: dict[str, Any]) -> pd.DataFrame:
    submission = snapshot.get("submission", {})
    path = submission.get("path")
    if not path:
        return pd.DataFrame()
    sub_path = Path(path)
    if not sub_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(sub_path)
    if df.empty or "Pred" not in df.columns:
        return pd.DataFrame()

    bins = pd.cut(
        df["Pred"],
        bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    counts = bins.value_counts(sort=False)
    out = counts.reset_index()
    out.columns = ["bin", "count"]
    out["bin"] = out["bin"].astype(str)
    return out


def _dataset_description_map() -> dict[str, dict[str, str]]:
    return {
        "sample_submission_configured": {
            "contains": "Submission IDs required by the active stage file (season/team matchups).",
            "used_for": "Defines exact prediction coverage and final ID validation.",
        },
        "m_regular_season_long": {
            "contains": "Men regular-season games in team-game long format with score/outcome context.",
            "used_for": "Builds rolling team form and efficiency features.",
        },
        "w_regular_season_long": {
            "contains": "Women regular-season games in team-game long format with score/outcome context.",
            "used_for": "Builds rolling team form and efficiency features.",
        },
        "m_tourney_compact": {
            "contains": "Men NCAA tournament historical game outcomes.",
            "used_for": "Creates labeled training matchups (`target`).",
        },
        "w_tourney_compact": {
            "contains": "Women NCAA tournament historical game outcomes.",
            "used_for": "Creates labeled training matchups (`target`).",
        },
        "m_tourney_seeds": {
            "contains": "Men tournament seed assignments by season/team.",
            "used_for": "Adds seed-based matchup features.",
        },
        "w_tourney_seeds": {
            "contains": "Women tournament seed assignments by season/team.",
            "used_for": "Adds seed-based matchup features.",
        },
        "m_seasons": {
            "contains": "Men season metadata.",
            "used_for": "Season context and sanity checks.",
        },
        "w_seasons": {
            "contains": "Women season metadata.",
            "used_for": "Season context and sanity checks.",
        },
        "m_train_features": {
            "contains": "Men historical tournament matchup feature deltas plus `target`.",
            "used_for": "Trains/validates men models with rolling holdouts.",
        },
        "w_train_features": {
            "contains": "Women historical tournament matchup feature deltas plus `target`.",
            "used_for": "Trains/validates women models with rolling holdouts.",
        },
        "m_inference_features": {
            "contains": "Men Stage submission matchup feature deltas, no target.",
            "used_for": "Generates men prediction probabilities for submission.",
        },
        "w_inference_features": {
            "contains": "Women Stage submission matchup feature deltas, no target.",
            "used_for": "Generates women prediction probabilities for submission.",
        },
        "m_team_snapshot": {
            "contains": "Men latest season team-level snapshot stats at inference cutoff.",
            "used_for": "Explains feature state feeding inference matchup deltas.",
        },
        "w_team_snapshot": {
            "contains": "Women latest season team-level snapshot stats at inference cutoff.",
            "used_for": "Explains feature state feeding inference matchup deltas.",
        },
    }


def _model_method_description_map() -> dict[str, str]:
    return {
        "logistic": "Regularized linear baseline on matchup feature deltas.",
        "hgb": "Histogram gradient-boosted trees for non-linear tabular patterns.",
        "rf": "Random forest ensemble over tabular features.",
        "xgb": "XGBoost boosted trees.",
        "catboost": "CatBoost boosted trees.",
        "elo": "Hand-crafted Elo probability curve from Elo rating deltas.",
        "blend": "Weighted average of base model probabilities.",
        "meta": "Stacked logistic meta-model trained on base model outputs.",
        "none": "No probability calibration.",
        "platt": "Logistic (Platt) calibration of raw probabilities.",
        "isotonic": "Isotonic calibration of raw probabilities.",
    }


def _dataset_overview_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    files_df = pd.DataFrame(snapshot.get("file_summaries", []))
    if files_df.empty:
        return pd.DataFrame()
    desc_map = _dataset_description_map()
    files_df["contains"] = files_df["name"].map(lambda x: desc_map.get(str(x), {}).get("contains", ""))
    files_df["used_for"] = files_df["name"].map(lambda x: desc_map.get(str(x), {}).get("used_for", ""))
    return files_df.sort_values(["category", "gender", "name"]).reset_index(drop=True)


def _model_overview_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    desc = _model_method_description_map()
    for gender, report in snapshot.get("train_reports", {}).items():
        if not isinstance(report, dict) or not report:
            continue
        enabled_models = report.get("enabled_models", [])
        for m in enabled_models:
            rows.append(
                {
                    "gender": gender,
                    "method": m,
                    "role": "base_model",
                    "selected": True,
                    "description": desc.get(m, ""),
                }
            )
        rows.append(
            {
                "gender": gender,
                "method": report.get("champion_raw_model"),
                "role": "final_raw_selector",
                "selected": True,
                "description": desc.get(str(report.get("champion_raw_model")), ""),
            }
        )
        rows.append(
            {
                "gender": gender,
                "method": report.get("calibration"),
                "role": "calibration",
                "selected": True,
                "description": desc.get(str(report.get("calibration")), ""),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def _load_submission_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    sub_path = snapshot.get("submission", {}).get("path")
    if not sub_path:
        return pd.DataFrame()
    p = Path(sub_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Pred" not in df.columns:
        return pd.DataFrame()
    return df


def _load_bracket_backtest(snapshot: dict[str, Any]) -> dict[str, Any]:
    backtest_path = snapshot.get("backtests", {}).get("bracket_2025", {}).get("path")
    if not backtest_path:
        return {}
    path = Path(backtest_path)
    if not path.exists():
        return {}
    return _load_json(path)


def _load_explainability_payload(snapshot: dict[str, Any], gender: str) -> dict[str, Any]:
    reports = snapshot.get("explainability_reports", {})
    report = reports.get(gender, {}) if isinstance(reports, dict) else {}
    path = report.get("path") if isinstance(report, dict) else None
    if path and Path(path).exists():
        return _load_json(Path(path))
    return report if isinstance(report, dict) else {}


def _round_sort_key(label: str) -> int:
    order = {
        "Play-In": 0,
        "Round of 64": 1,
        "Round of 32": 2,
        "Sweet 16": 3,
        "Elite 8": 4,
        "Final Four": 5,
        "Championship": 6,
    }
    return order.get(str(label), 99)


def _bracket_table_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df
    keep = [
        "round_label",
        "slot",
        "team_low_name",
        "team_low_seed",
        "team_high_name",
        "team_high_seed",
        "pred_team_low_win",
        "pred_team_low_win_raw",
        "winner_decision_rule",
        "winner_team_name",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values(["round_label", "slot"], key=lambda s: s.map(_round_sort_key) if s.name == "round_label" else s)
    rename = {
        "round_label": "Round",
        "slot": "Slot",
        "team_low_name": "Team Low",
        "team_low_seed": "Low Seed",
        "team_high_name": "Team High",
        "team_high_seed": "High Seed",
        "pred_team_low_win": "P(Low Wins)",
        "pred_team_low_win_raw": "Raw P(Low Wins)",
        "winner_decision_rule": "Pick Rule",
        "winner_team_name": "Winner",
    }
    return out.rename(columns=rename)


def _brier_example_df() -> pd.DataFrame:
    demo = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0, 1],
            "y_pred": [0.84, 0.30, 0.62, 0.20, 0.53],
        }
    )
    demo["squared_error"] = (demo["y_pred"] - demo["y_true"]) ** 2
    return demo


@st.cache_data
def _load_team_name_map() -> dict[int, str]:
    out: dict[int, str] = {}
    for path in [Path("data/raw/latest/MTeams.csv"), Path("data/raw/latest/WTeams.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "TeamID" not in df.columns or "TeamName" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            out[int(row.TeamID)] = str(row.TeamName)
    return out


@st.cache_data
def _load_seed_map() -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for path in [Path("data/curated/latest/M_tourney_seeds.csv"), Path("data/curated/latest/W_tourney_seeds.csv")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Season" not in df.columns or "TeamID" not in df.columns:
            continue
        seed_col = "seed_num" if "seed_num" in df.columns else "Seed"
        if seed_col not in df.columns:
            continue
        if seed_col == "Seed":
            parsed = df["Seed"].astype(str).str.extract(r"(\d{2})", expand=False)
            df = df.assign(seed_num=pd.to_numeric(parsed, errors="coerce"))
            seed_col = "seed_num"
        for row in df[["Season", "TeamID", seed_col]].dropna().itertuples(index=False):
            out[(int(row.Season), int(row.TeamID))] = float(getattr(row, seed_col))
    return out


def _infer_gender(low_id: int, high_id: int) -> str:
    if 1000 <= low_id < 2000 and 1000 <= high_id < 2000:
        return "M"
    if 3000 <= low_id < 4000 and 3000 <= high_id < 4000:
        return "W"
    return "?"


def _enrich_submission_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ID" not in df.columns:
        return df
    team_names = _load_team_name_map()
    seeds = _load_seed_map()
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        mid = parse_matchup_id(str(row.ID))
        team_low = int(mid.team_low)
        team_high = int(mid.team_high)
        season = int(mid.season)
        rows.append(
            {
                "ID": row.ID,
                "Season": season,
                "Gender": _infer_gender(team_low, team_high),
                "TeamLowID": team_low,
                "TeamLowName": team_names.get(team_low, f"Team {team_low}"),
                "TeamLowSeed": seeds.get((season, team_low)),
                "TeamHighID": team_high,
                "TeamHighName": team_names.get(team_high, f"Team {team_high}"),
                "TeamHighSeed": seeds.get((season, team_high)),
                "Pred": float(row.Pred),
            }
        )
    return pd.DataFrame(rows)


def _seeded_matchups_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["TeamLowSeed"].notna() & df["TeamHighSeed"].notna()].copy()


@st.cache_data
def _brier_with_real_matchups(max_rows_per_gender: int = 6000) -> tuple[pd.DataFrame, str | None]:
    team_names = _load_team_name_map()
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for gender in ["M", "W"]:
        feat_path = Path(f"data/features/latest/{gender}_train_features.csv")
        bundle_path = Path(f"artifacts/models/{gender}_bundle.joblib")
        if not feat_path.exists() or not bundle_path.exists():
            continue

        df = pd.read_csv(feat_path)
        if df.empty or "target" not in df.columns:
            continue

        # Keep this bounded for dashboard responsiveness.
        if len(df) > max_rows_per_gender:
            df = df.sample(n=max_rows_per_gender, random_state=42).sort_index()

        bundle = load_bundle(bundle_path)

        model_cfg = {"base_models": {"elo": {"scale": 400.0}}}
        preds = predict_gender(bundle, df, model_cfg=model_cfg)
        scored = df[["Season", "TeamID_low", "TeamID_high", "target", "low_seed_num", "high_seed_num"]].copy()
        scored["pred"] = preds
        scored["squared_error"] = (scored["pred"] - scored["target"].astype(float)) ** 2
        scored["Gender"] = gender
        scored["TeamLowName"] = scored["TeamID_low"].map(lambda t: team_names.get(int(t), f"Team {int(t)}"))
        scored["TeamHighName"] = scored["TeamID_high"].map(lambda t: team_names.get(int(t), f"Team {int(t)}"))
        out.append(scored)

    warning_text = "\n".join(warnings) if warnings else None
    if not out:
        return pd.DataFrame(), warning_text
    return pd.concat(out, ignore_index=True), warning_text


@st.cache_data
def _build_submission_bracket(submission_path: str, season: int) -> dict[str, Any]:
    return run_submission_bracket_forecast(submission_path=submission_path, raw_dir="data/raw/latest", season=season)


@st.cache_data
def _build_current_bracket(season: int) -> dict[str, Any]:
    return run_current_bracket_forecast(cfg=load_all_configs(), season=season)


def _team_names_for_round(rows: list[dict[str, Any]], round_num: int) -> list[str]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if df.empty or "round_num" not in df.columns:
        return []
    sub = df[df["round_num"] == round_num].sort_values("slot_order")
    seen: set[str] = set()
    ordered: list[str] = []
    for row in sub.itertuples(index=False):
        for name in [str(row.team_low_name), str(row.team_high_name)]:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def _first_non_null(series: pd.Series) -> float | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.iloc[0])


def _power_rankings_df(
    submission_enriched: pd.DataFrame,
    gender: str,
    bracket_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    sub = submission_enriched[submission_enriched["Gender"] == gender].copy()
    if sub.empty:
        return pd.DataFrame()

    low = pd.DataFrame(
        {
            "Season": sub["Season"],
            "TeamID": sub["TeamLowID"],
            "TeamName": sub["TeamLowName"],
            "TeamSeed": sub["TeamLowSeed"],
            "WinProb": sub["Pred"],
            "OppIsSeeded": sub["TeamHighSeed"].notna(),
        }
    )
    high = pd.DataFrame(
        {
            "Season": sub["Season"],
            "TeamID": sub["TeamHighID"],
            "TeamName": sub["TeamHighName"],
            "TeamSeed": sub["TeamHighSeed"],
            "WinProb": 1.0 - sub["Pred"],
            "OppIsSeeded": sub["TeamLowSeed"].notna(),
        }
    )
    long_df = pd.concat([low, high], ignore_index=True)

    base = (
        long_df.groupby(["Season", "TeamID", "TeamName"], as_index=False)
        .agg(
            GamesInMatrix=("WinProb", "size"),
            AvgWinProbAll=("WinProb", "mean"),
            ExpectedWinsAll=("WinProb", "sum"),
            TeamSeed=("TeamSeed", _first_non_null),
        )
    )
    seeded = (
        long_df[long_df["OppIsSeeded"]]
        .groupby(["Season", "TeamID", "TeamName"], as_index=False)
        .agg(
            TournamentOpps=("WinProb", "size"),
            AvgWinProbVsTournament=("WinProb", "mean"),
            ExpectedWinsVsTournament=("WinProb", "sum"),
        )
    )
    out = base.merge(seeded, on=["Season", "TeamID", "TeamName"], how="left")
    out["TournamentOpps"] = out["TournamentOpps"].fillna(0).astype(int)
    out["TeamSeedDisplay"] = out["TeamSeed"].map(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    out["IsTournamentTeam"] = out["TeamSeed"].notna()

    final_four_ids = set((bracket_summary or {}).get("final_four_team_ids", []))
    title_game_ids = set((bracket_summary or {}).get("title_game_team_ids", []))
    champion_id = (bracket_summary or {}).get("champion_team_id")

    def bracket_pick(team_id: int) -> str:
        if champion_id is not None and int(team_id) == int(champion_id):
            return "Champion"
        if int(team_id) in title_game_ids:
            return "Finalist"
        if int(team_id) in final_four_ids:
            return "Final Four"
        if pd.notna(out.loc[out["TeamID"] == team_id, "TeamSeed"].iloc[0]):
            return "Field"
        return ""

    out["BracketPick"] = out["TeamID"].map(bracket_pick)
    out["AvgWinPctAll"] = 100.0 * out["AvgWinProbAll"]
    out["AvgWinPctVsTournament"] = 100.0 * out["AvgWinProbVsTournament"]
    out = out.sort_values(["AvgWinProbAll", "ExpectedWinsAll", "TeamName"], ascending=[False, False, True]).reset_index(drop=True)
    out["Rk"] = out.index + 1

    tourney = out[out["IsTournamentTeam"]].copy().reset_index(drop=True)
    if not tourney.empty:
        tourney["TournamentRk"] = tourney.index + 1
    out = out.merge(
        tourney[["TeamID", "TournamentRk"]],
        on="TeamID",
        how="left",
    )
    return out


def _pick_rule_label(rule: str) -> str:
    mapping = {
        "calibrated": "Calibrated",
        "raw_tiebreak": "Raw tie-break",
        "low_team_id_fallback": "Fallback",
        "actual_outcome": "Actual",
    }
    return mapping.get(str(rule), str(rule))


def _named_side_label(side: str, low_name: str, high_name: str) -> str:
    if side == "Low":
        return low_name
    if side == "High":
        return high_name
    return "Even"


def _named_agreement_df(explanation: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(explanation.get("agreement_rows", [])).copy()
    if df.empty:
        return df
    low_name = str(explanation.get("team_low_name", "Team Low"))
    high_name = str(explanation.get("team_high_name", "Team High"))
    low_prob_col = f"P({low_name} Wins)"
    high_prob_col = f"P({high_name} Wins)"
    df[low_prob_col] = df["P(Low Wins)"]
    df[high_prob_col] = 1.0 - df["P(Low Wins)"]
    df["Pick"] = df["Pick"].map(lambda value: _named_side_label(str(value), low_name=low_name, high_name=high_name))
    return df[["Model", low_prob_col, high_prob_col, "Pick"]]


def _named_matchup_df(df: pd.DataFrame, explanation: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    low_name = str(explanation.get("team_low_name", "Team Low"))
    high_name = str(explanation.get("team_high_name", "Team High"))
    return df.rename(
        columns={
            "Low Team": low_name,
            "High Team": high_name,
            "Low - High": f"{low_name} - {high_name}",
        }
    )


@st.cache_data
def _active_feature_dictionary_df(gender: str) -> pd.DataFrame:
    bundle_path = Path(f"artifacts/models/{gender}_bundle.joblib")
    if not bundle_path.exists():
        return pd.DataFrame()
    bundle = load_bundle(bundle_path)
    return build_feature_dictionary_df(bundle.feature_cols)


def _men_explainer_options(
    predicted_bracket_current: dict[str, Any],
    bracket_backtest: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        (SOURCE_2026, "2026 predicted bracket", predicted_bracket_current.get("genders", {}).get("M", {})),
        (SOURCE_2025, "2025 backtest", bracket_backtest.get("genders", {}).get("M", {})),
    ]
    for source_key, source_label, payload in sources:
        if not payload or payload.get("status") not in {None, "ok"}:
            continue
        for game in payload.get("predicted_games", []):
            rows.append(
                {
                    "source_key": source_key,
                    "source_label": source_label,
                    "round_num": int(game.get("round_num", 99)),
                    "round_label": str(game.get("round_label", "Unknown")),
                    "slot_order": int(game.get("slot_order", 9999)),
                    "slot": str(game.get("slot", "")),
                    "team_low_id": int(game.get("team_low_id")),
                    "team_high_id": int(game.get("team_high_id")),
                    "team_low_name": str(game.get("team_low_name", "")),
                    "team_high_name": str(game.get("team_high_name", "")),
                    "winner_team_name": str(game.get("winner_team_name", "")),
                    "label": (
                        f"{game.get('round_label', 'Unknown')} | "
                        f"{game.get('team_low_name', '')} vs {game.get('team_high_name', '')} | "
                        f"Pick: {game.get('winner_team_name', '')}"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["source_label", "round_num", "slot_order"]).reset_index(drop=True)


def _render_men_matchup_explainer(
    *,
    predicted_bracket_current: dict[str, Any],
    bracket_backtest: dict[str, Any],
) -> None:
    st.subheader("3) Men Matchup Explainer")
    st.caption("Select a game from the 2026 predicted bracket or the 2025 backtest to see the model breakdown.")

    options = _men_explainer_options(predicted_bracket_current=predicted_bracket_current, bracket_backtest=bracket_backtest)
    if options.empty:
        st.info("No men matchup explanations are available yet.")
        return

    c1, c2, c3 = st.columns([1.2, 1.0, 2.6])
    source_label = c1.selectbox(
        "Bracket",
        options=options["source_label"].drop_duplicates().tolist(),
        index=0,
        key="men_matchup_explainer_source",
    )
    source_subset = options[options["source_label"] == source_label].copy()
    round_label = c2.selectbox(
        "Round",
        options=source_subset["round_label"].drop_duplicates().tolist(),
        index=0,
        key="men_matchup_explainer_round",
    )
    round_subset = source_subset[source_subset["round_label"] == round_label].copy()
    matchup_idx = c3.selectbox(
        "Matchup",
        options=round_subset.index.tolist(),
        format_func=lambda idx: str(round_subset.loc[idx, "label"]),
        key="men_matchup_explainer_matchup",
    )
    selected = round_subset.loc[int(matchup_idx)]
    explanation = build_men_matchup_explanation(
        source_key=str(selected["source_key"]),
        team_low_id=int(selected["team_low_id"]),
        team_high_id=int(selected["team_high_id"]),
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predicted Winner", str(explanation.get("predicted_winner_name", "n/a")))
    m2.metric("Winner Prob (Cal)", f"{float(explanation.get('winner_prob_calibrated', 0.0)):.1%}")
    m3.metric("Winner Prob (Raw)", f"{float(explanation.get('winner_prob_raw', 0.0)):.1%}")
    m4.metric("Pick Rule", _pick_rule_label(str(explanation.get("pick_rule", "n/a"))))
    if str(selected["source_key"]) == SOURCE_2025:
        actual_label = explanation.get("actual_winner_name") or "Did not occur"
        m5.metric("Actual 2025 Result", str(actual_label))
    else:
        m5.metric("Calibration", str(explanation.get("calibration", "n/a")))

    if str(selected["source_key"]) == SOURCE_2025:
        if explanation.get("actual_matchup_occurred"):
            st.caption(
                f"This exact matchup happened in the actual 2025 tournament. "
                f"Squared error for this game was {float(explanation.get('squared_error', 0.0)):.6f}."
            )
        else:
            st.caption("This exact matchup never happened in the actual 2025 tournament, so there is no realized winner or game-level error for it.")

    st.info(str(explanation.get("summary", "")))

    agreement_df = _named_agreement_df(explanation)
    if not agreement_df.empty:
        st.caption("Model agreement")
        prob_cols = [c for c in agreement_df.columns if c.startswith("P(")]
        st.dataframe(
            agreement_df.style.format({col: "{:.3f}" for col in prob_cols}),
            width="stretch",
        )

    comparison_df = _named_matchup_df(pd.DataFrame(explanation.get("team_comparison", [])), explanation)
    if not comparison_df.empty:
        st.caption("Team snapshot comparison")
        st.dataframe(comparison_df, width="stretch")

    differences_df = _named_matchup_df(pd.DataFrame(explanation.get("top_differences", [])), explanation)
    if not differences_df.empty:
        st.caption("Largest measured differences in this matchup")
        st.dataframe(differences_df, width="stretch")


def _render_pipeline_view(
    *,
    snapshot: dict[str, Any],
    dataset_df: pd.DataFrame,
    methods_df: pd.DataFrame,
    submission_enriched: pd.DataFrame,
    submission_seeded: pd.DataFrame,
    season_df: pd.DataFrame,
    train_df: pd.DataFrame,
    explainability: dict[str, dict[str, Any]],
) -> None:
    top = st.columns(4)
    top[0].metric("Snapshot UTC", snapshot.get("generated_at_utc", "unknown"))
    top[1].metric("Datasets tracked", int(len(dataset_df)))
    top[2].metric("Submission rows", int(len(submission_enriched)))
    mean_brier = float(train_df["oof_brier_champion_calibrated"].mean()) if not train_df.empty else float("nan")
    top[3].metric("Mean OOF Brier (M/W)", f"{mean_brier:.6f}" if not np.isnan(mean_brier) else "n/a")

    fam = snapshot.get("feature_families", {})
    if isinstance(fam, dict):
        values = list(fam.values())
        if values and all(isinstance(v, dict) for v in values):
            parts = []
            for gender, families in sorted(fam.items()):
                enabled = [k for k, v in families.items() if bool(v)]
                parts.append(f"{gender}: " + (", ".join(sorted(enabled)) if enabled else "baseline"))
            st.caption("Feature families: " + " | ".join(parts))
        else:
            enabled = [k for k, v in fam.items() if bool(v)]
            st.caption(
                "Feature families: "
                + (", ".join(sorted(enabled)) if enabled else "baseline (all candidate families disabled)")
            )

    st.subheader("1) Dataset Overview")
    if dataset_df.empty:
        st.info("No dataset summaries found. Run `make observe`.")
    else:
        cols = ["name", "gender", "category", "rows", "cols", "contains", "used_for", "path"]
        st.dataframe(dataset_df[cols], width="stretch")

    st.subheader("2) Modeling Methods")
    if methods_df.empty:
        st.info("No train reports found.")
    else:
        st.dataframe(methods_df[["gender", "role", "method", "description"]], width="stretch")
        st.caption("Selected per-gender model details")
        st.dataframe(
            train_df[
                [
                    "gender",
                    "rows",
                    "feature_count",
                    "champion_raw_model",
                    "calibration",
                    "oof_brier_champion_calibrated",
                    "lift_vs_baseline",
                    "promoted",
                    "baseline_run_id",
                ]
            ],
            width="stretch",
        )

    st.subheader("3) Feature Dictionary")
    genders = [g for g in ["M", "W"] if not train_df.empty and g in set(train_df["gender"].astype(str))]
    if not genders:
        genders = ["M", "W"]
    selected_gender = st.radio(
        "Feature dictionary gender",
        options=genders,
        index=0,
        horizontal=True,
        key="feature_dictionary_gender",
    )
    feature_dict_df = _active_feature_dictionary_df(selected_gender)
    if feature_dict_df.empty:
        st.info(f"No saved model bundle found for {selected_gender}, so the active feature dictionary is unavailable.")
    else:
        enabled_families = snapshot.get("feature_families", {}).get(selected_gender, {})
        if isinstance(enabled_families, dict):
            enabled_family_names = [name for name, enabled in enabled_families.items() if bool(enabled)]
            family_text = ", ".join(sorted(enabled_family_names)) if enabled_family_names else "baseline only"
            st.caption(
                f"This dictionary reflects the active features in the saved {selected_gender} model bundle. "
                f"Enabled feature families: {family_text}."
            )
        top = st.columns(2)
        top[0].metric("Active Features", str(len(feature_dict_df)))
        top[1].metric("Unique Categories", str(int(feature_dict_df["Category"].nunique())))
        st.dataframe(feature_dict_df, width="stretch")

    st.subheader("4) Explainability")
    for gender in ["M", "W"]:
        report = explainability.get(gender, {})
        st.markdown(f"**{gender} Explainability**")
        if not report or report.get("status") != "ok":
            st.info(f"No explainability report for {gender}. Run `make observe` after `make train`.")
            continue

        top = report.get("top_features", {})
        perm_top = pd.DataFrame(top.get("permutation", []))
        coef_top = pd.DataFrame(top.get("logistic_abs_coef", []))
        shap_top = top.get("shap_mean_abs", {})
        artifacts = report.get("artifacts", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout Rows", str(report.get("rows_scored", "n/a")))
        c2.metric("Feature Count", str(report.get("feature_count", "n/a")))
        c3.metric("Champion Raw Model", str(report.get("champion_raw_model", "n/a")))

        if not perm_top.empty:
            st.caption("Permutation importance (mean Brier increase when permuted)")
            st.dataframe(perm_top, width="stretch")
        if not coef_top.empty:
            st.caption("Logistic coefficient importance (absolute coefficient)")
            st.dataframe(coef_top, width="stretch")

        shap_artifacts = artifacts.get("shap", {}) if isinstance(artifacts, dict) else {}
        for model_name in ["hgb", "xgb", "catboost"]:
            model_payload = shap_artifacts.get(model_name, {})
            if not model_payload:
                continue
            st.caption(f"SHAP ({model_name})")
            if isinstance(shap_top, dict):
                top_df = pd.DataFrame(shap_top.get(model_name, []))
                if not top_df.empty:
                    st.dataframe(top_df, width="stretch")
            for key in ["bar_plot", "beeswarm_plot"]:
                path = model_payload.get(key)
                if path and Path(path).exists():
                    st.image(str(path), caption=f"{model_name} {key.replace('_', ' ')}", use_container_width=True)

    st.subheader("5) Prediction Visualization")
    if submission_seeded.empty:
        st.info("No submission file found in latest snapshot.")
    else:
        st.caption(
            f"Showing only seeded-vs-seeded rows (tournament-like matchups): "
            f"{len(submission_seeded):,} of {len(submission_enriched):,} total rows."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        q = submission_seeded["Pred"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        c1.metric("Min Pred", f"{submission_seeded['Pred'].min():.4f}")
        c2.metric("Median Pred", f"{q[0.5]:.4f}")
        c3.metric("Max Pred", f"{submission_seeded['Pred'].max():.4f}")
        c4.metric("P05", f"{q[0.05]:.4f}")
        c5.metric("P95", f"{q[0.95]:.4f}")

        seeded_hist = pd.cut(
            submission_seeded["Pred"],
            bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            include_lowest=True,
        ).value_counts(sort=False)
        seeded_hist_df = seeded_hist.reset_index()
        seeded_hist_df.columns = ["bin", "count"]
        seeded_hist_df["bin"] = seeded_hist_df["bin"].astype(str)
        st.bar_chart(seeded_hist_df.set_index("bin"))

        st.caption(
            "To avoid repeated teams from simple sorting, examples below are random samples from confidence bands."
        )
        ui1, ui2, ui3 = st.columns(3)
        random_seed = int(ui1.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1))
        sample_n = int(ui2.slider("Rows per sample", min_value=5, max_value=50, value=15, step=1))
        confident_pct = float(ui3.slider("Confident band (top % by Pred)", min_value=0.5, max_value=10.0, value=2.0, step=0.5))

        st.caption("Confident picks sample")
        conf_cut = float(submission_seeded["Pred"].quantile(1 - confident_pct / 100.0))
        confident_pool = submission_seeded[submission_seeded["Pred"] >= conf_cut].copy()
        if confident_pool.empty:
            confident_show = submission_seeded.sort_values("Pred", ascending=False).head(sample_n)
        else:
            confident_show = confident_pool.sample(n=min(sample_n, len(confident_pool)), random_state=random_seed)
        confident_show = confident_show.sort_values("Pred", ascending=False)
        st.dataframe(
            confident_show[
                ["Season", "Gender", "TeamLowName", "TeamLowSeed", "TeamHighName", "TeamHighSeed", "Pred"]
            ],
            width="stretch",
        )

        st.caption("Uncertain picks sample (closest to 0.5)")
        uncertain = submission_seeded.assign(uncertainty=(submission_seeded["Pred"] - 0.5).abs())
        uncertain_pool_cut = float(uncertain["uncertainty"].quantile(0.1))
        uncertain_pool = uncertain[uncertain["uncertainty"] <= uncertain_pool_cut].copy()
        if uncertain_pool.empty:
            uncertain_show = uncertain.sort_values("uncertainty", ascending=True).head(sample_n)
        else:
            uncertain_show = uncertain_pool.sample(n=min(sample_n, len(uncertain_pool)), random_state=random_seed + 1)
        uncertain_show = uncertain_show.sort_values("uncertainty", ascending=True)
        st.dataframe(
            uncertain_show[
                ["Season", "Gender", "TeamLowName", "TeamLowSeed", "TeamHighName", "TeamHighSeed", "Pred"]
            ],
            width="stretch",
        )

    st.subheader("6) How Brier Score Is Calculated")
    st.code("Brier = mean((y_pred - y_true)^2)", language="text")
    st.caption(
        "Lower is better. Perfect predictions have Brier = 0.0. "
        "In this repo, `target=1` means TeamLow (lower TeamID in matchup ID) won; `target=0` means TeamHigh won. "
        "`pred` is model probability that TeamLow wins."
    )

    demo = _brier_example_df()
    st.write("Worked example (toy numbers)")
    st.dataframe(demo, width="stretch")
    st.metric("Toy Example Brier", f"{demo['squared_error'].mean():.6f}")

    st.write("Worked example (real historical matchups + team names/seeds)")
    brier_real, brier_warning = _brier_with_real_matchups()
    if brier_warning:
        st.warning(brier_warning)
    if brier_real.empty:
        st.info("Real matchup Brier sample unavailable. Run `make train` to generate model bundles and train features.")
    else:
        sample_n = min(20, len(brier_real))
        real_sample = brier_real.sort_values("squared_error", ascending=False).head(sample_n)
        st.dataframe(
            real_sample[
                [
                    "Gender",
                    "Season",
                    "TeamLowName",
                    "low_seed_num",
                    "TeamHighName",
                    "high_seed_num",
                    "target",
                    "pred",
                    "squared_error",
                ]
            ].rename(columns={"low_seed_num": "TeamLowSeed", "high_seed_num": "TeamHighSeed"}),
            width="stretch",
        )
        st.metric("Real Sample Mean Squared Error", f"{real_sample['squared_error'].mean():.6f}")

    st.write("Your actual run metrics (from training reports)")
    if train_df.empty:
        st.info("No training metrics found.")
    else:
        st.dataframe(
            train_df[
                [
                    "gender",
                    "oof_brier_blend_raw",
                    "oof_brier_meta_raw",
                    "oof_brier_champion_raw",
                    "oof_brier_champion_calibrated",
                ]
            ],
            width="stretch",
        )
    if not season_df.empty and "brier_champion_calibrated" in season_df.columns:
        piv = season_df.pivot(index="season", columns="gender", values="brier_champion_calibrated").sort_index()
        st.caption("Per-season champion calibrated Brier")
        st.line_chart(piv)
        st.dataframe(
            season_df[["gender", "season", "brier_blend_equal", "brier_blend_tuned", "brier_champion_calibrated"]]
            .sort_values(["season", "gender"]),
            width="stretch",
        )


def _render_bracket_center(
    *,
    snapshot: dict[str, Any],
    bracket_backtest: dict[str, Any],
    submission_enriched: pd.DataFrame,
) -> None:
    top = st.columns(4)
    top[0].metric("Snapshot UTC", snapshot.get("generated_at_utc", "unknown"))
    top[1].metric("Submission rows", int(len(submission_enriched)))
    season = int(submission_enriched["Season"].max()) if not submission_enriched.empty else 2026
    top[2].metric("Predicted Season", str(season))
    top[3].metric("Bracket Mode", "Most likely")

    st.subheader(f"1) {season} Predicted Bracket")
    st.caption(
        "Generated from the saved local model bundle plus official seeds and bracket slots. "
        "Winners follow calibrated probabilities, with exact 0.5 ties broken by the raw pre-calibration model score."
    )
    sub_path = snapshot.get("submission", {}).get("path")
    predicted_bracket = {"genders": {}}
    if not sub_path or not Path(sub_path).exists():
        st.info("No latest submission file found in the snapshot. Run `make submit` and `make observe`.")
    else:
        predicted_bracket = _build_current_bracket(season)
        submission_bracket = _build_submission_bracket(str(sub_path), season)
        tabs = st.tabs(["Men", "Women"])
        for gender, tab in zip(["M", "W"], tabs):
            with tab:
                gender_payload = predicted_bracket.get("genders", {}).get(gender, {})
                source_label = "local model"
                if not gender_payload or gender_payload.get("status") != "ok":
                    gender_payload = submission_bracket.get("genders", {}).get(gender, {})
                    source_label = "submission fallback"
                if not gender_payload or gender_payload.get("status") != "ok":
                    st.info(f"No predicted bracket available for {gender}.")
                    continue
                summary = gender_payload.get("summary", {})
                rankings = _power_rankings_df(submission_enriched, gender, bracket_summary=summary)
                st.caption(f"Bracket source: {source_label}.")
                st.caption(
                    "Power rankings are derived from the latest submission matrix: for each team, "
                    "we average its predicted win probability across all possible matchups."
                )
                show_all = st.checkbox(
                    "Show all teams",
                    value=False,
                    key=f"{gender.lower()}_power_rank_show_all",
                )
                display_pool = rankings if show_all else rankings[rankings["IsTournamentTeam"]].copy()
                if not show_all:
                    st.caption("Showing the tournament field by default.")
                max_rows = max(10, min(68, len(display_pool))) if not display_pool.empty else 10
                top_n = int(
                    st.slider(
                        "Top teams to show",
                        min_value=10,
                        max_value=max_rows,
                        value=min(25, max_rows),
                        step=5,
                        key=f"{gender.lower()}_power_rank_n",
                    )
                )
                if display_pool.empty:
                    st.info(f"No power rankings available for {gender}.")
                else:
                    display = display_pool[
                        [
                            "Rk",
                            "TournamentRk",
                            "TeamName",
                            "TeamSeedDisplay",
                            "BracketPick",
                            "AvgWinPctAll",
                            "ExpectedWinsAll",
                            "AvgWinPctVsTournament",
                            "ExpectedWinsVsTournament",
                        ]
                    ].head(top_n).rename(
                        columns={
                            "Rk": "Overall Rk",
                            "TournamentRk": "Tourney Rk",
                            "TeamName": "Team",
                            "TeamSeedDisplay": "Seed",
                            "BracketPick": "Predicted Finish",
                            "AvgWinPctAll": "Avg Win % vs All",
                            "ExpectedWinsAll": "Expected Wins vs All",
                            "AvgWinPctVsTournament": "Avg Win % vs Tourney",
                            "ExpectedWinsVsTournament": "Expected Wins vs Tourney",
                        }
                    )
                    st.dataframe(
                        display.style.format(
                            {
                                "Avg Win % vs All": "{:.1f}",
                                "Expected Wins vs All": "{:.1f}",
                                "Avg Win % vs Tourney": "{:.1f}",
                                "Expected Wins vs Tourney": "{:.1f}",
                            },
                            na_rep="",
                        ),
                        width="stretch",
                    )
                c1, c2, c3 = st.columns(3)
                c1.metric("Champion", str(summary.get("champion_team_name", "n/a")))
                c2.metric("Games", str(summary.get("games_total", "n/a")))
                c3.metric("Raw Tie-Breaks", str(summary.get("raw_tiebreak_games", 0)))
                final_four = summary.get("final_four_team_names", [])
                title_game = summary.get("title_game_team_names", [])
                st.caption("Final Four: " + (", ".join(final_four) if final_four else "n/a"))
                st.caption("Title Game: " + (" vs ".join(title_game) if title_game else "n/a"))
                pred_df = _bracket_table_df(gender_payload.get("predicted_games", []))
                if pred_df.empty:
                    st.info(f"No predicted bracket rows for {gender}.")
                else:
                    if "Pick Rule" in pred_df.columns:
                        pred_df["Pick Rule"] = pred_df["Pick Rule"].map(_pick_rule_label)
                    st.dataframe(pred_df, width="stretch")

    st.subheader("2) 2025 Bracket Retrospective")
    if not bracket_backtest:
        st.info("No 2025 bracket backtest artifact found. Run `make observe`.")
        return

    season = bracket_backtest.get("season", 2025)
    st.caption(
        f"Leakage-safe setup: train on seasons <= 2024; infer season {season} bracket with cutoff DayNum "
        f"{bracket_backtest.get('daynum_cutoff', 'n/a')}."
    )
    st.caption("`Final Four overlap` counts the four semifinalists. `Title game overlap` counts the two championship participants.")

    tabs = st.tabs(["Men", "Women"])
    for gender, tab in zip(["M", "W"], tabs):
        with tab:
            gender_payload = bracket_backtest.get("genders", {}).get(gender, {})
            if not gender_payload:
                st.warning(f"No backtest payload for {gender}.")
                continue
            metrics = gender_payload.get("metrics", {})
            predicted_games = gender_payload.get("predicted_games", [])
            actual_games = gender_payload.get("actual_games", [])

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            brier_overall = metrics.get("brier_overall")
            mc1.metric("Brier", f"{brier_overall:.6f}" if isinstance(brier_overall, (int, float)) else "n/a")
            champ_hit = metrics.get("champion_hit")
            mc2.metric("Champion Hit", "n/a" if champ_hit is None else ("Yes" if champ_hit else "No"))
            mc3.metric("Final Four Overlap", str(metrics.get("final_four_overlap_count", "n/a")))
            mc4.metric("Title Game Overlap", str(metrics.get("title_game_overlap_count", "n/a")))
            mc5.metric("Games Scored", str(metrics.get("games_total", "n/a")))

            st.caption("Predicted Final Four: " + ", ".join(_team_names_for_round(predicted_games, 5)))
            st.caption("Actual Final Four: " + ", ".join(_team_names_for_round(actual_games, 5)))
            st.caption("Predicted Title Game: " + " vs ".join(_team_names_for_round(predicted_games, 6)))
            st.caption("Actual Title Game: " + " vs ".join(_team_names_for_round(actual_games, 6)))

            st.caption("Predicted bracket")
            pred_df = _bracket_table_df(predicted_games)
            if pred_df.empty:
                st.info(f"No predicted bracket rows for {gender}.")
            else:
                st.dataframe(pred_df, width="stretch")

            st.caption("Actual bracket")
            act_df = _bracket_table_df(actual_games)
            if act_df.empty:
                st.info(f"No actual bracket rows for {gender}.")
            else:
                st.dataframe(act_df, width="stretch")

            brier_rows = []
            for round_label, round_brier in metrics.get("brier_by_round", {}).items():
                brier_rows.append(
                    {
                        "Round": round_label,
                        "Games": int(metrics.get("games_by_round", {}).get(round_label, 0)),
                        "Brier": float(round_brier),
                    }
                )
            if brier_rows:
                brier_df = pd.DataFrame(brier_rows).sort_values("Round", key=lambda s: s.map(_round_sort_key))
                st.caption("Per-round Brier (realized 2025 bracket games)")
                st.dataframe(brier_df, width="stretch")

    _render_men_matchup_explainer(
        predicted_bracket_current=predicted_bracket,
        bracket_backtest=bracket_backtest,
    )


def main() -> None:
    st.set_page_config(page_title="MM2026 Observability", layout="wide")
    st.title("MM2026 Modeling Overview")
    st.caption("Datasets, modeling methods, prediction behavior, and Brier score calculation.")

    default_reports = Path("artifacts") / "reports"
    reports_dir = Path(st.sidebar.text_input("Reports directory", str(default_reports)))
    st.sidebar.markdown("Run `make observe` after data/features/train/submit to refresh.")
    view = st.sidebar.radio("View", ["Pipeline observability", "Bracket center"], index=0)

    snapshot = _load_latest_snapshot(reports_dir)
    if not snapshot:
        st.error(f"No observability snapshot found in {reports_dir}.")
        st.stop()

    dataset_df = _dataset_overview_df(snapshot)
    methods_df = _model_overview_df(snapshot)
    submission_df = _load_submission_df(snapshot)
    submission_enriched = _enrich_submission_rows(submission_df)
    submission_seeded = _seeded_matchups_only(submission_enriched)
    season_df = _season_rows(snapshot)
    train_df = _train_rows(snapshot)
    bracket_backtest = _load_bracket_backtest(snapshot)
    explainability = {g: _load_explainability_payload(snapshot, g) for g in ["M", "W"]}
    if view == "Pipeline observability":
        _render_pipeline_view(
            snapshot=snapshot,
            dataset_df=dataset_df,
            methods_df=methods_df,
            submission_enriched=submission_enriched,
            submission_seeded=submission_seeded,
            season_df=season_df,
            train_df=train_df,
            explainability=explainability,
        )
    else:
        _render_bracket_center(
            snapshot=snapshot,
            bracket_backtest=bracket_backtest,
            submission_enriched=submission_enriched,
        )


if __name__ == "__main__":
    main()
