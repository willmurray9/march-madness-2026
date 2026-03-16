from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from mm2026.features.build import (
    EloConfig,
    _add_efficiency_features,
    _add_rolling_features,
    _build_matchup_features_with_cutoff,
    _compute_elo_history_from_compact,
    _prep_seeds,
)
from mm2026.models.pipeline import load_bundle, predict_gender_with_raw, train_gender
from mm2026.utils.config import resolve_data_seasons, resolve_feature_families
from mm2026.utils.ids import parse_matchup_id
from mm2026.utils.io import read_csv

ROUND_LABELS = {
    0: "Play-In",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def _round_number_for_slot(slot: str) -> int:
    if slot.startswith("R") and len(slot) >= 2 and slot[1].isdigit():
        return int(slot[1])
    return 0


def _round_label_for_slot(slot: str) -> str:
    return ROUND_LABELS.get(_round_number_for_slot(slot), "Unknown")


def _prediction_payload(
    low: int,
    high: int,
    pred: float,
    raw_pred: float | None = None,
) -> dict[str, Any]:
    return {
        "team_low_id": int(low),
        "team_high_id": int(high),
        "pred_team_low_win": float(pred),
        "pred_team_low_win_raw": None if raw_pred is None else float(raw_pred),
    }


def _normalize_prediction_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        low = int(result["team_low_id"])
        high = int(result["team_high_id"])
        pred = float(result["pred_team_low_win"])
        raw_pred = result.get("pred_team_low_win_raw")
        return _prediction_payload(low=low, high=high, pred=pred, raw_pred=raw_pred)
    if isinstance(result, tuple) and len(result) == 3:
        low, high, pred = result
        return _prediction_payload(low=int(low), high=int(high), pred=float(pred))
    raise TypeError("Prediction result must be a dict or a 3-tuple of (low, high, pred).")


def _choose_predicted_winner(low: int, high: int, pred: float, raw_pred: float | None) -> tuple[int, str]:
    if pred > 0.5:
        return int(low), "calibrated"
    if pred < 0.5:
        return int(high), "calibrated"
    if raw_pred is not None:
        if raw_pred > 0.5:
            return int(low), "raw_tiebreak"
        if raw_pred < 0.5:
            return int(high), "raw_tiebreak"
    return int(low), "low_team_id_fallback"


def _load_team_name_map(raw_dir: Path, gender: str) -> dict[int, str]:
    path = raw_dir / f"{gender}Teams.csv"
    if not path.exists():
        return {}
    df = read_csv(path)
    if df.empty or "TeamID" not in df.columns or "TeamName" not in df.columns:
        return {}
    out: dict[int, str] = {}
    for row in df.itertuples(index=False):
        out[int(row.TeamID)] = str(row.TeamName)
    return out


def _seed_map_for_season(raw_dir: Path, gender: str, season: int) -> dict[str, int]:
    seeds = read_csv(raw_dir / f"{gender}NCAATourneySeeds.csv")
    seeds = seeds[seeds["Season"] == season].copy()
    out: dict[str, int] = {}
    for row in seeds.itertuples(index=False):
        out[str(row.Seed)] = int(row.TeamID)
    return out


def _slot_df_for_season(raw_dir: Path, gender: str, season: int) -> pd.DataFrame:
    slots = read_csv(raw_dir / f"{gender}NCAATourneySlots.csv")
    slots = slots[slots["Season"] == season].copy()
    return slots.reset_index(drop=True)


def _actual_winner_map_for_season(raw_dir: Path, gender: str, season: int) -> dict[tuple[int, int], int]:
    games = read_csv(raw_dir / f"{gender}NCAATourneyCompactResults.csv")
    games = games[games["Season"] == season].copy()
    out: dict[tuple[int, int], int] = {}
    for row in games.itertuples(index=False):
        low = int(min(row.WTeamID, row.LTeamID))
        high = int(max(row.WTeamID, row.LTeamID))
        out[(low, high)] = int(row.WTeamID)
    return out


def _load_snapshot_for_2025(cfg: dict[str, Any], gender: str) -> pd.DataFrame:
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    raw_dir = Path(data_cfg["raw_snapshot_dir"])
    curated_dir = Path(data_cfg["curated_dir"])

    reg_long = read_csv(curated_dir / f"{gender}_regular_season_long.csv")
    compact = read_csv(raw_dir / f"{gender}RegularSeasonCompactResults.csv")
    seeds_raw = read_csv(curated_dir / f"{gender}_tourney_seeds.csv")

    elo_cfg = EloConfig(
        base_rating=float(feat_cfg["elo"]["base_rating"]),
        k_factor=float(feat_cfg["elo"]["k_factor"]),
        home_advantage=float(feat_cfg["elo"]["home_advantage"]),
        use_margin_multiplier=bool(feat_cfg["elo"].get("use_margin_multiplier", False)),
        margin_multiplier_scale=float(feat_cfg["elo"].get("margin_multiplier_scale", 1.0)),
        use_dynamic_k=bool(feat_cfg["elo"].get("use_dynamic_k", False)),
        dynamic_k_base=float(feat_cfg["elo"].get("dynamic_k_base", feat_cfg["elo"]["k_factor"])),
        dynamic_k_min=float(feat_cfg["elo"].get("dynamic_k_min", 12.0)),
        dynamic_k_max=float(feat_cfg["elo"].get("dynamic_k_max", 40.0)),
        dynamic_k_margin_scale=float(feat_cfg["elo"].get("dynamic_k_margin_scale", 6.0)),
    )
    feature_families = resolve_feature_families(feat_cfg, gender)
    if not feature_families["elo_upgrades"]:
        elo_cfg.use_margin_multiplier = False
        elo_cfg.use_dynamic_k = False
    day_cutoff = int(feat_cfg.get("inference_daynum_cutoff", 133))

    reg_feat = _add_efficiency_features(reg_long)
    reg_roll = _add_rolling_features(reg_feat, feat_cfg["rolling_windows"], feature_families=feature_families)
    elo_hist = _compute_elo_history_from_compact(compact, elo_cfg)
    seeds = _prep_seeds(seeds_raw)

    teams_2025 = seeds[seeds["Season"] == 2025]["TeamID"].dropna().astype(int).unique().tolist()
    matchups = pd.DataFrame(
        {
            "Season": [2025 for _ in teams_2025],
            "TeamID_low": teams_2025,
            "TeamID_high": teams_2025,
            "Gender": [gender for _ in teams_2025],
            "DayNumCutoff": [day_cutoff for _ in teams_2025],
        }
    )
    snap = _build_matchup_features_with_cutoff(
        matchups=matchups,
        rolling_df=reg_roll,
        elo_hist_df=elo_hist,
        seeds_df=seeds,
        elo_base=elo_cfg.base_rating,
        feature_families=feature_families,
    )
    if snap.empty:
        return pd.DataFrame()

    low_cols = [c for c in snap.columns if c.startswith("low_") and c not in {"low_Season", "low_TeamID"}]
    keep = ["low_Season", "low_TeamID"] + low_cols
    out = snap[keep].copy().drop_duplicates(subset=["low_Season", "low_TeamID"])
    rename = {c: c.replace("low_", "") for c in out.columns}
    out = out.rename(columns=rename)
    out["Season"] = out["Season"].astype(int)
    out["TeamID"] = out["TeamID"].astype(int)
    return out


def _train_bundle_for_2025(cfg: dict[str, Any], gender: str) -> Any:
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    train_cfg = deepcopy(cfg["train"])
    season_cfg = resolve_data_seasons(data_cfg)
    features_dir = Path(data_cfg["features_dir"])

    train_df = read_csv(features_dir / f"{gender}_train_features.csv")
    train_df = train_df[
        train_df["Season"].between(
            season_cfg["min_train_season"],
            min(season_cfg["max_train_season"], 2024),
        )
    ].copy()
    if train_df.empty:
        raise ValueError(f"{gender}: no training rows with Season <= 2024.")

    holdouts = [int(s) for s in train_cfg.get("holdout_seasons", []) if int(s) <= 2024]
    if not holdouts:
        holdouts = [2024]
    train_cfg["holdout_seasons"] = holdouts

    bundle, _report = train_gender(train_df, model_cfg=model_cfg, train_cfg=train_cfg)
    return bundle


def _seed_map_for_2025(raw_dir: Path, gender: str) -> dict[str, int]:
    return _seed_map_for_season(raw_dir, gender, 2025)


def _slot_df_for_2025(raw_dir: Path, gender: str) -> pd.DataFrame:
    return _slot_df_for_season(raw_dir, gender, 2025)


def _actual_winner_map_for_2025(raw_dir: Path, gender: str) -> dict[tuple[int, int], int]:
    return _actual_winner_map_for_season(raw_dir, gender, 2025)


def _predict_low_win_prob(
    bundle: Any,
    model_cfg: dict[str, Any],
    snapshot: pd.DataFrame,
    season: int,
    gender: str,
    team_a: int,
    team_b: int,
) -> dict[str, Any]:
    low = int(min(team_a, team_b))
    high = int(max(team_a, team_b))
    matchup = pd.DataFrame(
        [
            {
                "Season": int(season),
                "TeamID_low": low,
                "TeamID_high": high,
                "Gender": gender,
            }
        ]
    )
    low_row = snapshot[snapshot["TeamID"] == low].add_prefix("low_")
    high_row = snapshot[snapshot["TeamID"] == high].add_prefix("high_")
    if low_row.empty or high_row.empty:
        raise ValueError(f"Missing snapshot row for matchup {season}_{low}_{high}.")
    feat = (
        matchup.merge(
            low_row,
            left_on=["Season", "TeamID_low"],
            right_on=["low_Season", "low_TeamID"],
            how="left",
        )
        .merge(
            high_row,
            left_on=["Season", "TeamID_high"],
            right_on=["high_Season", "high_TeamID"],
            how="left",
        )
        .copy()
    )
    for col in [c for c in snapshot.columns if c not in {"Season", "TeamID"}]:
        feat[f"diff_{col}"] = feat[f"low_{col}"] - feat[f"high_{col}"]
    feat["diff_seed_is_low_better"] = (feat["low_seed_num"] < feat["high_seed_num"]).astype(float)
    feat["diff_seed_gap_abs"] = (feat["diff_seed_num"]).abs()
    feat["diff_seed_sum"] = feat["low_seed_num"] + feat["high_seed_num"]
    feat["diff_elo_seed_interaction"] = feat["diff_elo_rating"] * feat["diff_seed_num"]
    diff_cols = [c for c in feat.columns if c.startswith("diff_")]
    feat[diff_cols] = feat[diff_cols].fillna(0.0)

    raw_pred, pred = predict_gender_with_raw(bundle, feat, model_cfg=model_cfg)
    return _prediction_payload(low=low, high=high, pred=float(pred[0]), raw_pred=float(raw_pred[0]))


def _submission_prob_map(
    submission_df: pd.DataFrame,
    season: int,
    gender: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if submission_df.empty:
        return out
    for row in submission_df.itertuples(index=False):
        mid = parse_matchup_id(str(row.ID))
        if int(mid.season) != int(season):
            continue
        if gender == "M" and not (1000 <= mid.team_low < 2000 and 1000 <= mid.team_high < 2000):
            continue
        if gender == "W" and not (3000 <= mid.team_low < 4000 and 3000 <= mid.team_high < 4000):
            continue
        out[(int(mid.team_low), int(mid.team_high))] = _prediction_payload(
            low=int(mid.team_low),
            high=int(mid.team_high),
            pred=float(row.Pred),
        )
    return out


def _local_model_prob_map(
    cfg: dict[str, Any],
    season: int,
    gender: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    features_dir = Path(data_cfg["features_dir"])
    models_dir = Path(data_cfg["artifacts_dir"]) / "models"

    feat_path = features_dir / f"{gender}_inference_features.csv"
    bundle_path = models_dir / f"{gender}_bundle.joblib"
    if not feat_path.exists() or not bundle_path.exists():
        return {}

    inference_df = read_csv(feat_path)
    if inference_df.empty:
        return {}
    if "Season" in inference_df.columns:
        inference_df = inference_df[inference_df["Season"] == int(season)].copy()
    if inference_df.empty:
        return {}

    bundle = load_bundle(bundle_path)
    raw_probs, pred_probs = predict_gender_with_raw(bundle, inference_df, model_cfg=model_cfg)

    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row, raw_pred, pred in zip(inference_df.itertuples(index=False), raw_probs, pred_probs):
        low = int(row.TeamID_low)
        high = int(row.TeamID_high)
        out[(low, high)] = _prediction_payload(low=low, high=high, pred=float(pred), raw_pred=float(raw_pred))
    return out


def _predict_low_win_prob_from_prob_map(
    prob_map: dict[tuple[int, int], dict[str, Any]],
    team_a: int,
    team_b: int,
) -> dict[str, Any]:
    low = int(min(team_a, team_b))
    high = int(max(team_a, team_b))
    pred = prob_map.get((low, high))
    if pred is None:
        raise ValueError(f"Missing submission probability for matchup {low} vs {high}.")
    return _normalize_prediction_result(pred)


def _resolve_token(token: str, seed_to_team: dict[str, int], slot_winners: dict[str, int]) -> int | None:
    if token in slot_winners:
        return slot_winners[token]
    if token in seed_to_team:
        return seed_to_team[token]
    return None


def _simulate_bracket(
    *,
    slots: pd.DataFrame,
    seed_to_team: dict[str, int],
    bundle: Any | None,
    model_cfg: dict[str, Any] | None,
    snapshot: pd.DataFrame | None,
    team_names: dict[int, str],
    gender: str,
    actual_winners: dict[tuple[int, int], int] | None,
    season: int = 2025,
    predict_matchup: Callable[[int, int], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if predict_matchup is None:
        if bundle is None or model_cfg is None or snapshot is None:
            raise ValueError("bundle, model_cfg, and snapshot are required when no custom predict_matchup callback is supplied.")

        def predict_matchup(team_a: int, team_b: int) -> dict[str, Any]:
            return _predict_low_win_prob(
                bundle=bundle,
                model_cfg=model_cfg,
                snapshot=snapshot,
                season=season,
                gender=gender,
                team_a=team_a,
                team_b=team_b,
            )

    pending = slots[["Slot", "StrongSeed", "WeakSeed"]].copy()
    slot_winners: dict[str, int] = {}
    order_map = {str(slot): idx for idx, slot in enumerate(slots["Slot"].tolist())}
    games: list[dict[str, Any]] = []

    while not pending.empty:
        progressed = False
        next_rows = []
        for row in pending.itertuples(index=False):
            slot = str(row.Slot)
            strong_ref = str(row.StrongSeed)
            weak_ref = str(row.WeakSeed)
            team_strong = _resolve_token(strong_ref, seed_to_team=seed_to_team, slot_winners=slot_winners)
            team_weak = _resolve_token(weak_ref, seed_to_team=seed_to_team, slot_winners=slot_winners)
            if team_strong is None or team_weak is None:
                next_rows.append({"Slot": slot, "StrongSeed": strong_ref, "WeakSeed": weak_ref})
                continue

            prediction = _normalize_prediction_result(predict_matchup(team_strong, team_weak))
            low = int(prediction["team_low_id"])
            high = int(prediction["team_high_id"])
            pred = float(prediction["pred_team_low_win"])
            raw_pred = prediction.get("pred_team_low_win_raw")
            if actual_winners is None:
                winner, winner_decision_rule = _choose_predicted_winner(
                    low=low,
                    high=high,
                    pred=pred,
                    raw_pred=None if raw_pred is None else float(raw_pred),
                )
            else:
                winner = actual_winners.get((low, high))
                if winner is None:
                    raise ValueError(f"{gender}: missing actual winner for slot {slot} pair {low} vs {high}.")
                winner_decision_rule = "actual_outcome"

            slot_winners[slot] = int(winner)
            round_num = _round_number_for_slot(slot)
            round_label = _round_label_for_slot(slot)
            y_true = 1 if int(winner) == low else 0
            se = float((pred - y_true) ** 2)
            games.append(
                {
                    "gender": gender,
                    "season": int(season),
                    "slot": slot,
                    "slot_order": int(order_map.get(slot, 9999)),
                    "round_num": int(round_num),
                    "round_label": round_label,
                    "strong_ref": strong_ref,
                    "weak_ref": weak_ref,
                    "team_low_id": low,
                    "team_low_name": team_names.get(low, f"Team {low}"),
                    "team_low_seed": seed_to_team and next((k for k, v in seed_to_team.items() if v == low), None),
                    "team_high_id": high,
                    "team_high_name": team_names.get(high, f"Team {high}"),
                    "team_high_seed": seed_to_team and next((k for k, v in seed_to_team.items() if v == high), None),
                    "pred_team_low_win": pred,
                    "pred_team_low_win_raw": raw_pred,
                    "winner_team_id": int(winner),
                    "winner_team_name": team_names.get(int(winner), f"Team {int(winner)}"),
                    "winner_decision_rule": winner_decision_rule,
                    "y_true_low_win": int(y_true),
                    "squared_error": se,
                }
            )
            progressed = True

        if not progressed:
            unresolved = pending["Slot"].astype(str).tolist()
            raise ValueError(f"{gender}: could not resolve all slots. unresolved={unresolved[:10]}")
        pending = pd.DataFrame(next_rows)

    games = sorted(games, key=lambda x: (int(x["round_num"]), int(x["slot_order"])))
    return games, slot_winners


def _participant_ids_for_round(rows_df: pd.DataFrame, round_num: int) -> set[int]:
    if rows_df.empty:
        return set()
    sub = rows_df[rows_df["round_num"] == round_num]
    if sub.empty:
        return set()
    if "team_low_id" not in sub.columns or "team_high_id" not in sub.columns:
        if "winner_team_id" not in sub.columns:
            return set()
        return set(sub["winner_team_id"].astype(int).tolist())
    ids = set(sub["team_low_id"].astype(int).tolist())
    ids |= set(sub["team_high_id"].astype(int).tolist())
    return ids


def _ordered_participants_for_round(rows_df: pd.DataFrame, round_num: int) -> list[tuple[int, str]]:
    if rows_df.empty:
        return []
    sub = rows_df[rows_df["round_num"] == round_num].sort_values("slot_order")
    seen: set[int] = set()
    ordered: list[tuple[int, str]] = []
    for row in sub.itertuples(index=False):
        for team_id, name in [
            (int(row.team_low_id), str(row.team_low_name)),
            (int(row.team_high_id), str(row.team_high_name)),
        ]:
            if team_id in seen:
                continue
            seen.add(team_id)
            ordered.append((team_id, name))
    return ordered


def _predicted_bracket_summary(predicted_games: list[dict[str, Any]]) -> dict[str, Any]:
    pred_df = pd.DataFrame(predicted_games)
    if pred_df.empty:
        return {}
    final_four = _ordered_participants_for_round(pred_df, 5)
    title_game = _ordered_participants_for_round(pred_df, 6)
    champ = pred_df[pred_df["slot"] == "R6CH"]["winner_team_id"]
    champ_name = pred_df[pred_df["slot"] == "R6CH"]["winner_team_name"]
    return {
        "games_total": int(len(pred_df)),
        "final_four_team_ids": [team_id for team_id, _ in final_four],
        "final_four_team_names": [name for _, name in final_four],
        "title_game_team_ids": [team_id for team_id, _ in title_game],
        "title_game_team_names": [name for _, name in title_game],
        "champion_team_id": int(champ.iloc[0]) if not champ.empty else None,
        "champion_team_name": str(champ_name.iloc[0]) if not champ_name.empty else None,
        "raw_tiebreak_games": int((pred_df.get("winner_decision_rule") == "raw_tiebreak").sum()),
        "low_team_id_fallback_games": int((pred_df.get("winner_decision_rule") == "low_team_id_fallback").sum()),
    }


def _metrics(
    predicted_games: list[dict[str, Any]],
    actual_games: list[dict[str, Any]],
) -> dict[str, Any]:
    actual_df = pd.DataFrame(actual_games)
    if actual_df.empty:
        return {
            "games_total": 0,
            "brier_overall": None,
            "brier_by_round": {},
            "games_by_round": {},
            "champion_hit": None,
            "champion_predicted_team_id": None,
            "champion_actual_team_id": None,
            "final_four_overlap_count": None,
            "title_game_overlap_count": None,
        }

    brier_by_round = (
        actual_df.groupby("round_label")["squared_error"]
        .mean()
        .sort_index()
        .to_dict()
    )
    games_by_round = (
        actual_df.groupby("round_label")["slot"]
        .count()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    pred_df = pd.DataFrame(predicted_games)
    pred_champ = pred_df[pred_df["slot"] == "R6CH"]["winner_team_id"]
    act_champ = actual_df[actual_df["slot"] == "R6CH"]["winner_team_id"]
    champion_pred = int(pred_champ.iloc[0]) if not pred_champ.empty else None
    champion_act = int(act_champ.iloc[0]) if not act_champ.empty else None

    pred_ff = _participant_ids_for_round(pred_df, 5)
    act_ff = _participant_ids_for_round(actual_df, 5)
    pred_title = _participant_ids_for_round(pred_df, 6)
    act_title = _participant_ids_for_round(actual_df, 6)

    return {
        "games_total": int(len(actual_df)),
        "brier_overall": float(actual_df["squared_error"].mean()),
        "brier_by_round": {k: float(v) for k, v in brier_by_round.items()},
        "games_by_round": {k: int(v) for k, v in games_by_round.items()},
        "champion_hit": champion_pred is not None and champion_act is not None and champion_pred == champion_act,
        "champion_predicted_team_id": champion_pred,
        "champion_actual_team_id": champion_act,
        "final_four_overlap_count": int(len(pred_ff & act_ff)),
        "title_game_overlap_count": int(len(pred_title & act_title)),
    }


def _run_gender(cfg: dict[str, Any], gender: str) -> dict[str, Any]:
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    raw_dir = Path(data_cfg["raw_snapshot_dir"])

    slots = _slot_df_for_2025(raw_dir, gender)
    seed_to_team = _seed_map_for_2025(raw_dir, gender)
    actual_winner_map = _actual_winner_map_for_2025(raw_dir, gender)
    team_names = _load_team_name_map(raw_dir, gender)
    snapshot = _load_snapshot_for_2025(cfg, gender)
    bundle = _train_bundle_for_2025(cfg, gender)

    predicted_games, _pred_winners = _simulate_bracket(
        slots=slots,
        seed_to_team=seed_to_team,
        bundle=bundle,
        model_cfg=model_cfg,
        snapshot=snapshot,
        team_names=team_names,
        gender=gender,
        actual_winners=None,
        season=2025,
    )
    actual_games, _act_winners = _simulate_bracket(
        slots=slots,
        seed_to_team=seed_to_team,
        bundle=bundle,
        model_cfg=model_cfg,
        snapshot=snapshot,
        team_names=team_names,
        gender=gender,
        actual_winners=actual_winner_map,
        season=2025,
    )
    return {
        "predicted_games": predicted_games,
        "actual_games": actual_games,
        "metrics": _metrics(predicted_games=predicted_games, actual_games=actual_games),
    }


def run_bracket_backtest_2025(cfg: dict[str, Any]) -> dict[str, Any]:
    season_cfg = resolve_data_seasons(cfg["data"])
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "season": 2025,
        "daynum_cutoff": int(cfg["features"].get("inference_daynum_cutoff", 133)),
        "train_max_season": min(season_cfg["max_train_season"], 2024),
        "genders": {},
    }
    for gender in cfg["data"].get("genders", ["M", "W"]):
        payload["genders"][gender] = _run_gender(cfg, gender)
    return payload


def run_submission_bracket_forecast(
    submission_path: str | Path,
    raw_dir: str | Path,
    season: int,
    genders: list[str] | None = None,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    submission_df = read_csv(submission_path)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "season": int(season),
        "submission_path": str(submission_path),
        "genders": {},
    }
    for gender in genders or ["M", "W"]:
        slots = _slot_df_for_season(raw_path, gender, int(season))
        seed_to_team = _seed_map_for_season(raw_path, gender, int(season))
        team_names = _load_team_name_map(raw_path, gender)
        prob_map = _submission_prob_map(submission_df, int(season), gender)
        if slots.empty or not seed_to_team or not prob_map:
            payload["genders"][gender] = {
                "status": "skipped",
                "reason": "Missing slots, seeds, or submission probabilities for requested season.",
            }
            continue

        predicted_games, _pred_winners = _simulate_bracket(
            slots=slots,
            seed_to_team=seed_to_team,
            bundle=None,
            model_cfg=None,
            snapshot=None,
            team_names=team_names,
            gender=gender,
            actual_winners=None,
            season=int(season),
            predict_matchup=lambda team_a, team_b, pm=prob_map: _predict_low_win_prob_from_prob_map(pm, team_a, team_b),
        )
        payload["genders"][gender] = {
            "status": "ok",
            "predicted_games": predicted_games,
            "summary": _predicted_bracket_summary(predicted_games),
        }
    return payload


def run_current_bracket_forecast(
    cfg: dict[str, Any],
    season: int,
    genders: list[str] | None = None,
) -> dict[str, Any]:
    raw_dir = Path(cfg["data"]["raw_snapshot_dir"])
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "season": int(season),
        "genders": {},
    }
    for gender in genders or cfg["data"].get("genders", ["M", "W"]):
        slots = _slot_df_for_season(raw_dir, gender, int(season))
        seed_to_team = _seed_map_for_season(raw_dir, gender, int(season))
        team_names = _load_team_name_map(raw_dir, gender)
        prob_map = _local_model_prob_map(cfg, int(season), gender)
        if slots.empty or not seed_to_team or not prob_map:
            payload["genders"][gender] = {
                "status": "skipped",
                "reason": "Missing slots, seeds, inference features, or model bundle for requested season.",
            }
            continue

        predicted_games, _pred_winners = _simulate_bracket(
            slots=slots,
            seed_to_team=seed_to_team,
            bundle=None,
            model_cfg=None,
            snapshot=None,
            team_names=team_names,
            gender=gender,
            actual_winners=None,
            season=int(season),
            predict_matchup=lambda team_a, team_b, pm=prob_map: _predict_low_win_prob_from_prob_map(pm, team_a, team_b),
        )
        payload["genders"][gender] = {
            "status": "ok",
            "predicted_games": predicted_games,
            "summary": _predicted_bracket_summary(predicted_games),
        }
    return payload
