from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mm2026.features.build import (
    EloConfig,
    _add_efficiency_features,
    _add_rolling_features,
    _build_matchup_features_with_cutoff,
    _compute_elo_history_from_compact,
    _prep_seeds,
)
from mm2026.models.pipeline import predict_gender, train_gender
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
    )
    day_cutoff = int(feat_cfg.get("inference_daynum_cutoff", 133))

    reg_feat = _add_efficiency_features(reg_long)
    reg_roll = _add_rolling_features(reg_feat, feat_cfg["rolling_windows"])
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
    features_dir = Path(data_cfg["features_dir"])

    train_df = read_csv(features_dir / f"{gender}_train_features.csv")
    train_df = train_df[train_df["Season"] <= 2024].copy()
    if train_df.empty:
        raise ValueError(f"{gender}: no training rows with Season <= 2024.")

    holdouts = [int(s) for s in train_cfg.get("holdout_seasons", []) if int(s) <= 2024]
    if not holdouts:
        holdouts = [2024]
    train_cfg["holdout_seasons"] = holdouts

    bundle, _report = train_gender(train_df, model_cfg=model_cfg, train_cfg=train_cfg)
    return bundle


def _seed_map_for_2025(raw_dir: Path, gender: str) -> dict[str, int]:
    seeds = read_csv(raw_dir / f"{gender}NCAATourneySeeds.csv")
    seeds = seeds[seeds["Season"] == 2025].copy()
    out: dict[str, int] = {}
    for row in seeds.itertuples(index=False):
        out[str(row.Seed)] = int(row.TeamID)
    return out


def _slot_df_for_2025(raw_dir: Path, gender: str) -> pd.DataFrame:
    slots = read_csv(raw_dir / f"{gender}NCAATourneySlots.csv")
    slots = slots[slots["Season"] == 2025].copy()
    return slots.reset_index(drop=True)


def _actual_winner_map_for_2025(raw_dir: Path, gender: str) -> dict[tuple[int, int], int]:
    games = read_csv(raw_dir / f"{gender}NCAATourneyCompactResults.csv")
    games = games[games["Season"] == 2025].copy()
    out: dict[tuple[int, int], int] = {}
    for row in games.itertuples(index=False):
        low = int(min(row.WTeamID, row.LTeamID))
        high = int(max(row.WTeamID, row.LTeamID))
        out[(low, high)] = int(row.WTeamID)
    return out


def _predict_low_win_prob(
    bundle: Any,
    model_cfg: dict[str, Any],
    snapshot: pd.DataFrame,
    season: int,
    gender: str,
    team_a: int,
    team_b: int,
) -> tuple[int, int, float]:
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

    pred = float(predict_gender(bundle, feat, model_cfg=model_cfg)[0])
    return low, high, pred


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
    bundle: Any,
    model_cfg: dict[str, Any],
    snapshot: pd.DataFrame,
    team_names: dict[int, str],
    gender: str,
    actual_winners: dict[tuple[int, int], int] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
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

            low, high, pred = _predict_low_win_prob(
                bundle=bundle,
                model_cfg=model_cfg,
                snapshot=snapshot,
                season=2025,
                gender=gender,
                team_a=team_strong,
                team_b=team_weak,
            )
            if actual_winners is None:
                winner = low if pred >= 0.5 else high
            else:
                winner = actual_winners.get((low, high))
                if winner is None:
                    raise ValueError(f"{gender}: missing actual winner for slot {slot} pair {low} vs {high}.")

            slot_winners[slot] = int(winner)
            round_num = _round_number_for_slot(slot)
            round_label = _round_label_for_slot(slot)
            y_true = 1 if int(winner) == low else 0
            se = float((pred - y_true) ** 2)
            games.append(
                {
                    "gender": gender,
                    "season": 2025,
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
                    "winner_team_id": int(winner),
                    "winner_team_name": team_names.get(int(winner), f"Team {int(winner)}"),
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

    pred_ff = set(pred_df[pred_df["round_num"] == 5]["winner_team_id"].astype(int).tolist())
    act_ff = set(actual_df[actual_df["round_num"] == 5]["winner_team_id"].astype(int).tolist())

    return {
        "games_total": int(len(actual_df)),
        "brier_overall": float(actual_df["squared_error"].mean()),
        "brier_by_round": {k: float(v) for k, v in brier_by_round.items()},
        "games_by_round": {k: int(v) for k, v in games_by_round.items()},
        "champion_hit": champion_pred is not None and champion_act is not None and champion_pred == champion_act,
        "champion_predicted_team_id": champion_pred,
        "champion_actual_team_id": champion_act,
        "final_four_overlap_count": int(len(pred_ff & act_ff)),
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
    )
    return {
        "predicted_games": predicted_games,
        "actual_games": actual_games,
        "metrics": _metrics(predicted_games=predicted_games, actual_games=actual_games),
    }


def run_bracket_backtest_2025(cfg: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "season": 2025,
        "daynum_cutoff": int(cfg["features"].get("inference_daynum_cutoff", 133)),
        "train_max_season": 2024,
        "genders": {},
    }
    for gender in cfg["data"].get("genders", ["M", "W"]):
        payload["genders"][gender] = _run_gender(cfg, gender)
    return payload
