from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mm2026.utils.config import load_all_configs
from mm2026.utils.ids import parse_matchup_id
from mm2026.utils.io import ensure_dir, read_csv, write_csv


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["poss"] = 0.5 * ((out["TeamScore"] + out["OppScore"]))
    out["points_for"] = out["TeamScore"]
    out["points_against"] = out["OppScore"]
    out["off_eff"] = 100.0 * _safe_div(out["points_for"], out["poss"])
    out["def_eff"] = 100.0 * _safe_div(out["points_against"], out["poss"])
    out["net_eff"] = out["off_eff"] - out["def_eff"]
    out["score_margin"] = out["points_for"] - out["points_against"]
    return out


def _add_rolling_features(df: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    out = df.sort_values(["Season", "TeamID", "DayNum"]).copy()
    base_cols = ["off_eff", "def_eff", "net_eff", "score_margin"]

    for col in base_cols:
        shifted = out.groupby(["Season", "TeamID"])[col].shift(1)
        out[f"{col}_season_mean"] = shifted.groupby([out["Season"], out["TeamID"]]).expanding().mean().reset_index(level=[0, 1], drop=True)
        for name, window in windows.items():
            out[f"{col}_roll_{name}"] = shifted.groupby([out["Season"], out["TeamID"]]).rolling(window=window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    out["games_played"] = out.groupby(["Season", "TeamID"]).cumcount()
    return out


def _compute_elo_from_compact(df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "elo_rating"])

    games = df.sort_values(["Season", "DayNum", "WTeamID", "LTeamID"]).copy()
    ratings: dict[tuple[int, int], float] = {}

    for row in games.itertuples(index=False):
        season = int(row.Season)
        wteam = int(row.WTeamID)
        lteam = int(row.LTeamID)
        wloc = getattr(row, "WLoc", "N")

        w_key = (season, wteam)
        l_key = (season, lteam)
        w_rating = ratings.get(w_key, cfg.base_rating)
        l_rating = ratings.get(l_key, cfg.base_rating)

        home_adj = cfg.home_advantage if wloc == "H" else -cfg.home_advantage if wloc == "A" else 0.0
        expected_w = 1.0 / (1.0 + 10.0 ** (-(w_rating + home_adj - l_rating) / 400.0))
        delta = cfg.k_factor * (1.0 - expected_w)

        ratings[w_key] = w_rating + delta
        ratings[l_key] = l_rating - delta

    out = pd.DataFrame(
        [{"Season": s, "TeamID": t, "elo_rating": r} for (s, t), r in ratings.items()]
    )
    return out


def _season_team_snapshot(rolling_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    if rolling_df.empty:
        return pd.DataFrame()

    last_rows = (
        rolling_df.sort_values(["Season", "TeamID", "DayNum"]) 
        .groupby(["Season", "TeamID"], as_index=False)
        .tail(1)
    )

    keep_cols = [
        "Season", "TeamID", "Gender", "games_played", "off_eff_season_mean", "def_eff_season_mean",
        "net_eff_season_mean", "score_margin_season_mean", "off_eff_roll_short", "off_eff_roll_mid",
        "off_eff_roll_long", "def_eff_roll_short", "def_eff_roll_mid", "def_eff_roll_long",
        "net_eff_roll_short", "net_eff_roll_mid", "net_eff_roll_long", "score_margin_roll_short",
        "score_margin_roll_mid", "score_margin_roll_long",
    ]

    snapshot = last_rows[keep_cols].copy()
    snapshot = snapshot.merge(elo_df, on=["Season", "TeamID"], how="left")
    snapshot["elo_rating"] = snapshot["elo_rating"].fillna(1500.0)
    return snapshot


def _build_tourney_train_matchups(tourney_df: pd.DataFrame, gender: str) -> pd.DataFrame:
    if tourney_df.empty:
        return pd.DataFrame()

    out = tourney_df[["Season", "WTeamID", "LTeamID"]].copy()
    out["TeamID_low"] = out[["WTeamID", "LTeamID"]].min(axis=1)
    out["TeamID_high"] = out[["WTeamID", "LTeamID"]].max(axis=1)
    out["target"] = (out["WTeamID"] == out["TeamID_low"]).astype(int)
    out["Gender"] = gender
    return out[["Season", "TeamID_low", "TeamID_high", "Gender", "target"]]


def _build_stage2_matchups(sample_df: pd.DataFrame, gender: str) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    rows = []
    for row in sample_df.itertuples(index=False):
        mid = parse_matchup_id(row.ID)
        if gender == "M" and not (1000 <= mid.team_low < 2000 and 1000 <= mid.team_high < 2000):
            continue
        if gender == "W" and not (3000 <= mid.team_low < 4000 and 3000 <= mid.team_high < 4000):
            continue
        rows.append(
            {
                "ID": row.ID,
                "Season": mid.season,
                "TeamID_low": mid.team_low,
                "TeamID_high": mid.team_high,
                "Gender": gender,
            }
        )

    return pd.DataFrame(rows)


def _matchup_features(matchups: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if matchups.empty:
        return pd.DataFrame()

    low = snapshot.add_prefix("low_")
    high = snapshot.add_prefix("high_")

    out = matchups.merge(
        low,
        left_on=["Season", "TeamID_low"],
        right_on=["low_Season", "low_TeamID"],
        how="left",
    ).merge(
        high,
        left_on=["Season", "TeamID_high"],
        right_on=["high_Season", "high_TeamID"],
        how="left",
    )

    base_stats = [
        "games_played", "off_eff_season_mean", "def_eff_season_mean", "net_eff_season_mean",
        "score_margin_season_mean", "off_eff_roll_short", "off_eff_roll_mid", "off_eff_roll_long",
        "def_eff_roll_short", "def_eff_roll_mid", "def_eff_roll_long", "net_eff_roll_short",
        "net_eff_roll_mid", "net_eff_roll_long", "score_margin_roll_short", "score_margin_roll_mid",
        "score_margin_roll_long", "elo_rating",
    ]

    for col in base_stats:
        out[f"diff_{col}"] = out[f"low_{col}"] - out[f"high_{col}"]

    feature_cols = [c for c in out.columns if c.startswith("diff_")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    raw_dir = Path(data_cfg["raw_snapshot_dir"])
    curated_dir = Path(data_cfg["curated_dir"])
    features_dir = ensure_dir(data_cfg["features_dir"])

    elo_cfg = EloConfig(
        base_rating=feat_cfg["elo"]["base_rating"],
        k_factor=feat_cfg["elo"]["k_factor"],
        home_advantage=feat_cfg["elo"]["home_advantage"],
    )

    sample_df = read_csv(curated_dir / "SampleSubmissionStage2.csv")

    for gender in data_cfg.get("genders", ["M", "W"]):
        reg_long = read_csv(curated_dir / f"{gender}_regular_season_long.csv")
        tourney = read_csv(curated_dir / f"{gender}_tourney_compact.csv")
        compact = read_csv(raw_dir / f"{gender}RegularSeasonCompactResults.csv")

        if reg_long.empty:
            print(f"Skipping {gender}: no regular season data found.")
            continue

        reg_feat = _add_efficiency_features(reg_long)
        reg_roll = _add_rolling_features(reg_feat, feat_cfg["rolling_windows"])
        elo_df = _compute_elo_from_compact(compact, elo_cfg)
        snapshot = _season_team_snapshot(reg_roll, elo_df)

        train_matchups = _build_tourney_train_matchups(tourney, gender)
        train_features = _matchup_features(train_matchups, snapshot)
        infer_matchups = _build_stage2_matchups(sample_df, gender)
        infer_features = _matchup_features(infer_matchups, snapshot)

        write_csv(snapshot, features_dir / f"{gender}_team_snapshot.csv")
        write_csv(train_features, features_dir / f"{gender}_train_features.csv")
        write_csv(infer_features, features_dir / f"{gender}_inference_features.csv")

    print(f"Feature files written to {features_dir}")


if __name__ == "__main__":
    run()
