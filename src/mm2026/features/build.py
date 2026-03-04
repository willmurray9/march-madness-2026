from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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


SNAPSHOT_STATS = [
    "games_played",
    "off_eff_season_mean",
    "def_eff_season_mean",
    "net_eff_season_mean",
    "score_margin_season_mean",
    "off_eff_roll_short",
    "off_eff_roll_mid",
    "off_eff_roll_long",
    "def_eff_roll_short",
    "def_eff_roll_mid",
    "def_eff_roll_long",
    "net_eff_roll_short",
    "net_eff_roll_mid",
    "net_eff_roll_long",
    "score_margin_roll_short",
    "score_margin_roll_mid",
    "score_margin_roll_long",
    "elo_rating",
    "seed_num",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["poss"] = 0.5 * (out["TeamScore"] + out["OppScore"])
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
        out[f"{col}_season_mean"] = (
            shifted.groupby([out["Season"], out["TeamID"]]).expanding().mean().reset_index(level=[0, 1], drop=True)
        )
        for name, window in windows.items():
            out[f"{col}_roll_{name}"] = (
                shifted.groupby([out["Season"], out["TeamID"]])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

    out["games_played"] = out.groupby(["Season", "TeamID"]).cumcount()
    return out


def _compute_elo_history_from_compact(df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "DayNum", "TeamID", "elo_rating"])

    games = df.sort_values(["Season", "DayNum", "WTeamID", "LTeamID"]).copy()
    ratings: dict[tuple[int, int], float] = {}
    rows: list[dict[str, float | int]] = []

    for row in games.itertuples(index=False):
        season = int(row.Season)
        day_num = int(row.DayNum)
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

        w_post = w_rating + delta
        l_post = l_rating - delta
        ratings[w_key] = w_post
        ratings[l_key] = l_post

        rows.append({"Season": season, "DayNum": day_num, "TeamID": wteam, "elo_rating": w_post})
        rows.append({"Season": season, "DayNum": day_num, "TeamID": lteam, "elo_rating": l_post})

    return pd.DataFrame(rows)


def _parse_seed_value(raw: str) -> int | None:
    if not isinstance(raw, str):
        return None
    match = re.search(r"(\d{2})", raw)
    if not match:
        return None
    return int(match.group(1))


def _prep_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    if seeds_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "seed_num"])
    out = seeds_df[["Season", "TeamID", "Seed"]].copy()
    out["seed_num"] = out["Seed"].map(_parse_seed_value)
    return out[["Season", "TeamID", "seed_num"]]


def _season_team_snapshot_asof(
    rolling_df: pd.DataFrame,
    elo_hist_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    season: int,
    day_cutoff: int,
    elo_base: float,
) -> pd.DataFrame:
    season_roll = rolling_df[(rolling_df["Season"] == season) & (rolling_df["DayNum"] < day_cutoff)].copy()
    season_elo = elo_hist_df[(elo_hist_df["Season"] == season) & (elo_hist_df["DayNum"] < day_cutoff)].copy()
    season_seeds = seeds_df[seeds_df["Season"] == season][["TeamID", "seed_num"]].copy()

    if season_roll.empty and season_seeds.empty:
        return pd.DataFrame(columns=["Season", "TeamID"] + SNAPSHOT_STATS)

    if season_roll.empty:
        stats = pd.DataFrame(columns=["Season", "TeamID"] + [c for c in SNAPSHOT_STATS if c not in {"elo_rating", "seed_num"}])
    else:
        last_rows = (
            season_roll.sort_values(["Season", "TeamID", "DayNum"])
            .groupby(["Season", "TeamID"], as_index=False)
            .tail(1)
        )
        keep_cols = [
            "Season",
            "TeamID",
            "games_played",
            "off_eff_season_mean",
            "def_eff_season_mean",
            "net_eff_season_mean",
            "score_margin_season_mean",
            "off_eff_roll_short",
            "off_eff_roll_mid",
            "off_eff_roll_long",
            "def_eff_roll_short",
            "def_eff_roll_mid",
            "def_eff_roll_long",
            "net_eff_roll_short",
            "net_eff_roll_mid",
            "net_eff_roll_long",
            "score_margin_roll_short",
            "score_margin_roll_mid",
            "score_margin_roll_long",
        ]
        stats = last_rows[keep_cols].copy()

    if season_elo.empty:
        elo = pd.DataFrame(columns=["TeamID", "elo_rating"])
    else:
        elo = (
            season_elo.sort_values(["TeamID", "DayNum"])
            .groupby("TeamID", as_index=False)
            .tail(1)[["TeamID", "elo_rating"]]
        )

    team_ids = pd.Series(
        sorted(set(stats["TeamID"].tolist()) | set(season_seeds["TeamID"].tolist()) | set(elo["TeamID"].tolist()))
    )
    out = pd.DataFrame({"Season": season, "TeamID": team_ids})
    out = out.merge(stats, on=["Season", "TeamID"], how="left")
    out = out.merge(elo, on="TeamID", how="left")
    out = out.merge(season_seeds, on="TeamID", how="left")

    out["games_played"] = out["games_played"].fillna(0.0)
    out["elo_rating"] = out["elo_rating"].fillna(elo_base)
    for col in out.columns:
        if col.startswith("off_eff") or col.startswith("def_eff") or col.startswith("net_eff") or col.startswith("score_margin"):
            out[col] = out[col].fillna(0.0)
    out["seed_num"] = out["seed_num"].fillna(99.0)
    return out


def _build_tourney_train_matchups(tourney_df: pd.DataFrame, gender: str) -> pd.DataFrame:
    if tourney_df.empty:
        return pd.DataFrame()

    out = tourney_df[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    out["TeamID_low"] = out[["WTeamID", "LTeamID"]].min(axis=1)
    out["TeamID_high"] = out[["WTeamID", "LTeamID"]].max(axis=1)
    out["target"] = (out["WTeamID"] == out["TeamID_low"]).astype(int)
    out["Gender"] = gender
    out["DayNumCutoff"] = out["DayNum"]
    return out[["Season", "TeamID_low", "TeamID_high", "Gender", "target", "DayNumCutoff"]]


def _build_stage2_matchups(sample_df: pd.DataFrame, gender: str, inference_daynum_cutoff: int) -> pd.DataFrame:
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
                "DayNumCutoff": inference_daynum_cutoff,
            }
        )

    return pd.DataFrame(rows)


def _matchup_features_for_snapshot(matchups: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if matchups.empty:
        return pd.DataFrame()

    low = snapshot.add_prefix("low_")
    high = snapshot.add_prefix("high_")

    out = (
        matchups.merge(
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
    )

    for col in SNAPSHOT_STATS:
        out[f"diff_{col}"] = out[f"low_{col}"] - out[f"high_{col}"]

    out["diff_seed_is_low_better"] = (out["low_seed_num"] < out["high_seed_num"]).astype(float)
    out["diff_seed_gap_abs"] = (out["diff_seed_num"]).abs()
    out["diff_seed_sum"] = out["low_seed_num"] + out["high_seed_num"]
    out["diff_elo_seed_interaction"] = out["diff_elo_rating"] * out["diff_seed_num"]

    feature_cols = [c for c in out.columns if c.startswith("diff_")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_matchup_features_with_cutoff(
    matchups: pd.DataFrame,
    rolling_df: pd.DataFrame,
    elo_hist_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    elo_base: float,
) -> pd.DataFrame:
    if matchups.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    unique_cutoffs = matchups[["Season", "DayNumCutoff"]].drop_duplicates().sort_values(["Season", "DayNumCutoff"])

    for row in unique_cutoffs.itertuples(index=False):
        season = int(row.Season)
        day_cutoff = int(row.DayNumCutoff)
        sub = matchups[(matchups["Season"] == season) & (matchups["DayNumCutoff"] == day_cutoff)].copy()
        snapshot = _season_team_snapshot_asof(
            rolling_df=rolling_df,
            elo_hist_df=elo_hist_df,
            seeds_df=seeds_df,
            season=season,
            day_cutoff=day_cutoff,
            elo_base=elo_base,
        )
        if snapshot.empty:
            continue
        parts.append(_matchup_features_for_snapshot(sub, snapshot))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


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
    inference_daynum_cutoff = int(feat_cfg.get("inference_daynum_cutoff", 133))

    sample_df = read_csv(curated_dir / "SampleSubmissionConfigured.csv")

    for gender in data_cfg.get("genders", ["M", "W"]):
        reg_long = read_csv(curated_dir / f"{gender}_regular_season_long.csv")
        tourney = read_csv(curated_dir / f"{gender}_tourney_compact.csv")
        seeds_raw = read_csv(curated_dir / f"{gender}_tourney_seeds.csv")
        compact = read_csv(raw_dir / f"{gender}RegularSeasonCompactResults.csv")

        if reg_long.empty:
            print(f"Skipping {gender}: no regular season data found.")
            continue

        reg_feat = _add_efficiency_features(reg_long)
        reg_roll = _add_rolling_features(reg_feat, feat_cfg["rolling_windows"])
        elo_hist = _compute_elo_history_from_compact(compact, elo_cfg)
        seeds = _prep_seeds(seeds_raw)

        train_matchups = _build_tourney_train_matchups(tourney, gender)
        train_features = _build_matchup_features_with_cutoff(
            matchups=train_matchups,
            rolling_df=reg_roll,
            elo_hist_df=elo_hist,
            seeds_df=seeds,
            elo_base=elo_cfg.base_rating,
        )
        infer_matchups = _build_stage2_matchups(sample_df, gender, inference_daynum_cutoff=inference_daynum_cutoff)
        infer_features = _build_matchup_features_with_cutoff(
            matchups=infer_matchups,
            rolling_df=reg_roll,
            elo_hist_df=elo_hist,
            seeds_df=seeds,
            elo_base=elo_cfg.base_rating,
        )

        latest_snapshot_parts: list[pd.DataFrame] = []
        for season in sorted(infer_matchups["Season"].unique()):
            latest_snapshot_parts.append(
                _season_team_snapshot_asof(
                    rolling_df=reg_roll,
                    elo_hist_df=elo_hist,
                    seeds_df=seeds,
                    season=int(season),
                    day_cutoff=inference_daynum_cutoff,
                    elo_base=elo_cfg.base_rating,
                )
            )
        snapshot = pd.concat(latest_snapshot_parts, ignore_index=True) if latest_snapshot_parts else pd.DataFrame()

        write_csv(snapshot, features_dir / f"{gender}_team_snapshot.csv")
        write_csv(train_features, features_dir / f"{gender}_train_features.csv")
        write_csv(infer_features, features_dir / f"{gender}_inference_features.csv")

    print(f"Feature files written to {features_dir}")


if __name__ == "__main__":
    run()
