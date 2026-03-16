from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from mm2026.utils.config import load_all_configs, resolve_data_seasons, resolve_feature_families
from mm2026.utils.ids import parse_matchup_id
from mm2026.utils.io import ensure_dir, read_csv, write_csv


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    use_margin_multiplier: bool = False
    margin_multiplier_scale: float = 1.0
    use_dynamic_k: bool = False
    dynamic_k_base: float = 20.0
    dynamic_k_min: float = 12.0
    dynamic_k_max: float = 40.0
    dynamic_k_margin_scale: float = 6.0


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
    "rebound_rate_season_mean",
    "rebound_rate_roll_short",
    "rebound_rate_roll_mid",
    "rebound_rate_roll_long",
    "turnover_rate_season_mean",
    "turnover_rate_roll_short",
    "turnover_rate_roll_mid",
    "turnover_rate_roll_long",
    "ft_rate_season_mean",
    "ft_rate_roll_short",
    "ft_rate_roll_mid",
    "ft_rate_roll_long",
    "three_rate_season_mean",
    "three_rate_roll_short",
    "three_rate_roll_mid",
    "three_rate_roll_long",
    "off_eff_vol_short",
    "off_eff_vol_mid",
    "off_eff_vol_long",
    "def_eff_vol_short",
    "def_eff_vol_mid",
    "def_eff_vol_long",
    "net_eff_vol_short",
    "net_eff_vol_mid",
    "net_eff_vol_long",
    "score_margin_vol_short",
    "score_margin_vol_mid",
    "score_margin_vol_long",
    "off_eff_trend_short_long",
    "def_eff_trend_short_long",
    "net_eff_trend_short_long",
    "score_margin_trend_short_long",
    "rebound_rate_trend_short_long",
    "turnover_rate_trend_short_long",
    "ft_rate_trend_short_long",
    "three_rate_trend_short_long",
    "sos_off_eff_season_adj",
    "sos_def_eff_season_adj",
    "sos_net_eff_season_adj",
    "opp_net_eff_season_mean",
    "elo_rating",
    "seed_num",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0.0)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["poss"] = 0.5 * (out["TeamScore"] + out["OppScore"])
    out["points_for"] = out["TeamScore"]
    out["points_against"] = out["OppScore"]
    out["off_eff"] = 100.0 * _safe_div(out["points_for"], out["poss"])
    out["def_eff"] = 100.0 * _safe_div(out["points_against"], out["poss"])
    out["net_eff"] = out["off_eff"] - out["def_eff"]
    out["score_margin"] = out["points_for"] - out["points_against"]
    team_reb = _numeric_series(out, "TeamOR") + _numeric_series(out, "TeamDR")
    opp_reb = _numeric_series(out, "OppOR") + _numeric_series(out, "OppDR")
    out["rebound_rate"] = _safe_div(team_reb, team_reb + opp_reb)
    out["turnover_rate"] = _safe_div(_numeric_series(out, "TeamTO"), out["poss"])
    out["ft_rate"] = _safe_div(_numeric_series(out, "TeamFTA"), _numeric_series(out, "TeamFGA"))
    out["three_rate"] = _safe_div(_numeric_series(out, "TeamFGA3"), _numeric_series(out, "TeamFGA"))
    return out


def _add_sos_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    opp_lookup = out[
        ["Season", "DayNum", "TeamID", "off_eff_season_mean", "def_eff_season_mean", "net_eff_season_mean"]
    ].rename(
        columns={
            "TeamID": "OppTeamID",
            "off_eff_season_mean": "opp_off_eff_season_mean",
            "def_eff_season_mean": "opp_def_eff_season_mean",
            "net_eff_season_mean": "opp_net_eff_season_mean",
        }
    )
    out = out.merge(opp_lookup, on=["Season", "DayNum", "OppTeamID"], how="left")
    out["sos_off_eff_season_adj"] = out["off_eff_season_mean"] - out["opp_def_eff_season_mean"].fillna(0.0)
    out["sos_def_eff_season_adj"] = out["def_eff_season_mean"] - out["opp_off_eff_season_mean"].fillna(0.0)
    out["sos_net_eff_season_adj"] = out["net_eff_season_mean"] - out["opp_net_eff_season_mean"].fillna(0.0)
    out["opp_net_eff_season_mean"] = out["opp_net_eff_season_mean"].fillna(0.0)
    return out


def _add_rolling_features(
    df: pd.DataFrame,
    windows: dict[str, int],
    feature_families: dict[str, bool] | None = None,
) -> pd.DataFrame:
    families = feature_families or {}
    out = df.sort_values(["Season", "TeamID", "DayNum"]).copy()
    base_cols = ["off_eff", "def_eff", "net_eff", "score_margin", "rebound_rate", "turnover_rate", "ft_rate", "three_rate"]

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

    if bool(families.get("volatility", False)):
        vol_cols = ["off_eff", "def_eff", "net_eff", "score_margin"]
        for col in vol_cols:
            shifted = out.groupby(["Season", "TeamID"])[col].shift(1)
            for name, window in windows.items():
                out[f"{col}_vol_{name}"] = (
                    shifted.groupby([out["Season"], out["TeamID"]])
                    .rolling(window=window, min_periods=2)
                    .std(ddof=0)
                    .reset_index(level=[0, 1], drop=True)
                )

    if bool(families.get("trend", False)):
        trend_cols = ["off_eff", "def_eff", "net_eff", "score_margin", "rebound_rate", "turnover_rate", "ft_rate", "three_rate"]
        for col in trend_cols:
            short = out.get(f"{col}_roll_short")
            long = out.get(f"{col}_roll_long")
            if short is None or long is None:
                continue
            out[f"{col}_trend_short_long"] = short - long

    if bool(families.get("sos_adjusted", False)):
        out = _add_sos_features(out)

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
        margin = abs(float(getattr(row, "WScore", 0.0)) - float(getattr(row, "LScore", 0.0)))
        k_eff = cfg.k_factor
        if cfg.use_dynamic_k:
            k_eff = float(
                np.clip(
                    cfg.dynamic_k_base + margin / max(cfg.dynamic_k_margin_scale, 1e-6),
                    cfg.dynamic_k_min,
                    cfg.dynamic_k_max,
                )
            )
        delta = k_eff * (1.0 - expected_w)
        if cfg.use_margin_multiplier:
            delta *= np.log1p(margin) * cfg.margin_multiplier_scale

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
        keep_cols = ["Season", "TeamID"] + [c for c in SNAPSHOT_STATS if c not in {"elo_rating", "seed_num"}]
        for col in keep_cols:
            if col not in last_rows.columns:
                last_rows[col] = 0.0
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
        if col not in {"Season", "TeamID", "seed_num", "elo_rating"}:
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


def _build_stage2_matchups(
    sample_df: pd.DataFrame,
    gender: str,
    inference_daynum_cutoff: int,
    predict_season: int,
) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    rows = []
    for row in sample_df.itertuples(index=False):
        mid = parse_matchup_id(row.ID)
        if mid.season != predict_season:
            continue
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


def _matchup_features_for_snapshot(
    matchups: pd.DataFrame,
    snapshot: pd.DataFrame,
    feature_families: dict[str, bool] | None = None,
) -> pd.DataFrame:
    families = feature_families or {}
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
        low_col = f"low_{col}"
        high_col = f"high_{col}"
        if low_col not in out.columns or high_col not in out.columns:
            continue
        out[f"diff_{col}"] = out[low_col] - out[high_col]

    out["diff_seed_is_low_better"] = (out["low_seed_num"] < out["high_seed_num"]).astype(float)
    out["diff_seed_gap_abs"] = (out["diff_seed_num"]).abs()
    out["diff_seed_sum"] = out["low_seed_num"] + out["high_seed_num"]
    out["diff_elo_seed_interaction"] = out["diff_elo_rating"] * out["diff_seed_num"]

    if not bool(families.get("advanced_rates", False)):
        adv_prefixes = ("diff_rebound_rate", "diff_turnover_rate", "diff_ft_rate", "diff_three_rate")
        out = out[[c for c in out.columns if not c.startswith(adv_prefixes)]]
    if not bool(families.get("sos_adjusted", False)):
        out = out[[c for c in out.columns if not (c.startswith("diff_sos_") or c == "diff_opp_net_eff_season_mean")]]
    if not bool(families.get("volatility", False)):
        out = out[[c for c in out.columns if "_vol_" not in c]]
    if not bool(families.get("trend", False)):
        out = out[[c for c in out.columns if "_trend_" not in c]]

    feature_cols = [c for c in out.columns if c.startswith("diff_")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_matchup_features_with_cutoff(
    matchups: pd.DataFrame,
    rolling_df: pd.DataFrame,
    elo_hist_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    elo_base: float,
    feature_families: dict[str, bool] | None = None,
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
        parts.append(_matchup_features_for_snapshot(sub, snapshot, feature_families=feature_families))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    season_cfg = resolve_data_seasons(data_cfg)

    raw_dir = Path(data_cfg["raw_snapshot_dir"])
    curated_dir = Path(data_cfg["curated_dir"])
    features_dir = ensure_dir(data_cfg["features_dir"])

    elo_cfg = EloConfig(
        base_rating=feat_cfg["elo"]["base_rating"],
        k_factor=feat_cfg["elo"]["k_factor"],
        home_advantage=feat_cfg["elo"]["home_advantage"],
        use_margin_multiplier=bool(feat_cfg["elo"].get("use_margin_multiplier", False)),
        margin_multiplier_scale=float(feat_cfg["elo"].get("margin_multiplier_scale", 1.0)),
        use_dynamic_k=bool(feat_cfg["elo"].get("use_dynamic_k", False)),
        dynamic_k_base=float(feat_cfg["elo"].get("dynamic_k_base", feat_cfg["elo"]["k_factor"])),
        dynamic_k_min=float(feat_cfg["elo"].get("dynamic_k_min", 12.0)),
        dynamic_k_max=float(feat_cfg["elo"].get("dynamic_k_max", 40.0)),
        dynamic_k_margin_scale=float(feat_cfg["elo"].get("dynamic_k_margin_scale", 6.0)),
    )
    inference_daynum_cutoff = int(feat_cfg.get("inference_daynum_cutoff", 133))

    sample_df = read_csv(curated_dir / "SampleSubmissionConfigured.csv")

    for gender in data_cfg.get("genders", ["M", "W"]):
        feature_families = resolve_feature_families(feat_cfg, gender)
        gender_elo_cfg = EloConfig(**vars(elo_cfg))
        if not feature_families["elo_upgrades"]:
            gender_elo_cfg.use_margin_multiplier = False
            gender_elo_cfg.use_dynamic_k = False

        reg_long = read_csv(curated_dir / f"{gender}_regular_season_long.csv")
        tourney = read_csv(curated_dir / f"{gender}_tourney_compact.csv")
        seeds_raw = read_csv(curated_dir / f"{gender}_tourney_seeds.csv")
        compact = read_csv(raw_dir / f"{gender}RegularSeasonCompactResults.csv")

        if reg_long.empty:
            print(f"Skipping {gender}: no regular season data found.")
            continue

        reg_feat = _add_efficiency_features(reg_long)
        reg_roll = _add_rolling_features(reg_feat, feat_cfg["rolling_windows"], feature_families=feature_families)
        elo_hist = _compute_elo_history_from_compact(compact, gender_elo_cfg)
        seeds = _prep_seeds(seeds_raw)

        train_matchups = _build_tourney_train_matchups(tourney, gender)
        train_features = _build_matchup_features_with_cutoff(
            matchups=train_matchups,
            rolling_df=reg_roll,
            elo_hist_df=elo_hist,
            seeds_df=seeds,
            elo_base=gender_elo_cfg.base_rating,
            feature_families=feature_families,
        )
        if not train_features.empty and "Season" in train_features.columns:
            train_features = train_features[
                train_features["Season"].between(
                    season_cfg["min_train_season"],
                    season_cfg["max_train_season"],
                )
            ].copy()

        infer_matchups = _build_stage2_matchups(
            sample_df,
            gender,
            inference_daynum_cutoff=inference_daynum_cutoff,
            predict_season=season_cfg["predict_season"],
        )
        infer_features = _build_matchup_features_with_cutoff(
            matchups=infer_matchups,
            rolling_df=reg_roll,
            elo_hist_df=elo_hist,
            seeds_df=seeds,
            elo_base=gender_elo_cfg.base_rating,
            feature_families=feature_families,
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
                    elo_base=gender_elo_cfg.base_rating,
                )
            )
        snapshot = pd.concat(latest_snapshot_parts, ignore_index=True) if latest_snapshot_parts else pd.DataFrame()

        write_csv(snapshot, features_dir / f"{gender}_team_snapshot.csv")
        write_csv(train_features, features_dir / f"{gender}_train_features.csv")
        write_csv(infer_features, features_dir / f"{gender}_inference_features.csv")

    print(f"Feature files written to {features_dir}")


if __name__ == "__main__":
    run()
