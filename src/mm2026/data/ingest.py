from __future__ import annotations

from pathlib import Path

import pandas as pd

from mm2026.utils.config import load_all_configs
from mm2026.utils.io import ensure_dir, read_csv, write_csv


def _build_long_games(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    winners = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["WTeamID"],
            "OppTeamID": df["LTeamID"],
            "TeamScore": df["WScore"],
            "OppScore": df["LScore"],
            "Loc": df.get("WLoc", "N"),
            "NumOT": df.get("NumOT", 0),
            "IsWin": 1,
            "Gender": gender,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["LTeamID"],
            "OppTeamID": df["WTeamID"],
            "TeamScore": df["LScore"],
            "OppScore": df["WScore"],
            "Loc": df.get("WLoc", "N").map({"H": "A", "A": "H", "N": "N"}),
            "NumOT": df.get("NumOT", 0),
            "IsWin": 0,
            "Gender": gender,
        }
    )
    long_df = pd.concat([winners, losers], ignore_index=True)
    long_df = long_df.sort_values(["Season", "DayNum", "TeamID"]).reset_index(drop=True)
    return long_df


def run() -> None:
    cfg = load_all_configs()
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_snapshot_dir"])
    curated_dir = ensure_dir(data_cfg["curated_dir"])

    for gender in data_cfg.get("genders", ["M", "W"]):
        reg_detailed_path = raw_dir / f"{gender}RegularSeasonDetailedResults.csv"
        reg_compact_path = raw_dir / f"{gender}RegularSeasonCompactResults.csv"
        tourney_compact_path = raw_dir / f"{gender}NCAATourneyCompactResults.csv"

        reg_df = read_csv(reg_detailed_path)
        if reg_df.empty:
            reg_df = read_csv(reg_compact_path)

        long_df = _build_long_games(reg_df, gender)
        write_csv(long_df, curated_dir / f"{gender}_regular_season_long.csv")

        tourney_df = read_csv(tourney_compact_path)
        write_csv(tourney_df, curated_dir / f"{gender}_tourney_compact.csv")

        seeds_df = read_csv(raw_dir / f"{gender}NCAATourneySeeds.csv")
        write_csv(seeds_df, curated_dir / f"{gender}_tourney_seeds.csv")

        seasons_df = read_csv(raw_dir / f"{gender}Seasons.csv")
        write_csv(seasons_df, curated_dir / f"{gender}_seasons.csv")

    sample_stage1 = read_csv(raw_dir / "SampleSubmissionStage1.csv")
    if not sample_stage1.empty:
        write_csv(sample_stage1, curated_dir / "SampleSubmissionStage1.csv")

    sample_stage2 = read_csv(raw_dir / "SampleSubmissionStage2.csv")
    if not sample_stage2.empty:
        write_csv(sample_stage2, curated_dir / "SampleSubmissionStage2.csv")

    configured_sample = read_csv(data_cfg["sample_submission_file"])
    if not configured_sample.empty:
        write_csv(configured_sample, curated_dir / "SampleSubmissionConfigured.csv")
    print(f"Curated data written to {curated_dir}")


if __name__ == "__main__":
    run()
