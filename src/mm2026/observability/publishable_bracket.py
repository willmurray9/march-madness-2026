from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mm2026.backtest.bracket2025 import run_current_bracket_forecast, run_submission_bracket_forecast
from mm2026.observability.matchup_explainer import (
    SOURCE_2025,
    SOURCE_2026,
    build_men_matchup_explanation,
)
from mm2026.utils.config import load_all_configs
from mm2026.utils.ids import parse_matchup_id
from mm2026.utils.io import read_csv


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


def _load_submission_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    sub_path = snapshot.get("submission", {}).get("path")
    if not sub_path:
        return pd.DataFrame()
    path = Path(sub_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
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


def _team_name_map(cfg: dict[str, Any]) -> dict[int, str]:
    raw_dir = Path(cfg["data"]["raw_snapshot_dir"])
    out: dict[int, str] = {}
    for gender in ["M", "W"]:
        df = read_csv(raw_dir / f"{gender}Teams.csv")
        if df.empty or "TeamID" not in df.columns or "TeamName" not in df.columns:
            continue
        for row in df.itertuples(index=False):
            out[int(row.TeamID)] = str(row.TeamName)
    return out


def _seed_map(cfg: dict[str, Any]) -> dict[tuple[int, int], float]:
    curated_dir = Path(cfg["data"]["curated_dir"])
    out: dict[tuple[int, int], float] = {}
    for gender in ["M", "W"]:
        df = read_csv(curated_dir / f"{gender}_tourney_seeds.csv")
        if df.empty or "Season" not in df.columns or "TeamID" not in df.columns:
            continue
        if "seed_num" not in df.columns:
            if "Seed" not in df.columns:
                continue
            parsed = df["Seed"].astype(str).str.extract(r"(\d{2})", expand=False)
            df = df.assign(seed_num=pd.to_numeric(parsed, errors="coerce"))
        for row in df[["Season", "TeamID", "seed_num"]].dropna().itertuples(index=False):
            out[(int(row.Season), int(row.TeamID))] = float(row.seed_num)
    return out


def _infer_gender(low_id: int, high_id: int) -> str:
    if 1000 <= low_id < 2000 and 1000 <= high_id < 2000:
        return "M"
    if 3000 <= low_id < 4000 and 3000 <= high_id < 4000:
        return "W"
    return "?"


def _enrich_submission_rows(submission_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if submission_df.empty or "ID" not in submission_df.columns:
        return pd.DataFrame()
    team_names = _team_name_map(cfg)
    seeds = _seed_map(cfg)
    rows: list[dict[str, Any]] = []
    for row in submission_df.itertuples(index=False):
        matchup = parse_matchup_id(str(row.ID))
        team_low = int(matchup.team_low)
        team_high = int(matchup.team_high)
        season = int(matchup.season)
        rows.append(
            {
                "ID": str(row.ID),
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
        return out.merge(tourney[["TeamID", "TournamentRk"]], on="TeamID", how="left")
    out["TournamentRk"] = np.nan
    return out


def _matchup_key(source_key: str, team_low_id: int, team_high_id: int) -> str:
    return f"{source_key}:{int(team_low_id)}:{int(team_high_id)}"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return _to_builtin(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value


def _selected_bracket_payload(
    current_payload: dict[str, Any],
    submission_payload: dict[str, Any],
    gender: str,
) -> tuple[dict[str, Any], str]:
    current_gender = current_payload.get("genders", {}).get(gender, {})
    if current_gender and current_gender.get("status") == "ok":
        return current_gender, "local model"

    submission_gender = submission_payload.get("genders", {}).get(gender, {})
    if submission_gender and submission_gender.get("status") == "ok":
        return submission_gender, "submission fallback"

    return {}, "unavailable"


def _men_explainer_payload(
    predicted_m_payload: dict[str, Any],
    bracket_backtest: dict[str, Any],
) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    explanations: dict[str, Any] = {}
    sources = [
        (SOURCE_2026, "2026 predicted bracket", predicted_m_payload),
        (SOURCE_2025, "2025 backtest", bracket_backtest.get("genders", {}).get("M", {})),
    ]
    for source_key, source_label, payload in sources:
        if not payload or payload.get("status") not in {None, "ok"}:
            continue
        predicted_games = payload.get("predicted_games", [])
        for game in predicted_games:
            team_low_id = int(game.get("team_low_id"))
            team_high_id = int(game.get("team_high_id"))
            matchup_key = _matchup_key(source_key, team_low_id, team_high_id)
            options.append(
                {
                    "matchup_key": matchup_key,
                    "source_key": source_key,
                    "source_label": source_label,
                    "round_num": int(game.get("round_num", 99)),
                    "round_label": str(game.get("round_label", "Unknown")),
                    "slot_order": int(game.get("slot_order", 9999)),
                    "slot": str(game.get("slot", "")),
                    "team_low_id": team_low_id,
                    "team_high_id": team_high_id,
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
            explanations[matchup_key] = build_men_matchup_explanation(
                source_key=source_key,
                team_low_id=team_low_id,
                team_high_id=team_high_id,
            )
    options = sorted(
        options,
        key=lambda row: (
            str(row["source_label"]),
            int(row["round_num"]),
            int(row["slot_order"]),
            str(row["slot"]),
        ),
    )
    return {
        "options": options,
        "explanations": explanations,
    }


def build_publishable_bracket_payload(reports_dir: str | Path = "artifacts/reports") -> dict[str, Any]:
    reports_path = Path(reports_dir)
    snapshot = _load_latest_snapshot(reports_path)
    if not snapshot:
        raise FileNotFoundError(f"No observability snapshot found in {reports_path}.")

    cfg = load_all_configs()
    submission_df = _load_submission_df(snapshot)
    submission_enriched = _enrich_submission_rows(submission_df, cfg)
    if submission_enriched.empty:
        raise FileNotFoundError("No submission file was found in the latest observability snapshot.")

    season = int(submission_enriched["Season"].max())
    current_bracket = run_current_bracket_forecast(cfg=cfg, season=season)

    sub_path = snapshot.get("submission", {}).get("path")
    submission_bracket = {"genders": {}}
    if sub_path and Path(sub_path).exists():
        submission_bracket = run_submission_bracket_forecast(
            submission_path=sub_path,
            raw_dir=cfg["data"]["raw_snapshot_dir"],
            season=season,
        )

    predicted_bracket: dict[str, Any] = {}
    power_rankings: dict[str, Any] = {}
    for gender in ["M", "W"]:
        gender_payload, source_label = _selected_bracket_payload(
            current_payload=current_bracket,
            submission_payload=submission_bracket,
            gender=gender,
        )
        predicted_bracket[gender] = {
            "status": gender_payload.get("status", "unavailable") if gender_payload else "unavailable",
            "source_label": source_label,
            "predicted_games": gender_payload.get("predicted_games", []),
            "summary": gender_payload.get("summary", {}),
            "reason": gender_payload.get("reason"),
        }
        power_rankings[gender] = _power_rankings_df(
            submission_enriched,
            gender,
            bracket_summary=gender_payload.get("summary", {}),
        )

    bracket_backtest = _load_bracket_backtest(snapshot)
    men_explainer = _men_explainer_payload(predicted_bracket.get("M", {}), bracket_backtest)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "snapshot_generated_at_utc": snapshot.get("generated_at_utc"),
        "submission_rows": int(len(submission_enriched)),
        "predicted_season": season,
        "predicted_bracket": predicted_bracket,
        "power_rankings": power_rankings,
        "backtest_2025": bracket_backtest,
        "men_matchup_explainer": men_explainer,
    }
    return _to_builtin(payload)


def write_publishable_bracket_payload(
    output_path: str | Path,
    reports_dir: str | Path = "artifacts/reports",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_publishable_bracket_payload(reports_dir=reports_dir)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a publishable bracket-center payload for Streamlit Cloud.")
    parser.add_argument("--reports-dir", default="artifacts/reports", help="Observability reports directory.")
    parser.add_argument(
        "--output",
        default="deploy/bracket_center_payload.json",
        help="Output JSON path for the publishable bracket payload.",
    )
    args = parser.parse_args()

    out_path = write_publishable_bracket_payload(output_path=args.output, reports_dir=args.reports_dir)
    print(out_path)


if __name__ == "__main__":
    main()
