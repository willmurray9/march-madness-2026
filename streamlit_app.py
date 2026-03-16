from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_PAYLOAD_PATH = Path("deploy/bracket_center_payload.json")


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


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


def _pick_rule_label(rule: str) -> str:
    mapping = {
        "calibrated": "Calibrated",
        "raw_tiebreak": "Raw tie-break",
        "low_team_id_fallback": "Fallback",
        "actual_outcome": "Actual",
    }
    return mapping.get(str(rule), str(rule))


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
    return out.rename(
        columns={
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
    )


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


def _render_predicted_bracket(payload: dict[str, Any]) -> None:
    season = int(payload.get("predicted_season", 2026))
    predicted = payload.get("predicted_bracket", {})
    power_rankings = payload.get("power_rankings", {})

    st.subheader(f"1) {season} Predicted Bracket")
    st.caption(
        "This page is a frozen bracket snapshot exported from the local modeling environment. "
        "Winners follow calibrated probabilities, with exact 0.5 ties broken by the raw pre-calibration model score."
    )

    tabs = st.tabs(["Men", "Women"])
    for gender, tab in zip(["M", "W"], tabs):
        with tab:
            gender_payload = predicted.get(gender, {})
            if gender_payload.get("status") != "ok":
                st.info(f"No predicted bracket available for {gender}.")
                continue

            rankings = pd.DataFrame(power_rankings.get(gender, []))
            st.caption(f"Bracket source: {gender_payload.get('source_label', 'unknown')}.")
            st.caption(
                "Power rankings are derived from the latest submission matrix: for each team, "
                "we average its predicted win probability across all possible matchups."
            )
            top_n = int(
                st.slider(
                    "Top teams to show",
                    min_value=10,
                    max_value=50,
                    value=25,
                    step=5,
                    key=f"{gender.lower()}_deploy_power_rank_n",
                )
            )
            if rankings.empty:
                st.info(f"No power rankings available for {gender}.")
            else:
                display = rankings[
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

            summary = gender_payload.get("summary", {})
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


def _render_backtest(payload: dict[str, Any]) -> None:
    backtest = payload.get("backtest_2025", {})
    if not backtest:
        st.subheader("2) 2025 Bracket Retrospective")
        st.info("No 2025 bracket backtest is available in the published payload.")
        return

    season = backtest.get("season", 2025)
    st.subheader("2) 2025 Bracket Retrospective")
    st.caption(
        f"Leakage-safe setup: train on seasons <= 2024; infer season {season} bracket with cutoff DayNum "
        f"{backtest.get('daynum_cutoff', 'n/a')}."
    )
    st.caption("`Final Four overlap` counts the four semifinalists. `Title game overlap` counts the two championship participants.")

    tabs = st.tabs(["Men", "Women"])
    for gender, tab in zip(["M", "W"], tabs):
        with tab:
            gender_payload = backtest.get("genders", {}).get(gender, {})
            if not gender_payload:
                st.info(f"No 2025 backtest payload available for {gender}.")
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
                if "Pick Rule" in pred_df.columns:
                    pred_df["Pick Rule"] = pred_df["Pick Rule"].map(_pick_rule_label)
                st.dataframe(pred_df, width="stretch")

            st.caption("Actual bracket")
            act_df = _bracket_table_df(actual_games)
            if act_df.empty:
                st.info(f"No actual bracket rows for {gender}.")
            else:
                if "Pick Rule" in act_df.columns:
                    act_df["Pick Rule"] = act_df["Pick Rule"].map(_pick_rule_label)
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


def _render_matchup_explainer(payload: dict[str, Any]) -> None:
    explainer = payload.get("men_matchup_explainer", {})
    options = pd.DataFrame(explainer.get("options", []))
    explanations = explainer.get("explanations", {})

    st.subheader("3) Men Matchup Explainer")
    if options.empty or not explanations:
        st.info("No matchup explanations are available in the published payload.")
        return

    c1, c2, c3 = st.columns([1.2, 1.0, 2.6])
    source_label = c1.selectbox(
        "Bracket",
        options=options["source_label"].drop_duplicates().tolist(),
        index=0,
        key="deploy_men_matchup_source",
    )
    source_subset = options[options["source_label"] == source_label].copy()
    round_label = c2.selectbox(
        "Round",
        options=source_subset["round_label"].drop_duplicates().tolist(),
        index=0,
        key="deploy_men_matchup_round",
    )
    round_subset = source_subset[source_subset["round_label"] == round_label].copy()
    matchup_idx = c3.selectbox(
        "Matchup",
        options=round_subset.index.tolist(),
        format_func=lambda idx: str(round_subset.loc[idx, "label"]),
        key="deploy_men_matchup_matchup",
    )
    selected = round_subset.loc[int(matchup_idx)]
    explanation = explanations.get(str(selected["matchup_key"]), {})
    if not explanation:
        st.info("No explanation payload is available for that game.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predicted Winner", str(explanation.get("predicted_winner_name", "n/a")))
    m2.metric("Winner Prob (Cal)", f"{float(explanation.get('winner_prob_calibrated', 0.0)):.1%}")
    m3.metric("Winner Prob (Raw)", f"{float(explanation.get('winner_prob_raw', 0.0)):.1%}")
    m4.metric("Pick Rule", _pick_rule_label(str(explanation.get("pick_rule", "n/a"))))
    if str(selected["source_key"]) == "2025_backtest":
        actual_label = explanation.get("actual_winner_name") or "Did not occur"
        m5.metric("Actual 2025 Result", str(actual_label))
    else:
        m5.metric("Calibration", str(explanation.get("calibration", "n/a")))

    if str(selected["source_key"]) == "2025_backtest":
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
        st.dataframe(agreement_df.style.format({col: "{:.3f}" for col in prob_cols}), width="stretch")

    comparison_df = _named_matchup_df(pd.DataFrame(explanation.get("team_comparison", [])), explanation)
    if not comparison_df.empty:
        st.caption("Team snapshot comparison")
        st.dataframe(comparison_df, width="stretch")

    differences_df = _named_matchup_df(pd.DataFrame(explanation.get("top_differences", [])), explanation)
    if not differences_df.empty:
        st.caption("Largest measured differences in this matchup")
        st.dataframe(differences_df, width="stretch")


def main() -> None:
    st.set_page_config(page_title="MM2026 Bracket Center", layout="wide")
    st.title("MM2026 Bracket Center")
    st.caption("Shareable bracket page exported from the local modeling environment.")

    payload_path = Path(os.environ.get("MM2026_BRACKET_PAYLOAD", str(DEFAULT_PAYLOAD_PATH)))
    payload = _load_payload(payload_path)
    if not payload:
        st.error(
            f"No published bracket payload was found at {payload_path}. "
            "Run `make publish-bracket` locally and commit the generated JSON before deploying."
        )
        st.stop()

    top = st.columns(4)
    top[0].metric("Payload UTC", str(payload.get("generated_at_utc", "unknown")))
    top[1].metric("Snapshot UTC", str(payload.get("snapshot_generated_at_utc", "unknown")))
    top[2].metric("Submission Rows", str(payload.get("submission_rows", "n/a")))
    top[3].metric("Predicted Season", str(payload.get("predicted_season", "n/a")))

    st.caption(
        "This deployed app is intentionally frozen. To refresh it, rerun `make publish-bracket`, "
        "commit `deploy/bracket_center_payload.json`, and push to GitHub."
    )

    _render_predicted_bracket(payload)
    _render_backtest(payload)
    _render_matchup_explainer(payload)


if __name__ == "__main__":
    main()
