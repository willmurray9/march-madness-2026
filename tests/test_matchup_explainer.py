import pandas as pd

from mm2026.observability.matchup_explainer import (
    SOURCE_2026,
    _summary_text,
    _top_difference_rows,
    build_men_matchup_explanation,
)


def test_top_difference_rows_excludes_engineered_features_and_sorts_by_gap() -> None:
    feature_row = pd.DataFrame(
        [
            {
                "low_seed_num": 1,
                "high_seed_num": 4,
                "diff_seed_num": -3,
                "low_elo_rating": 1800,
                "high_elo_rating": 1700,
                "diff_elo_rating": 100,
                "low_net_eff_trend_short_long": 8.0,
                "high_net_eff_trend_short_long": 2.0,
                "diff_net_eff_trend_short_long": 6.0,
                "diff_seed_gap_abs": 3.0,
            }
        ]
    )
    feature_cols = [
        "diff_seed_num",
        "diff_elo_rating",
        "diff_net_eff_trend_short_long",
        "diff_seed_gap_abs",
    ]
    feature_stds = {
        "diff_seed_num": 2.0,
        "diff_elo_rating": 50.0,
        "diff_net_eff_trend_short_long": 1.0,
        "diff_seed_gap_abs": 1.0,
    }

    rows = _top_difference_rows(feature_row, feature_cols, feature_stds, limit=3)

    assert [row["Metric"] for row in rows] == ["Net Efficiency Trend", "Elo Rating", "Seed"]
    assert all(row["Metric"] != "Seed Gap Abs" for row in rows)


def test_summary_text_mentions_raw_tiebreak() -> None:
    text = _summary_text(
        predicted_winner_name="Duke",
        calibrated_low_win_prob=0.5,
        raw_low_win_prob=0.45,
        winner_side="high",
        pick_rule="raw_tiebreak",
        top_differences=[{"Metric": "Elo Rating"}, {"Metric": "Seed"}, {"Metric": "Net Efficiency Trend"}],
    )
    assert "raw model" in text
    assert "Elo Rating" in text


def test_build_men_matchup_explanation_for_2026_returns_expected_fields() -> None:
    payload = build_men_matchup_explanation(SOURCE_2026, 1101, 1102)

    assert payload["status"] == "ok"
    assert payload["season"] == 2026
    assert payload["team_low_id"] == 1101
    assert payload["team_high_id"] == 1102
    assert 0.0 <= float(payload["calibrated_low_win_prob"]) <= 1.0
    assert 0.0 <= float(payload["raw_low_win_prob"]) <= 1.0
    assert payload["agreement_rows"]
    assert payload["team_comparison"]
    assert payload["top_differences"]
