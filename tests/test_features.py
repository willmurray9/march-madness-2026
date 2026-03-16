import pandas as pd

from mm2026.features.build import (
    _add_efficiency_features,
    _add_rolling_features,
    _build_stage2_matchups,
    _matchup_features_for_snapshot,
    _build_tourney_train_matchups,
    _prep_seeds,
)


def test_rolling_features_use_shifted_history() -> None:
    df = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 30],
            "TeamID": [1111, 1111, 1111],
            "OppTeamID": [2222, 3333, 4444],
            "TeamScore": [70, 80, 90],
            "OppScore": [60, 70, 85],
            "Loc": ["N", "N", "N"],
            "NumOT": [0, 0, 0],
            "IsWin": [1, 1, 1],
            "Gender": ["M", "M", "M"],
        }
    )

    feat = _add_efficiency_features(df)
    roll = _add_rolling_features(feat, {"short": 2, "mid": 2, "long": 2})

    # First game should not use current game as history.
    assert pd.isna(roll.loc[0, "off_eff_roll_short"])
    # Second game should have history from game 1 only.
    assert abs(roll.loc[1, "off_eff_roll_short"] - roll.loc[0, "off_eff"]) < 1e-9


def test_seed_parsing_and_day_cutoff_columns() -> None:
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [1101, 1102],
            "Seed": ["W01", "X16b"],
        }
    )
    parsed = _prep_seeds(seeds)
    assert parsed["seed_num"].tolist() == [1, 16]

    tourney = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [136],
            "WTeamID": [1101],
            "LTeamID": [1102],
        }
    )
    matchups = _build_tourney_train_matchups(tourney, "M")
    assert "DayNumCutoff" in matchups.columns
    assert int(matchups.loc[0, "DayNumCutoff"]) == 136


def test_advanced_rates_from_box_score_columns() -> None:
    df = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [10],
            "TeamID": [1111],
            "OppTeamID": [2222],
            "TeamScore": [80],
            "OppScore": [70],
            "TeamFGA": [60],
            "TeamFGA3": [20],
            "TeamFTA": [15],
            "TeamTO": [10],
            "TeamOR": [8],
            "TeamDR": [20],
            "OppOR": [6],
            "OppDR": [18],
            "Loc": ["N"],
            "NumOT": [0],
            "IsWin": [1],
            "Gender": ["M"],
        }
    )
    feat = _add_efficiency_features(df)
    assert abs(float(feat.loc[0, "rebound_rate"]) - (28.0 / 52.0)) < 1e-9
    assert abs(float(feat.loc[0, "ft_rate"]) - (15.0 / 60.0)) < 1e-9
    assert abs(float(feat.loc[0, "three_rate"]) - (20.0 / 60.0)) < 1e-9
    assert abs(float(feat.loc[0, "turnover_rate"]) - (10.0 / 75.0)) < 1e-9


def test_matchup_feature_family_filters_drop_disabled_columns() -> None:
    snapshot = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [1101, 1102],
            "games_played": [10, 11],
            "off_eff_season_mean": [110.0, 105.0],
            "def_eff_season_mean": [95.0, 100.0],
            "net_eff_season_mean": [15.0, 5.0],
            "score_margin_season_mean": [8.0, 3.0],
            "off_eff_roll_short": [111.0, 104.0],
            "off_eff_roll_mid": [110.0, 104.0],
            "off_eff_roll_long": [109.0, 103.0],
            "def_eff_roll_short": [94.0, 101.0],
            "def_eff_roll_mid": [95.0, 101.0],
            "def_eff_roll_long": [96.0, 102.0],
            "net_eff_roll_short": [17.0, 3.0],
            "net_eff_roll_mid": [15.0, 3.0],
            "net_eff_roll_long": [13.0, 1.0],
            "score_margin_roll_short": [9.0, 2.0],
            "score_margin_roll_mid": [8.0, 2.0],
            "score_margin_roll_long": [7.0, 1.0],
            "rebound_rate_season_mean": [0.55, 0.50],
            "rebound_rate_roll_short": [0.56, 0.49],
            "rebound_rate_roll_mid": [0.55, 0.49],
            "rebound_rate_roll_long": [0.54, 0.48],
            "turnover_rate_season_mean": [0.12, 0.14],
            "turnover_rate_roll_short": [0.11, 0.15],
            "turnover_rate_roll_mid": [0.12, 0.14],
            "turnover_rate_roll_long": [0.13, 0.14],
            "ft_rate_season_mean": [0.25, 0.22],
            "ft_rate_roll_short": [0.26, 0.21],
            "ft_rate_roll_mid": [0.25, 0.22],
            "ft_rate_roll_long": [0.24, 0.21],
            "three_rate_season_mean": [0.38, 0.35],
            "three_rate_roll_short": [0.39, 0.34],
            "three_rate_roll_mid": [0.38, 0.35],
            "three_rate_roll_long": [0.37, 0.34],
            "net_eff_vol_short": [2.0, 3.0],
            "off_eff_trend_short_long": [2.0, 1.0],
            "sos_net_eff_season_adj": [5.0, 1.0],
            "opp_net_eff_season_mean": [10.0, 8.0],
            "elo_rating": [1530.0, 1480.0],
            "seed_num": [1.0, 16.0],
        }
    )
    matchups = pd.DataFrame(
        {"Season": [2025], "TeamID_low": [1101], "TeamID_high": [1102], "Gender": ["M"], "DayNumCutoff": [136]}
    )
    out = _matchup_features_for_snapshot(
        matchups,
        snapshot,
        feature_families={"advanced_rates": False, "sos_adjusted": False, "volatility": False, "trend": False},
    )
    cols = set(out.columns)
    assert "diff_rebound_rate_season_mean" not in cols
    assert "diff_sos_net_eff_season_adj" not in cols
    assert "diff_opp_net_eff_season_mean" not in cols
    assert all("_vol_" not in c for c in cols)
    assert all("_trend_" not in c for c in cols)


def test_build_stage2_matchups_filters_to_predict_season_and_gender() -> None:
    sample = pd.DataFrame(
        {
            "ID": [
                "2025_1101_1102",
                "2026_1101_1102",
                "2026_3101_3102",
            ]
        }
    )
    men = _build_stage2_matchups(sample, "M", inference_daynum_cutoff=133, predict_season=2026)
    women = _build_stage2_matchups(sample, "W", inference_daynum_cutoff=133, predict_season=2026)

    assert men["ID"].tolist() == ["2026_1101_1102"]
    assert women["ID"].tolist() == ["2026_3101_3102"]
    assert men["DayNumCutoff"].tolist() == [133]
