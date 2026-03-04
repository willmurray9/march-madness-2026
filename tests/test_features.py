import pandas as pd

from mm2026.features.build import _add_efficiency_features, _add_rolling_features


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
