from mm2026.observability.feature_dictionary import build_feature_dictionary_df, describe_feature


def test_describe_standard_feature_contains_window_and_direction() -> None:
    row = describe_feature("diff_def_eff_roll_short")

    assert row["Category"] == "Defense"
    assert row["Window"] == "Last 5 games"
    assert "defensive efficiency" in row["Meaning"].lower()
    assert "worse" in row["Positive Values Mean"].lower()


def test_describe_derived_seed_feature_is_non_directional() -> None:
    row = describe_feature("diff_seed_gap_abs")

    assert row["Category"] == "Seed"
    assert row["Window"] == "Derived matchup"
    assert "non-directional" in row["Positive Values Mean"].lower()


def test_build_feature_dictionary_preserves_feature_order() -> None:
    df = build_feature_dictionary_df(
        [
            "diff_elo_rating",
            "diff_seed_is_low_better",
            "diff_off_eff_trend_short_long",
        ]
    )

    assert df["Feature"].tolist() == [
        "diff_elo_rating",
        "diff_seed_is_low_better",
        "diff_off_eff_trend_short_long",
    ]
