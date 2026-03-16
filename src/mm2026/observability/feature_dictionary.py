from __future__ import annotations

import re

import pandas as pd


BASE_METRICS: dict[str, dict[str, str | bool]] = {
    "games_played": {
        "label": "Games Played",
        "category": "Availability",
        "definition": "Number of games played entering the matchup.",
        "higher_better": "context",
    },
    "off_eff": {
        "label": "Offensive Efficiency",
        "category": "Offense",
        "definition": "100 * points_for / possessions.",
        "higher_better": "yes",
    },
    "def_eff": {
        "label": "Defensive Efficiency",
        "category": "Defense",
        "definition": "100 * points_against / possessions.",
        "higher_better": "no",
    },
    "net_eff": {
        "label": "Net Efficiency",
        "category": "Overall Strength",
        "definition": "Offensive efficiency minus defensive efficiency.",
        "higher_better": "yes",
    },
    "score_margin": {
        "label": "Score Margin",
        "category": "Overall Strength",
        "definition": "Points for minus points against.",
        "higher_better": "yes",
    },
    "rebound_rate": {
        "label": "Rebound Rate",
        "category": "Advanced Rates",
        "definition": "Team rebounds divided by total rebounds in the game.",
        "higher_better": "yes",
    },
    "turnover_rate": {
        "label": "Turnover Rate",
        "category": "Advanced Rates",
        "definition": "Turnovers divided by estimated possessions.",
        "higher_better": "no",
    },
    "ft_rate": {
        "label": "Free Throw Rate",
        "category": "Advanced Rates",
        "definition": "Free-throw attempts divided by field-goal attempts.",
        "higher_better": "yes",
    },
    "three_rate": {
        "label": "Three-Point Rate",
        "category": "Advanced Rates",
        "definition": "Three-point attempts divided by field-goal attempts.",
        "higher_better": "context",
    },
    "sos_off_eff_season_adj": {
        "label": "SOS-Adjusted Offensive Efficiency",
        "category": "Strength of Schedule",
        "definition": "Season offensive efficiency adjusted by opponents' defensive quality.",
        "higher_better": "yes",
    },
    "sos_def_eff_season_adj": {
        "label": "SOS-Adjusted Defensive Efficiency",
        "category": "Strength of Schedule",
        "definition": "Season defensive efficiency adjusted by opponents' offensive quality.",
        "higher_better": "no",
    },
    "sos_net_eff_season_adj": {
        "label": "SOS-Adjusted Net Efficiency",
        "category": "Strength of Schedule",
        "definition": "Season net efficiency adjusted by opponent strength.",
        "higher_better": "yes",
    },
    "opp_net_eff_season_mean": {
        "label": "Opponent Net Efficiency",
        "category": "Strength of Schedule",
        "definition": "Average season net efficiency of the opponents faced.",
        "higher_better": "context",
    },
    "elo_rating": {
        "label": "Elo Rating",
        "category": "Elo",
        "definition": "Latest Elo rating entering the matchup.",
        "higher_better": "yes",
    },
    "seed_num": {
        "label": "Seed Number",
        "category": "Seed",
        "definition": "Numeric NCAA tournament seed; lower is better.",
        "higher_better": "no",
    },
}

WINDOW_LABELS = {
    "season_mean": "Season-to-date mean",
    "roll_short": "Last 5 games",
    "roll_mid": "Last 10 games",
    "roll_long": "Last 20 games",
    "trend_short_long": "Short vs long trend",
    "vol_short": "Last 5 games volatility",
    "vol_mid": "Last 10 games volatility",
    "vol_long": "Last 20 games volatility",
    "snapshot": "Current snapshot",
    "matchup": "Derived matchup",
}


def _higher_better_note(flag: str) -> str:
    if flag == "yes":
        return "Higher values are better for the underlying metric."
    if flag == "no":
        return "Lower values are better for the underlying metric."
    return "Higher values are not inherently better or worse without context."


def _positive_meaning(metric_label: str, flag: str) -> str:
    if flag == "yes":
        return f"The low-TeamID side has a higher {metric_label.lower()} than the high-TeamID side."
    if flag == "no":
        return (
            f"The low-TeamID side has a higher {metric_label.lower()} than the high-TeamID side, "
            "which is worse on this metric."
        )
    return f"The low-TeamID side has a higher {metric_label.lower()} than the high-TeamID side."


def _parse_metric_suffix(metric: str) -> tuple[str, str]:
    for suffix in [
        "season_mean",
        "roll_short",
        "roll_mid",
        "roll_long",
        "trend_short_long",
        "vol_short",
        "vol_mid",
        "vol_long",
    ]:
        token = f"_{suffix}"
        if metric.endswith(token):
            return metric[: -len(token)], suffix
    return metric, "snapshot"


def _describe_standard_diff(feature: str) -> dict[str, str]:
    metric = feature.replace("diff_", "", 1)
    metric_base, window_key = _parse_metric_suffix(metric)
    meta = BASE_METRICS.get(metric_base)
    if meta is None:
        label = metric_base.replace("_", " ").title()
        category = "Other"
        definition = metric_base.replace("_", " ")
        higher_better = "context"
    else:
        label = str(meta["label"])
        category = str(meta["category"])
        definition = str(meta["definition"])
        higher_better = str(meta["higher_better"])

    window = WINDOW_LABELS.get(window_key, window_key.replace("_", " ").title())
    meaning = f"Difference in {window.lower()} {label.lower()}. {definition}"
    return {
        "Feature": feature,
        "Category": category,
        "Window": window,
        "Meaning": meaning,
        "Positive Values Mean": _positive_meaning(label, higher_better),
        "Metric Direction": _higher_better_note(higher_better),
    }


def describe_feature(feature: str) -> dict[str, str]:
    if feature == "diff_seed_is_low_better":
        return {
            "Feature": feature,
            "Category": "Seed",
            "Window": WINDOW_LABELS["matchup"],
            "Meaning": "Indicator for whether the low-TeamID side has the better numeric seed.",
            "Positive Values Mean": "The low-TeamID side has the better seed.",
            "Metric Direction": "Lower seed numbers are better.",
        }
    if feature == "diff_seed_gap_abs":
        return {
            "Feature": feature,
            "Category": "Seed",
            "Window": WINDOW_LABELS["matchup"],
            "Meaning": "Absolute difference between the two seed numbers.",
            "Positive Values Mean": "The matchup has a larger seed gap; this feature is non-directional.",
            "Metric Direction": "This is a matchup-shape feature, not a team-strength direction.",
        }
    if feature == "diff_seed_sum":
        return {
            "Feature": feature,
            "Category": "Seed",
            "Window": WINDOW_LABELS["matchup"],
            "Meaning": "Sum of the two seed numbers in the matchup.",
            "Positive Values Mean": "The matchup is between weaker-seeded teams overall.",
            "Metric Direction": "This is a matchup-context feature, not a directional team edge.",
        }
    if feature == "diff_elo_seed_interaction":
        return {
            "Feature": feature,
            "Category": "Elo / Seed",
            "Window": WINDOW_LABELS["matchup"],
            "Meaning": "Interaction term equal to Elo difference multiplied by seed-number difference.",
            "Positive Values Mean": "The matchup has a more positive Elo-vs-seed interaction for the low-TeamID side.",
            "Metric Direction": "This is a derived interaction feature, not a single raw basketball metric.",
        }
    if feature.startswith("diff_"):
        return _describe_standard_diff(feature)
    return {
        "Feature": feature,
        "Category": "Other",
        "Window": WINDOW_LABELS["snapshot"],
        "Meaning": feature.replace("_", " "),
        "Positive Values Mean": "Feature meaning is context-dependent.",
        "Metric Direction": "Context-dependent.",
    }


def build_feature_dictionary_df(feature_cols: list[str]) -> pd.DataFrame:
    rows = [describe_feature(feature) for feature in feature_cols]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Feature"] = pd.Categorical(out["Feature"], categories=feature_cols, ordered=True)
    out = out.sort_values("Feature").reset_index(drop=True)
    out["Feature"] = out["Feature"].astype(str)
    return out
