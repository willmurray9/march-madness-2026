import pandas as pd
from pathlib import Path

from mm2026.backtest.bracket2025 import (
    _actual_winner_map_for_2025,
    _metrics,
    _seed_map_for_2025,
    _simulate_bracket,
    _slot_df_for_2025,
)


def test_slot_counts_for_2025_men_and_women() -> None:
    raw_dir = Path("data/raw/latest")
    m_slots = _slot_df_for_2025(raw_dir, "M")
    w_slots = _slot_df_for_2025(raw_dir, "W")
    assert len(m_slots) == 67
    assert len(w_slots) == 67
    assert "R6CH" in set(m_slots["Slot"].astype(str).tolist())
    assert "R6CH" in set(w_slots["Slot"].astype(str).tolist())


def test_playin_seed_suffixes_present_in_2025_seeds() -> None:
    raw_dir = Path("data/raw/latest")
    m_seed_map = _seed_map_for_2025(raw_dir, "M")
    w_seed_map = _seed_map_for_2025(raw_dir, "W")
    assert any(seed.endswith("a") for seed in m_seed_map)
    assert any(seed.endswith("b") for seed in m_seed_map)
    assert any(seed.endswith("a") for seed in w_seed_map)
    assert any(seed.endswith("b") for seed in w_seed_map)


def test_simulate_bracket_resolves_dependency_slots(monkeypatch) -> None:
    slots = pd.DataFrame(
        {
            "Slot": ["W16", "R1W1", "R6CH"],
            "StrongSeed": ["W16a", "W01", "R1W1"],
            "WeakSeed": ["W16b", "W16", "R1X1"],
        }
    )
    seed_to_team = {"W01": 1101, "W16a": 1116, "W16b": 1117, "X01": 1201, "X16": 1216}
    snapshot = pd.DataFrame({"Season": [2025], "TeamID": [1101]})
    team_names = {1101: "A", 1116: "B", 1117: "C", 1201: "D", 1216: "E"}

    def fake_predict(**kwargs):
        low = min(kwargs["team_a"], kwargs["team_b"])
        high = max(kwargs["team_a"], kwargs["team_b"])
        if (low, high) == (1116, 1117):
            return low, high, 0.55
        if (low, high) == (1101, 1116):
            return low, high, 0.80
        return low, high, 0.40

    monkeypatch.setattr("mm2026.backtest.bracket2025._predict_low_win_prob", fake_predict)

    slots = pd.concat(
        [
            slots,
            pd.DataFrame({"Slot": ["R1X1"], "StrongSeed": ["X01"], "WeakSeed": ["X16"]}),
        ],
        ignore_index=True,
    )
    games, winners = _simulate_bracket(
        slots=slots,
        seed_to_team=seed_to_team,
        bundle=object(),
        model_cfg={"base_models": {"elo": {"scale": 400.0}}},
        snapshot=snapshot,
        team_names=team_names,
        gender="M",
        actual_winners=None,
    )
    assert len(games) == 4
    assert winners["W16"] == 1116
    assert winners["R1W1"] == 1101
    assert winners["R6CH"] == 1216
    assert all(0.0 <= float(g["pred_team_low_win"]) <= 1.0 for g in games)


def test_simulate_bracket_uses_raw_tiebreak_toward_high_team() -> None:
    slots = pd.DataFrame({"Slot": ["R1W1"], "StrongSeed": ["W01"], "WeakSeed": ["W16"]})
    seed_to_team = {"W01": 1101, "W16": 1116}
    team_names = {1101: "A", 1116: "B"}

    games, winners = _simulate_bracket(
        slots=slots,
        seed_to_team=seed_to_team,
        bundle=None,
        model_cfg=None,
        snapshot=None,
        team_names=team_names,
        gender="M",
        actual_winners=None,
        predict_matchup=lambda _a, _b: {
            "team_low_id": 1101,
            "team_high_id": 1116,
            "pred_team_low_win": 0.5,
            "pred_team_low_win_raw": 0.49,
        },
    )

    assert winners["R1W1"] == 1116
    assert games[0]["winner_decision_rule"] == "raw_tiebreak"
    assert float(games[0]["pred_team_low_win"]) == 0.5
    assert float(games[0]["pred_team_low_win_raw"]) == 0.49


def test_simulate_bracket_uses_raw_tiebreak_toward_low_team() -> None:
    slots = pd.DataFrame({"Slot": ["R1W1"], "StrongSeed": ["W01"], "WeakSeed": ["W16"]})
    seed_to_team = {"W01": 1101, "W16": 1116}
    team_names = {1101: "A", 1116: "B"}

    games, winners = _simulate_bracket(
        slots=slots,
        seed_to_team=seed_to_team,
        bundle=None,
        model_cfg=None,
        snapshot=None,
        team_names=team_names,
        gender="M",
        actual_winners=None,
        predict_matchup=lambda _a, _b: {
            "team_low_id": 1101,
            "team_high_id": 1116,
            "pred_team_low_win": 0.5,
            "pred_team_low_win_raw": 0.51,
        },
    )

    assert winners["R1W1"] == 1101
    assert games[0]["winner_decision_rule"] == "raw_tiebreak"
    assert float(games[0]["pred_team_low_win"]) == 0.5
    assert float(games[0]["pred_team_low_win_raw"]) == 0.51


def test_actual_winner_map_has_championship_team_2025() -> None:
    raw_dir = Path("data/raw/latest")
    m_map = _actual_winner_map_for_2025(raw_dir, "M")
    w_map = _actual_winner_map_for_2025(raw_dir, "W")
    assert len(m_map) == 67
    assert len(w_map) == 67


def test_metrics_round_weighting_matches_overall() -> None:
    predicted_games = [
        {"slot": "R1W1", "round_num": 1, "winner_team_id": 1},
        {"slot": "R6CH", "round_num": 6, "winner_team_id": 1},
    ]
    actual_games = [
        {"slot": "R1W1", "round_label": "Round of 64", "round_num": 1, "winner_team_id": 2, "squared_error": 0.10},
        {"slot": "R1W2", "round_label": "Round of 64", "round_num": 1, "winner_team_id": 3, "squared_error": 0.30},
        {"slot": "R6CH", "round_label": "Championship", "round_num": 6, "winner_team_id": 1, "squared_error": 0.20},
    ]
    m = _metrics(predicted_games=predicted_games, actual_games=actual_games)
    overall = float(m["brier_overall"])
    by_round = m["brier_by_round"]
    n_round = m["games_by_round"]
    weighted = sum(float(by_round[r]) * int(n_round[r]) for r in by_round) / sum(int(n_round[r]) for r in by_round)
    assert abs(overall - weighted) < 1e-12


def test_metrics_final_four_overlap_uses_semifinalists_not_semifinal_winners() -> None:
    predicted_games = [
        {
            "slot": "R5WX",
            "round_label": "Final Four",
            "round_num": 5,
            "slot_order": 0,
            "team_low_id": 1,
            "team_low_name": "Duke",
            "team_high_id": 2,
            "team_high_name": "Houston",
            "winner_team_id": 1,
        },
        {
            "slot": "R5YZ",
            "round_label": "Final Four",
            "round_num": 5,
            "slot_order": 1,
            "team_low_id": 3,
            "team_low_name": "Auburn",
            "team_high_id": 4,
            "team_high_name": "Florida",
            "winner_team_id": 3,
        },
        {
            "slot": "R6CH",
            "round_label": "Championship",
            "round_num": 6,
            "slot_order": 2,
            "team_low_id": 1,
            "team_low_name": "Duke",
            "team_high_id": 3,
            "team_high_name": "Auburn",
            "winner_team_id": 1,
        },
    ]
    actual_games = [
        {
            "slot": "R5WX",
            "round_label": "Final Four",
            "round_num": 5,
            "slot_order": 0,
            "team_low_id": 1,
            "team_low_name": "Duke",
            "team_high_id": 2,
            "team_high_name": "Houston",
            "winner_team_id": 2,
            "squared_error": 0.25,
        },
        {
            "slot": "R5YZ",
            "round_label": "Final Four",
            "round_num": 5,
            "slot_order": 1,
            "team_low_id": 3,
            "team_low_name": "Auburn",
            "team_high_id": 4,
            "team_high_name": "Florida",
            "winner_team_id": 4,
            "squared_error": 0.25,
        },
        {
            "slot": "R6CH",
            "round_label": "Championship",
            "round_num": 6,
            "slot_order": 2,
            "team_low_id": 2,
            "team_low_name": "Houston",
            "team_high_id": 4,
            "team_high_name": "Florida",
            "winner_team_id": 4,
            "squared_error": 0.25,
        },
    ]

    m = _metrics(predicted_games=predicted_games, actual_games=actual_games)
    assert m["final_four_overlap_count"] == 4
    assert m["title_game_overlap_count"] == 0
