from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mm2026.observability.publishable_bracket import _matchup_key, _seed_map, _selected_bracket_payload, _to_builtin


def test_matchup_key_is_stable() -> None:
    assert _matchup_key("2026_predicted", 1101, 1242) == "2026_predicted:1101:1242"


def test_selected_bracket_payload_prefers_current_then_submission() -> None:
    current = {"genders": {"M": {"status": "ok", "summary": {"champion_team_name": "Duke"}}}}
    submission = {"genders": {"M": {"status": "ok", "summary": {"champion_team_name": "Florida"}}}}

    payload, source = _selected_bracket_payload(current, submission, "M")

    assert source == "local model"
    assert payload["summary"]["champion_team_name"] == "Duke"

    payload, source = _selected_bracket_payload({"genders": {}}, submission, "M")

    assert source == "submission fallback"
    assert payload["summary"]["champion_team_name"] == "Florida"


def test_to_builtin_normalizes_numpy_pandas_and_paths() -> None:
    value = {
        "int": np.int64(7),
        "float": np.float64(0.25),
        "bool": np.bool_(True),
        "nan": np.nan,
        "path": Path("deploy/bracket_center_payload.json"),
        "df": pd.DataFrame([{"team": "Duke", "prob": np.float64(0.6)}]),
    }

    out = _to_builtin(value)

    assert out == {
        "int": 7,
        "float": 0.25,
        "bool": True,
        "nan": None,
        "path": "deploy/bracket_center_payload.json",
        "df": [{"team": "Duke", "prob": 0.6}],
    }


def test_seed_map_parses_numeric_seed_from_seed_code(monkeypatch) -> None:
    def fake_read_csv(path: Path) -> pd.DataFrame:
        if path.name == "M_tourney_seeds.csv":
            return pd.DataFrame(
                [
                    {"Season": 2026, "Seed": "W01", "TeamID": 1104},
                    {"Season": 2026, "Seed": "X12", "TeamID": 1242},
                ]
            )
        if path.name == "W_tourney_seeds.csv":
            return pd.DataFrame(columns=["Season", "Seed", "TeamID"])
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr("mm2026.observability.publishable_bracket.read_csv", fake_read_csv)
    cfg = {"data": {"curated_dir": "data/curated/latest"}}

    out = _seed_map(cfg)

    assert out == {
        (2026, 1104): 1.0,
        (2026, 1242): 12.0,
    }
