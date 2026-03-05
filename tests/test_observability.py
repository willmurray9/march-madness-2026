from pathlib import Path

import pandas as pd

from mm2026.observability.report import _baseline_signature, _csv_summary, _family_signature, _promotion_for_gender, _upsert_runs_index


def test_csv_summary_counts_cells_and_nulls(tmp_path: Path) -> None:
    p = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, None], "b": [3, 4]}).to_csv(p, index=False)

    summary = _csv_summary(p)
    assert summary["exists"] is True
    assert summary["rows"] == 2
    assert summary["cols"] == 2
    assert summary["null_cells"] == 1
    assert abs(summary["null_fraction"] - 0.25) < 1e-9


def test_upsert_runs_index_replaces_existing_run_id(tmp_path: Path) -> None:
    p = tmp_path / "runs_index.csv"
    first = {"run_id": "A", "generated_at_utc": "20260101T000000Z", "m_oof_brier": 0.1}
    second = {"run_id": "A", "generated_at_utc": "20260101T000000Z", "m_oof_brier": 0.2}
    third = {"run_id": "B", "generated_at_utc": "20260102T000000Z", "m_oof_brier": 0.3}

    _upsert_runs_index(p, first)
    _upsert_runs_index(p, second)
    _upsert_runs_index(p, third)

    out = pd.read_csv(p)
    assert len(out) == 2
    assert out.loc[out["run_id"] == "A", "m_oof_brier"].iloc[0] == 0.2
    assert out["run_id"].tolist() == ["A", "B"]


def test_family_signature_is_stable() -> None:
    sig = _family_signature(
        {
            "trend": True,
            "advanced_rates": False,
            "volatility": True,
            "sos_adjusted": False,
            "elo_upgrades": True,
        }
    )
    assert sig == "advanced_rates=0|elo_upgrades=1|sos_adjusted=0|trend=1|volatility=1"
    assert "advanced_rates=0" in _baseline_signature()


def test_promotion_for_gender_uses_material_lift_and_non_worse() -> None:
    current = {
        "oof_brier_champion_calibrated": 0.1600,
        "season_metrics": {
            "2021": {"brier_champion_calibrated": 0.162},
            "2022": {"brier_champion_calibrated": 0.159},
            "2023": {"brier_champion_calibrated": 0.158},
            "2024": {"brier_champion_calibrated": 0.161},
            "2025": {"brier_champion_calibrated": 0.160},
        },
    }
    baseline = {
        "oof_brier_champion_calibrated": 0.1610,
        "season_metrics": {
            "2021": {"brier_champion_calibrated": 0.163},
            "2022": {"brier_champion_calibrated": 0.160},
            "2023": {"brier_champion_calibrated": 0.159},
            "2024": {"brier_champion_calibrated": 0.160},
            "2025": {"brier_champion_calibrated": 0.161},
        },
    }
    decision = _promotion_for_gender(
        gender="M",
        train_report=current,
        baseline_report=baseline,
        baseline_run_id="R0",
        gates={"min_material_lift": 0.0005, "min_non_worse_seasons": 4},
    )
    assert decision["status"] == "ok"
    assert decision["promoted"] is True
    assert decision["non_worse_seasons"] == 4
