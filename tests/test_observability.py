from pathlib import Path

import pandas as pd

from mm2026.observability.report import _csv_summary, _upsert_runs_index


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

