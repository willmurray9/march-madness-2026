import pandas as pd
import pytest

from mm2026.submission.generate import _validate_submission, _validate_submission_against_sample


def test_validate_submission_happy_path() -> None:
    df = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})
    _validate_submission(df)


def test_validate_submission_bounds() -> None:
    df = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [1.5]})
    with pytest.raises(ValueError):
        _validate_submission(df)


def test_validate_submission_rejects_duplicate_ids() -> None:
    df = pd.DataFrame({"ID": ["2026_1101_1102", "2026_1101_1102"], "Pred": [0.5, 0.6]})
    with pytest.raises(ValueError, match="duplicate IDs"):
        _validate_submission(df)


def test_validate_submission_against_sample_checks_row_count_and_ids() -> None:
    sample = pd.DataFrame({"ID": ["2026_1101_1102", "2026_1101_1103"], "Pred": [0.5, 0.5]})
    submission = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})

    with pytest.raises(ValueError, match="row count mismatch"):
        _validate_submission_against_sample(submission, sample)
