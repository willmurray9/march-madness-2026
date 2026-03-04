import pandas as pd
import pytest

from mm2026.submission.generate import _validate_submission


def test_validate_submission_happy_path() -> None:
    df = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [0.5]})
    _validate_submission(df)


def test_validate_submission_bounds() -> None:
    df = pd.DataFrame({"ID": ["2026_1101_1102"], "Pred": [1.5]})
    with pytest.raises(ValueError):
        _validate_submission(df)
