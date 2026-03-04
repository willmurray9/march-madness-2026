import numpy as np

from mm2026.utils.metrics import brier_score


def test_brier_score() -> None:
    y = np.array([0, 1, 1, 0], dtype=float)
    p = np.array([0.2, 0.8, 0.9, 0.1], dtype=float)
    assert abs(brier_score(y, p) - 0.025) < 1e-9
