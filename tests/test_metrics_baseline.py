import numpy as np
import pytest

from pzu import metrics as M
from pzu.baseline import evaluate_baselines
from pzu.features import build_training_set


def test_basic_metrics():
    y, p = np.array([10.0, 20, 30]), np.array([12.0, 18, 30])
    assert M.mae(y, p) == pytest.approx(4 / 3)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(8 / 3))
    assert M.smape([100], [50]) == pytest.approx(100 * 2 * 50 / 150)
    assert M.smape([0], [0]) == 0


def test_directional_and_coverage():
    assert M.directional_hit_rate([1, 2, 1, 3], [5, 6, 5, 7]) == pytest.approx(1.0)
    assert M.directional_hit_rate([1, 2, 1], [2, 1, 2]) == 0.0
    assert M.coverage([1, 5, 9], [0, 0, 0], [2, 6, 8]) == pytest.approx(2 / 3)


def test_baselines_run(synth):
    X, y = build_training_set(*synth)
    res = evaluate_baselines(X, y)
    assert set(res.index) == {"naive_lag24h", "naive_lag7d", "naive_mean7d"}
    assert (res["mae"] > 0).all()
