import numpy as np
import pandas as pd
import pytest

from pzu import features as F
from pzu.timegrid import day_index


def test_static_leakage_check_rejects_short_lags():
    bad = {**F.FEATURE_SPECS, "price_lag_1h": ("lag", 1)}
    with pytest.raises(AssertionError):
        F.assert_no_leakage(bad)


def test_no_neighbor_d1_prices_in_features():
    # preturile vecinilor apar doar ca lag-uri
    for name in F.FEATURES:
        if name.startswith("price_") and name.split("_")[1] in {"hu", "bg", "rs", "sk"}:
            assert "lag" in name


def test_features_shape_and_columns(synth):
    inputs, prices = synth
    X = F.build_features(inputs, prices, day_index("2026-01-20"))
    assert list(X.columns) == F.FEATURES and len(X) == 96
    assert X["price_lag_24h"].notna().all()


def test_dynamic_leakage_check_passes(synth):
    inputs, prices = synth
    F.verify_no_leakage(inputs, prices, "2026-01-25")


def test_dynamic_leakage_check_detects_leak(synth, monkeypatch):
    inputs, prices = synth
    real = F._lag

    def leaky(s, index, delta):  # simuleaza un lag de 0h (pret de azi in feature)
        return real(s, index, pd.Timedelta(0))

    monkeypatch.setattr(F, "_lag", leaky)
    with pytest.raises(AssertionError):
        F.verify_no_leakage(inputs, prices, "2026-01-25")


def test_rolling_mean_is_mean_of_prior_days(synth):
    inputs, prices = synth
    idx = day_index("2026-01-20")[:1]
    X = F.build_features(inputs, prices, idx)
    expected = np.mean([prices["RO"].loc[idx[0] - pd.Timedelta(days=k)] for k in range(1, 8)])
    assert X["price_rolling_mean_7d"].iloc[0] == pytest.approx(expected)


def test_calendar_features_use_local_time(synth):
    inputs, prices = synth
    X = F.build_features(inputs, prices, day_index("2026-01-20"))
    assert X["hour"].iloc[0] == 0 and X["quarter"].iloc[1] == 1 and X["hour"].iloc[-1] == 23
    assert F.build_features(inputs, prices, day_index("2026-01-01"))["is_holiday"].all()


def test_training_set_has_no_missing_target(synth):
    inputs, prices = synth
    X, y = F.build_training_set(inputs, prices)
    assert len(X) == len(y) > 0 and y.notna().all()
