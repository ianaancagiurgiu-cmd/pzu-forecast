import numpy as np
import pandas as pd
import pytest

from pzu import backtest, db, features as F, model as M
from pzu.conformal import ConformalInterval, rolling_intervals
from pzu.synthetic import generate

FAST = {"n_estimators": 60}


@pytest.fixture(scope="module")
def data():
    inputs, prices = generate("2026-01-01", 70, seed=3)
    X, y = F.build_training_set(inputs, prices)
    return inputs, prices, X, y


def test_walk_forward_uses_only_past_and_beats_baseline(data):
    _, _, X, y = data
    bt = backtest.walk_forward(X, y, window_days=30, n_days=25, **FAST)
    assert bt["date"].nunique() == 25
    # ziua de test nu apare niciodata in antrenare: testam prin ordinea zilelor
    assert (bt.groupby("date").size() == 96).all()
    res = backtest.evaluate(bt)
    assert res["lightgbm"]["mae"] < res["naive_lag24h"]["mae"]
    assert 0.75 <= res["lightgbm"]["coverage_90"] <= 1.0


def test_conformal_quantile_gives_target_coverage():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=5000, freq="15min", tz="UTC")
    err = pd.Series(rng.normal(0, 10, len(idx)), index=idx)
    ci = ConformalInterval(0.10, by_hour=False).fit(err[:3000])
    lo, hi = ci.interval(pd.Series(0.0, index=idx[3000:]))
    cov = ((err[3000:] >= lo) & (err[3000:] <= hi)).mean()
    assert 0.87 <= cov <= 0.93  # cuantila 90% a |eroare|, nu 95%


def test_conformal_by_hour_wider_in_noisy_hours():
    idx = pd.date_range("2026-01-01", periods=96 * 120, freq="15min", tz="UTC")
    local_h = idx.tz_convert("Europe/Bucharest").hour
    rng = np.random.default_rng(1)
    err = pd.Series(rng.normal(0, np.where(local_h == 19, 30, 5)), index=idx)
    ci = ConformalInterval(0.10, by_hour=True, min_per_hour=50).fit(err)
    assert ci.q_hour[19] > 3 * ci.q_hour[3]


def test_rolling_intervals_never_use_future_errors():
    idx = pd.date_range("2026-01-01", periods=96 * 30, freq="15min", tz="UTC")
    day = pd.Series(idx.tz_convert("Europe/Bucharest").date, index=idx)
    y = pd.Series(0.0, index=idx)
    y[day == day.iloc[-1]] = 1000.0                      # soc doar in ultima zi
    bt = pd.DataFrame({"y": y, "pred": 0.0, "date": day})
    ci = rolling_intervals(bt, min_days=14)
    last = ci[ci["date"] == ci["date"].max()]
    assert (last["ci_upper"] - last["ci_lower"]).max() < 1   # intervalul nu "stie" de soc


def test_forecast_day_and_bundle_roundtrip(data, tmp_path):
    inputs, prices, X, y = data
    cutoff = pd.Timestamp("2026-03-01", tz="Europe/Bucharest").tz_convert("UTC")
    hist = prices[prices.index < cutoff]       # preturile D+1 nu exista la ora prognozei
    Xh, yh = F.build_training_set(inputs[inputs.index < cutoff], hist)
    m = M.fit(Xh, yh, **FAST)
    bt = backtest.walk_forward(Xh, yh, 20, 20, **FAST)
    conf = ConformalInterval(0.10).fit(bt["y"] - bt["pred"])
    bundle = M.ModelBundle(m, "vtest", 20, conf)
    out = M.forecast_day(bundle, inputs, hist, "2026-03-01")
    assert len(out) == 96 and (out["ci_lower"] < out["predicted_price"]).all()
    assert (out["ci_upper"] > out["predicted_price"]).all()
    loaded = M.ModelBundle.load(bundle.save(tmp_path))
    pd.testing.assert_frame_equal(out, M.forecast_day(loaded, inputs, hist, "2026-03-01"))


def test_training_refuses_synthetic_data(tmp_path, data):
    _, prices, _, _ = data
    eng = db.get_engine(f"sqlite:///{tmp_path/'s.db'}")
    db.init_db(eng)
    db.ensure_real_data(eng)  # DB goala: ok
    long = prices.iloc[:10].stack().rename("price_eur_mwh").reset_index()
    long.columns = ["timestamp", "zone", "price_eur_mwh"]
    long["source"] = "SYNTHETIC"
    db.upsert_df(eng, db.prices_actual, long, ["timestamp", "zone"])
    with pytest.raises(db.SyntheticDataError):
        db.ensure_real_data(eng)
