import numpy as np
import pandas as pd
import pytest

from pzu import value as V

B = V.Battery(power_mw=10, energy_mwh=20, round_trip_eff=1.0, max_cycles_per_day=1.0, cost_per_mwh=0.0)


def test_battery_buys_cheap_sells_expensive():
    price = np.array([10.0] * 8 + [100.0] * 8 + [10.0] * 8)
    c, d = V.dispatch(price, B)
    assert c[:8].sum() * V.DT == pytest.approx(20) and d[8:16].sum() * V.DT == pytest.approx(20)
    assert V.battery_revenue(c, d, price, B) == pytest.approx(20 * 90)


def test_battery_respects_limits_and_efficiency():
    b = V.Battery(10, 20, 0.81, 1.0, 0.0)
    rng = np.random.default_rng(0)
    price = rng.uniform(0, 200, 96)
    c, d = V.dispatch(price, b)
    soc = np.cumsum(0.9 * c * V.DT - d * V.DT / 0.9)
    assert soc.min() >= -1e-6 and soc.max() <= 20 + 1e-6 and c.max() <= 10 + 1e-9
    assert d.sum() * V.DT <= 20 + 1e-6                       # maxim 1 ciclu pe zi


def test_flat_prices_with_costs_do_nothing():
    b = V.Battery(10, 20, 0.85, 1.0, 4.0)
    c, d = V.dispatch(np.full(96, 50.0), b)
    assert c.sum() + d.sum() == pytest.approx(0, abs=1e-6)


def test_perfect_foresight_beats_wrong_forecast():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=96 * 3, freq="15min", tz="UTC")
    actual = 80 + 40 * np.sin(np.arange(len(idx)) / 96 * 2 * np.pi * 3) + rng.normal(0, 5, len(idx))
    bt = pd.DataFrame({"y": actual, "bad": -actual, "date": np.repeat([1, 2, 3], 96)}, index=idx)
    d = V.simulate_battery(bt, {"perfect": "y", "bad": "bad"}, V.Battery())
    assert (d["perfect"] > d["bad"]).all() and (d["perfect"] > 0).all()


def test_curtailment_avoids_negative_prices():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    bt = pd.DataFrame({"y": [50.0, -20.0, 50.0, -20.0], "model": [40.0, -10.0, 60.0, 5.0], "date": 1}, index=idx)
    gen = pd.Series(100.0, index=idx)
    d = V.simulate_curtailment(bt, gen, {"model": "model", "perfect": "y"})
    assert d.loc[1, "always_sell"] == pytest.approx(100 * (50 - 20 + 50 - 20) * 0.25)   # 1500
    assert d.loc[1, "model"] == pytest.approx(100 * (50 + 50 - 20) * 0.25)           # vinde in intervalele 0, 2, 3
    assert d.loc[1, "perfect"] > d.loc[1, "model"] > d.loc[1, "always_sell"]
