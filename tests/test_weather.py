from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pzu import model as M, weather as W


def _groups(days=4, wind_by_day=(10.0, 8.0, 4.0, 4.0)):
    idx = pd.date_range("2026-05-31 21:00", periods=24 * days, freq="1h", tz="UTC")   # zile locale 1-4 iunie
    d = {g: np.full(len(idx), 5.0) for g in W.GROUPS}
    d["wind_nw"] = np.repeat(np.array(wind_by_day[:days], dtype=float), 24)
    return pd.DataFrame(d, index=idx)


def test_weather_features_use_previous_day_even_for_a_single_day_index():
    g = _groups()
    day = pd.date_range("2026-06-02 21:00", periods=96, freq="15min", tz="UTC")   # 3 iunie ora locala
    f = W.weather_features(g, day)
    assert list(f.columns) == W.FEATURE_NAMES
    assert f["wx_wind_nw_zi"].iloc[0] == pytest.approx(4.0)
    assert f["wx_wind_nw_chg"].iloc[0] == pytest.approx(-4.0, abs=0.2)      # 4 fata de 8 din ziua precedenta (interpolarea orara netezeste putin granita)
    assert f["wx_wind_nw_max"].iloc[0] == pytest.approx(4.0)


def test_weather_features_are_nan_without_data():
    idx = pd.date_range("2026-06-03", periods=4, freq="15min", tz="UTC")
    assert W.weather_features(pd.DataFrame(), idx).isna().all().all()


def _bundle(with_wx=True):
    cols = ["load_forecast_mw", "wx_wind_nw_zi"] if with_wx else ["load_forecast_mw"]
    return SimpleNamespace(model="M", features=cols, nowx_model="N" if with_wx else None, nowx_features=["load_forecast_mw"],
                           fallback_model="F", fallback_features=["load_forecast_mw"])


def test_variant_falls_back_when_weather_or_neighbor_data_is_missing():
    X = pd.DataFrame({"load_forecast_mw": [1.0, 2.0], "wx_wind_nw_zi": [3.0, 4.0]})
    assert M.pick_variant(_bundle(), X)[2] == "main"
    assert M.pick_variant(_bundle(), X.assign(wx_wind_nw_zi=np.nan))[2] == "noweather"
    assert M.pick_variant(_bundle(), X.drop(columns="wx_wind_nw_zi"))[2] == "noweather"
    b = _bundle()
    b.features = b.features + ["load_hu"]
    assert M.pick_variant(b, X.assign(load_hu=np.nan))[2] == "fallback"
