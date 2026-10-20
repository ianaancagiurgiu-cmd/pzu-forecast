import httpx
import pandas as pd

from pzu import db
from pzu.ingest import entsoe, openmeteo


def test_db_roundtrip_and_upsert(tmp_path, synth):
    inputs, prices = synth
    eng = db.get_engine(f"sqlite:///{tmp_path/'t.db'}")
    db.init_db(eng)
    long = prices.iloc[:200].stack().rename("price_eur_mwh").reset_index()
    long.columns = ["timestamp", "zone", "price_eur_mwh"]
    long["source"] = "SYNTHETIC"
    assert db.upsert_df(eng, db.prices_actual, long, ["timestamp", "zone"]) == len(long)
    db.upsert_df(eng, db.prices_actual, long, ["timestamp", "zone"])  # idempotent
    back = db.read_prices(eng)
    assert back.shape == (200, 5)
    pd.testing.assert_series_equal(back["RO"], prices["RO"].iloc[:200], check_freq=False, check_names=False, check_index_type=False)
    db.upsert_df(eng, db.forecasts_input, inputs.iloc[:96], ["timestamp"])
    assert len(db.read_df(eng, db.forecasts_input)) == 96


class FakeEntsoe:
    def __init__(self):
        self.idx = pd.date_range("2026-01-10", periods=24, freq="h", tz="Europe/Bucharest")

    def query_day_ahead_prices(self, code, start, end):
        return pd.Series(range(24), index=self.idx, dtype=float)

    def query_wind_and_solar_forecast(self, code, start, end, **kw):
        return pd.DataFrame({"Solar": 1.0, "Wind Onshore": 2.0}, index=self.idx)

    def query_load_forecast(self, code, start, end):
        return pd.DataFrame({"Forecasted Load": 5000.0}, index=self.idx)

    def query_net_transfer_capacity_dayahead(self, a, b, start, end):
        return pd.Series(300.0, index=self.idx)


def test_entsoe_normalizes_to_15min_utc():
    out = entsoe.fetch_inputs(FakeEntsoe(), "2026-01-10", "2026-01-11")
    assert str(out.index.tz) == "UTC" and len(out) == 96
    assert {"solar_forecast_mw", "wind_forecast_mw", "load_forecast_mw", "capacity_hu_ro_mw"} <= set(out)
    assert out["wind_forecast_mw"].eq(2.0).all()


def test_entsoe_missing_data_does_not_raise():
    class Broken(FakeEntsoe):
        def query_day_ahead_prices(self, *a, **k):
            raise RuntimeError("NoMatchingDataError")

    assert entsoe.fetch_prices(Broken(), "SK", "2026-01-10", "2026-01-11").empty


def test_openmeteo_parses_and_averages():
    def handler(request: httpx.Request) -> httpx.Response:
        times = [f"2026-01-10T{h:02d}:00" for h in range(24)]
        return httpx.Response(200, json={"hourly": {
            "time": times, "temperature_2m": [5.0] * 24,
            "wind_speed_10m": [3.0] * 24, "shortwave_radiation": [100.0] * 24}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    w = openmeteo.fetch_weather("2026-01-10", "2026-01-10", client=client)
    assert list(w.columns) == ["temp_c", "wind_speed_ms", "radiation_wm2"]
    assert len(w) == 96 and w["temp_c"].eq(5.0).all()


def test_entsoe_errors_never_log_the_token(caplog):
    class Leaky(FakeEntsoe):
        def query_day_ahead_prices(self, *a, **k):
            raise RuntimeError("url=/api?securityToken=SECRET-123&periodStart=1")

    with caplog.at_level("WARNING"):
        entsoe.fetch_prices(Leaky(), "RO", "2026-01-10", "2026-01-11")
    assert "SECRET-123" not in caplog.text and "securityToken=***" in caplog.text


def test_openmeteo_retries_on_429(monkeypatch):
    monkeypatch.setattr(openmeteo, "SLEEP", False)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, json={"error": True})
        times = [f"2026-01-10T{h:02d}:00" for h in range(24)]
        return httpx.Response(200, json={"hourly": {
            "time": times, "temperature_2m": [5.0] * 24,
            "wind_speed_10m": [3.0] * 24, "shortwave_radiation": [100.0] * 24}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(openmeteo.fetch_weather("2026-01-10", "2026-01-10", client=client)) == 96


def test_openmeteo_retries_on_connection_errors(monkeypatch):
    monkeypatch.setattr(openmeteo, "SLEEP", False)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise httpx.ConnectTimeout("handshake timed out")
        times = [f"2026-01-10T{h:02d}:00" for h in range(24)]
        return httpx.Response(200, json={"hourly": {
            "time": times, "temperature_2m": [5.0] * 24,
            "wind_speed_10m": [3.0] * 24, "shortwave_radiation": [100.0] * 24}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert len(openmeteo.fetch_weather("2026-01-10", "2026-01-10", client=client)) == 96
