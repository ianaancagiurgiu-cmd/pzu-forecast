import io
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pzu import db, model as M, pipeline
from pzu.api.main import create_app
from pzu.ingest import openmeteo
from pzu.synthetic import generate
from pzu.timegrid import day_index

DAY = date(2026, 3, 2)   # ziua tinta din setul de test


class OfflineClient:
    """ENTSO-E fara date noi: tot ce e in DB ramane neschimbat."""
    def query_day_ahead_prices(self, *a, **k): raise RuntimeError("NoMatchingDataError")
    query_wind_and_solar_forecast = query_load_forecast = query_net_transfer_capacity_dayahead = \
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("NoMatchingDataError"))


class PricesClient(OfflineClient):
    def __init__(self, prices: pd.Series):
        self.prices = prices

    def query_day_ahead_prices(self, code, start, end):
        return self.prices.loc[start:end].iloc[:-1].resample("h").first().dropna().tz_convert("Europe/Bucharest")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """DB cu 70 zile de date marcate ENTSOE (fixture de test, nu antrenare reala) + model mic."""
    mp = pytest.MonkeyPatch()
    mp.setattr(openmeteo, "fetch_weather", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    mp.setitem(M.PARAMS, "n_estimators", 40)
    path = tmp_path_factory.mktemp("db") / "p.db"
    eng = db.get_engine(f"sqlite:///{path}")
    db.init_db(eng)
    inputs, prices = generate("2026-01-01", 70, seed=5)
    long = prices.stack().rename("price_eur_mwh").reset_index()
    long.columns = ["timestamp", "zone", "price_eur_mwh"]
    long["source"] = "ENTSOE"
    db.upsert_df(eng, db.prices_actual, long, ["timestamp", "zone"])
    db.upsert_df(eng, db.forecasts_input, inputs[[c for c in db.INPUT_COLS if c in inputs]], ["timestamp"])
    msg = pipeline.retrain(eng, None, window_days=20, calib_days=20, version="vtest")
    assert "vtest" in msg
    yield eng, prices["RO"]
    mp.undo()


def test_daily_creates_96_intervals_and_is_idempotent(env):
    eng, _ = env
    assert "salvat" in pipeline.daily(eng, OfflineClient(), DAY)
    idx = day_index(DAY)
    fc = db.read_forecast(eng, idx)
    assert len(fc) == 96 and (fc["ci_lower"] < fc["ci_upper"]).all()
    assert "exista deja" in pipeline.daily(eng, OfflineClient(), DAY)


def test_daily_raises_not_ready_when_inputs_missing(env):
    eng, _ = env
    with pytest.raises(pipeline.NotReady):
        pipeline.daily(eng, OfflineClient(), date(2026, 6, 1), force=True)


def test_daily_requires_a_trained_model(tmp_path):
    eng = db.get_engine(f"sqlite:///{tmp_path/'x.db'}")
    db.init_db(eng)
    with pytest.raises(RuntimeError, match="model"):
        pipeline.daily(eng, OfflineClient(), DAY)


def test_evaluate_writes_accuracy(env):
    eng, ro = env
    pipeline.daily(eng, OfflineClient(), DAY)
    msg = pipeline.evaluate(eng, PricesClient(ro), DAY)
    assert "MAE" in msg
    acc = db.read_df(eng, db.forecast_accuracy, index=None)
    assert len(acc) == 1 and acc["mae"].iloc[0] > 0 and 0 <= acc["coverage_90"].iloc[0] <= 1
    assert "existenta" in pipeline.evaluate(eng, PricesClient(ro), DAY)


def test_evaluate_not_ready_when_prices_unpublished(env):
    eng, _ = env
    with pytest.raises(pipeline.NotReady):
        pipeline.evaluate(eng, OfflineClient(), DAY, force=True)


def test_target_day_is_tomorrow_local():
    now = pd.Timestamp("2026-10-20 03:30", tz="UTC")     # 06:30 ora Romaniei
    assert pipeline.target_day(now) == date(2026, 10, 21)
    late = pd.Timestamp("2026-10-20 22:30", tz="UTC")    # deja 21 oct local
    assert pipeline.target_day(late) == date(2026, 10, 22)


@pytest.fixture
def client(env):
    eng, _ = env
    pipeline.daily(eng, OfflineClient(), DAY)
    return TestClient(create_app(eng))


def test_api_forecast_matches_spec(client):
    r = client.get("/api/v1/forecast/pzu", params={"date": "2026-03-02"})
    assert r.status_code == 200
    body = r.json()
    assert body["date"] == "2026-03-02" and body["model_version"] == "vtest" and len(body["intervals"]) == 96
    first = body["intervals"][0]
    assert first["start"] == "2026-03-02T00:00:00+02:00" and first["end"] == "2026-03-02T00:15:00+02:00"
    assert first["ci_90_lower"] < first["predicted_price"] < first["ci_90_upper"]


def test_api_404_and_validation(client):
    assert client.get("/api/v1/forecast/pzu", params={"date": "2030-01-01"}).status_code == 404
    assert client.get("/api/v1/forecast/pzu", params={"date": "nu-e-data"}).status_code == 422


def test_api_summary_indicators(client):
    s = client.get("/api/v1/forecast/pzu/summary", params={"date": "2026-03-02"}).json()
    assert s["spread"] == pytest.approx(s["max"] - s["min"], abs=0.02)
    assert s["hours_below_30"] == s["intervals_below_30"] / 4


def test_api_exports(client):
    csv = client.get("/api/v1/forecast/pzu/export", params={"date": "2026-03-02", "format": "csv"})
    assert csv.headers["content-type"].startswith("text/csv")
    assert len(pd.read_csv(io.StringIO(csv.text))) == 96
    xl = client.get("/api/v1/forecast/pzu/export", params={"date": "2026-03-02", "format": "xlsx"})
    assert len(pd.read_excel(io.BytesIO(xl.content))) == 96


def test_api_dates_accuracy_and_dashboard(client):
    assert "2026-03-02" in client.get("/api/v1/forecast/dates").json()["dates"]
    assert "items" in client.get("/api/v1/accuracy").json()
    page = client.get("/")
    assert page.status_code == 200 and "Forecast PZU" in page.text
    assert client.get("/health").json() == {"status": "ok"}


def test_api_includes_benchmarks(client):
    body = client.get("/api/v1/forecast/pzu", params={"date": "2026-03-02"}).json()
    first = body["intervals"][0]
    assert first["mean_7d_price"] is not None and first["blend_price"] is not None
    assert first["blend_price"] == pytest.approx(0.5 * first["predicted_price"] + 0.5 * first["mean_7d_price"], abs=0.02)
    csv = client.get("/api/v1/forecast/pzu/export", params={"date": "2026-03-02", "format": "csv"}).text
    assert "blend_price" in csv.splitlines()[0]


def test_init_db_adds_new_columns_to_an_old_database(tmp_path):
    from sqlalchemy import inspect, text
    eng = db.get_engine(f"sqlite:///{tmp_path/'old.db'}")
    with eng.begin() as conn:   # schema veche, fara coloanele noi
        conn.execute(text("CREATE TABLE price_forecast (timestamp DATETIME PRIMARY KEY, forecast_generated_at DATETIME, "
                          "predicted_price FLOAT, ci_lower FLOAT, ci_upper FLOAT, model_version VARCHAR(20), features_used JSON)"))
        conn.execute(text("INSERT INTO price_forecast (timestamp, predicted_price) VALUES ('2026-01-01 00:00:00', 50.0)"))
    db.init_db(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("price_forecast")}
    assert {"mean7d_price", "blend_price"} <= cols
    with eng.connect() as conn:   # datele existente raman
        assert conn.execute(text("SELECT predicted_price FROM price_forecast")).scalar_one() == 50.0
    db.init_db(eng)               # idempotent


def test_api_returns_actual_prices_for_comparison(env, client):
    eng, _ = env
    body = client.get("/api/v1/forecast/pzu", params={"date": "2026-03-02"}).json()
    got = [x["actual_price"] for x in body["intervals"]]
    want = db.read_actual_ro(eng, day_index(DAY)).round(2).tolist()   # ce e publicat in DB
    assert got == pytest.approx(want, abs=0.011)


def test_actual_is_null_until_published(env, client):
    eng, _ = env
    from sqlalchemy import text
    with eng.begin() as conn:     # simulam "pretul oficial inca nepublicat"
        conn.execute(text("DELETE FROM prices_actual WHERE zone='RO' AND timestamp >= '2026-03-01 22:00:00'"))
    body = client.get("/api/v1/forecast/pzu", params={"date": "2026-03-02"}).json()
    assert all(x["actual_price"] is None for x in body["intervals"])


# ---------------------------------------------------------------- reantrenare zilnica ----
def _stale(bundle, days_ago_trained=1, days_ago_calibrated=2):
    import dataclasses
    now = pd.Timestamp.now(tz="UTC")
    return dataclasses.replace(bundle, trained_at=(now - pd.Timedelta(days=days_ago_trained)).isoformat(),
                               calibrated_at=(now - pd.Timedelta(days=days_ago_calibrated)).isoformat())


def _stored(eng):
    return M.ModelBundle.from_bytes(db.load_model_bytes(eng)[1])


def test_refresh_is_a_noop_when_model_is_fresh(env):
    eng, _ = env
    b = _stored(eng)
    assert pipeline.refresh_model(eng, b) is b


def test_refresh_refits_but_keeps_calibration_when_stale(env):
    eng, _ = env
    b = _stale(_stored(eng))
    fresh = pipeline.refresh_model(eng, b)
    assert fresh.version != b.version and fresh.version.startswith("v2-")
    assert fresh.conformal is b.conformal and fresh.features == b.features       # calibrarea refolosita
    assert _stored(eng).version == fresh.version                                  # salvat in DB
    assert pipeline.refresh_model(eng, fresh) is fresh                            # a doua incercare: nimic


def test_refresh_recalibrates_when_calibration_is_old(env):
    eng, _ = env
    b = _stale(_stored(eng), days_ago_calibrated=10)
    fresh = pipeline.refresh_model(eng, b, before=day_index(DAY)[0])
    assert fresh.calibrated_at is not None and fresh.conformal is not b.conformal


def test_refresh_never_trains_on_the_target_day(env, monkeypatch):
    eng, _ = env
    seen = {}
    real = M.fit

    def spy(X, y, cols=None, **kw):
        seen["last"] = X.index.max()
        return real(X, y, cols, **kw)

    monkeypatch.setattr(M, "fit", spy)
    cutoff = day_index(DAY)[0]
    pipeline.refresh_model(eng, _stale(_stored(eng)), before=cutoff)
    assert seen["last"] < cutoff


def test_daily_falls_back_to_existing_model_if_refresh_fails(env, monkeypatch):
    eng, _ = env
    monkeypatch.setattr(pipeline, "refresh_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    msg = pipeline.daily(eng, OfflineClient(), DAY, force=True)
    assert "salvat" in msg or "actualizat" in msg and "varianta" in msg


def test_regional_features_are_dropped_when_history_is_missing(env):
    eng, _ = env
    X, y = __import__("pzu.features", fromlist=["x"]).build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng))
    cols = pipeline._main_cols(X, slice(None))          # DB-ul de test nu are preturi regionale / consum vecini
    from pzu import features as F
    assert not set(F.NEIGHBOR_LOADS) & set(cols)                       # fara consum vecini in DB
    assert not any(c in cols for c in ("price_at_lag_24h", "price_de_lag_24h", "price_gr_lag_7d"))
    assert set(F.BASE_PRODUCTION_FEATURES) <= set(cols)


# --------------------------------------------- actualizari: rezerva -> complet, date schimbate, istoric ----
EARLY = pd.Timestamp("2026-03-01 09:00", tz="Europe/Bucharest")     # inainte de 13:00 din ziua dinaintea lui DAY


def F_NB():
    from pzu import features as F
    return F.NEIGHBOR_LOADS


def test_gate_closes_at_13_local_time_the_day_before():
    assert not pipeline.gate_closed(DAY, pd.Timestamp("2026-03-01 12:59", tz="Europe/Bucharest"))
    assert pipeline.gate_closed(DAY, pd.Timestamp("2026-03-01 13:00", tz="Europe/Bucharest"))
    assert pipeline.gate_closed(DAY, pd.Timestamp("2026-03-02 08:00", tz="Europe/Bucharest"))


def test_fingerprint_changes_only_when_the_known_data_change(env):
    eng, _ = env
    inputs, prices = db.read_df(eng, db.forecasts_input), db.read_prices(eng)
    prices = prices[prices.index < day_index(DAY)[0]]
    base = pipeline.inputs_fingerprint(inputs, None, prices, DAY)
    assert base == pipeline.inputs_fingerprint(inputs.copy(), None, prices, DAY)
    revised = inputs.copy()
    revised.loc[day_index(DAY)[5], "load_forecast_mw"] += 50
    assert pipeline.inputs_fingerprint(revised, None, prices, DAY) != base
    nl = pd.DataFrame({z: 3000.0 for z in ("HU", "BG", "RS", "SK")}, index=inputs.index)
    assert pipeline.inputs_fingerprint(inputs, nl, prices, DAY) != base


def test_fallback_forecast_is_upgraded_when_neighbor_load_appears(env):
    eng, _ = env
    inputs = db.read_df(eng, db.forecasts_input)
    idx = day_index(DAY)
    ts = inputs.index
    full = pd.concat([pd.DataFrame({"timestamp": ts, "zone": z, "load_forecast_mw": 3000.0 + (np.arange(len(ts)) % 96)})
                      for z in ("HU", "BG", "RS", "SK")])
    db.upsert_df(eng, db.neighbor_load, full[full["timestamp"] < idx[0]], ["timestamp", "zone"])   # istoric; ziua tinta lipseste
    pipeline.retrain(eng, None, window_days=20, calib_days=20, version="vnb", before=idx[0])
    assert any(c in _stored(eng).features for c in F_NB()), "modelul principal trebuie sa foloseasca consumul vecinilor"

    assert "varianta fallback" in pipeline.daily(eng, OfflineClient(), DAY, force=True, now=EARLY)
    assert pipeline._existing_variant(eng, idx) == "fallback"
    assert "datele sunt neschimbate" in pipeline.daily(eng, OfflineClient(), DAY, now=EARLY)     # nimic nou -> nu rescrie
    db.upsert_df(eng, db.neighbor_load, full[full["timestamp"].isin(idx)], ["timestamp", "zone"])  # apar consumurile vecinilor
    msg = pipeline.daily(eng, OfflineClient(), DAY, now=EARLY + pd.Timedelta(hours=1))
    assert "actualizat" in msg and "varianta main" in msg
    assert pipeline._existing_variant(eng, idx) == "main"
    assert "datele sunt neschimbate" in pipeline.daily(eng, OfflineClient(), DAY, now=EARLY + pd.Timedelta(hours=2))


def test_revised_load_forecast_creates_a_new_version_until_the_gate_closes(env):
    eng, _ = env
    idx = day_index(DAY)
    n0 = db.read_forecast_history(eng, idx)["generated_at"].nunique()
    inputs = db.read_df(eng, db.forecasts_input)
    inputs.loc[idx, "load_forecast_mw"] = inputs.loc[idx, "load_forecast_mw"] + 120        # revizuire a prognozei de consum
    db.upsert_df(eng, db.forecasts_input, inputs.loc[idx], ["timestamp"])
    assert "actualizat" in pipeline.daily(eng, OfflineClient(), DAY, now=EARLY + pd.Timedelta(hours=3))
    assert db.read_forecast_history(eng, idx)["generated_at"].nunique() == n0 + 1
    inputs.loc[idx, "load_forecast_mw"] = inputs.loc[idx, "load_forecast_mw"] + 120
    db.upsert_df(eng, db.forecasts_input, inputs.loc[idx], ["timestamp"])
    late = pd.Timestamp("2026-03-01 13:30", tz="Europe/Bucharest")                          # piata s-a inchis
    assert "piata e inchisa" in pipeline.daily(eng, OfflineClient(), DAY, now=late)
    assert db.read_forecast_history(eng, idx)["generated_at"].nunique() == n0 + 1           # nicio versiune noua


def test_api_exposes_versions_variant_and_errors(env):
    eng, ro = env
    idx = day_index(DAY)
    long = ro.reindex(idx).dropna().rename("price_eur_mwh").rename_axis("timestamp").reset_index()   # pretul oficial (sters de un test anterior)
    long["zone"], long["source"] = "RO", "ENTSOE"
    db.upsert_df(eng, db.prices_actual, long, ["timestamp", "zone"])
    client = TestClient(create_app(eng))
    body = client.get("/api/v1/forecast/pzu", params={"date": "2026-03-02"}).json()
    assert body["variant"] in ("main", "fallback") and body["updates"] >= 2
    v = client.get("/api/v1/forecast/pzu/versions", params={"date": "2026-03-02"}).json()["versions"]
    assert len(v) == body["updates"] and v == sorted(v, key=lambda x: x["generated_at"])
    assert all(x["mae"] is not None for x in v)            # exista pretul oficial in DB-ul de test
