import numpy as np
import pandas as pd
import pytest

from pzu import db, pipeline
from pzu.ingest import opcom
from pzu.timegrid import day_index

SAMPLE = """Rezultate Cuplare PZU: 19/11/2026
"DAM INDEX"
"RO","Prices (EUR/MWh)","Change (%)"
"ROPEX_DAM_Base","216.37","13.57"

Interval,Price RO(EUR/MWh),Cross-Border flow (MW),,Available Transfer Capacity [MW],
,,RO_HU,HU_RO,RO_BG,BG_RO,RO_BG,BG_RO
32,263.24,2352.8,0.0,0.0,1573.0,2227.0,1573.0
33,89.60,1591.5,0.0,0.0,1535.0,2227.0,1573.0
36,0.00,1619.8,0.0,0.0,1573.0,2227.0,1573.0
37,252.82,3080.7,0.0,0.0,1573.0,2227.0,1573.0
"""


def test_parse_maps_market_intervals_cet_to_utc():
    df = opcom.parse_market_results(SAMPLE, "2026-11-19")
    assert list(df.columns) == opcom.COLS and len(df) == 4
    # intervalul 33 = 08:00 CET = 07:00 UTC = 09:00 ora Romaniei (pretul de 89,60 EUR din 19 nov 2026)
    assert df.loc[pd.Timestamp("2026-11-19 07:00", tz="UTC"), "price_eur"] == pytest.approx(89.60)
    assert df.loc[pd.Timestamp("2026-11-19 08:00", tz="UTC"), "flow_ro_hu"] == pytest.approx(3080.7)
    assert opcom.parse_market_results("nimic util", "2026-11-19").empty


def test_missing_ro_prices_are_filled_from_opcom_without_overwriting(tmp_path, monkeypatch):
    eng = db.get_engine(f"sqlite:///{tmp_path/'o.db'}")
    db.init_db(eng)
    day = "2026-11-20"
    idx = day_index(day)
    known = idx[:10]                                                  # ENTSO-E a publicat doar primele 10 intervale
    db.upsert_df(eng, db.prices_actual, pd.DataFrame({"timestamp": known, "zone": "RO", "price_eur_mwh": 100.0,
                                                       "source": "ENTSOE"}), ["timestamp", "zone"])
    asked = []

    def fake(mday, **k):
        asked.append(str(mday))
        t = pd.date_range(pd.Timestamp(mday, tz=opcom.MARKET_TZ), periods=96, freq="15min").tz_convert("UTC")
        return pd.DataFrame({c: np.full(96, 7.0) for c in opcom.COLS}, index=t).rename_axis("timestamp")

    monkeypatch.setattr(opcom, "fetch_market_results", fake)
    n = pipeline.fill_ro_from_opcom(eng, [pd.Timestamp(day).date()])
    got = db.read_actual_ro(eng, idx)
    assert n == len(idx) - 10 and got.notna().all()
    assert (got[known] == 100.0).all() and (got.drop(known) == 7.0).all()   # nimic suprascris
    assert asked == ["2026-11-20"]                                         # doar ziua de piata cu goluri
    assert pipeline.fill_ro_from_opcom(eng, [pd.Timestamp(day).date()]) == 0   # complet: nu mai cere nimic


def test_snapshot_is_used_when_opcom_blocks(tmp_path, monkeypatch):
    eng = db.get_engine(f"sqlite:///{tmp_path/'s.db'}")
    db.init_db(eng)
    day = "2026-11-21"
    t = pd.date_range(pd.Timestamp("2026-11-20", tz=opcom.MARKET_TZ), periods=96 * 2, freq="15min").tz_convert("UTC")
    snap = pd.DataFrame({c: np.full(len(t), 5.0) for c in opcom.COLS}, index=t).rename_axis("timestamp")
    monkeypatch.setattr(opcom, "fetch_market_results", lambda *a, **k: pd.DataFrame(columns=opcom.COLS))
    monkeypatch.setattr(opcom, "load_snapshot", lambda *a, **k: snap)
    assert pipeline.fill_ro_from_opcom(eng, [pd.Timestamp(day).date()]) == 96
    assert db.read_actual_ro(eng, day_index(day)).eq(5.0).all()


def test_forbidden_stops_further_requests(monkeypatch):
    calls = []

    class C:
        def get(self, url):
            calls.append(url)
            return type("R", (), {"status_code": 403, "text": ""})()

    monkeypatch.setattr(opcom, "_blocked", False)
    monkeypatch.setattr(opcom.time, "sleep", lambda s: None)
    assert opcom.fetch_market_results("2026-11-20", client=C()).empty
    assert opcom.fetch_market_results("2026-11-21", client=C()).empty
    assert len(calls) == 1
