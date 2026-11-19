import numpy as np
import pandas as pd

from pzu import db, insights as I
from pzu.timegrid import day_index

DAY = "2026-06-10"


def _engine(tmp_path, ro, hu, prev_level=200.0):
    eng = db.get_engine(f"sqlite:///{tmp_path/'i.db'}")
    db.init_db(eng)
    idx = day_index(DAY)
    prev = day_index(pd.Timestamp(DAY) - pd.Timedelta(days=1))
    rows = []
    for zone, cur in (("RO", ro), ("HU", hu)):
        rows += [{"timestamp": t, "zone": zone, "price_eur_mwh": prev_level, "source": "ENTSOE"} for t in prev]
        rows += [{"timestamp": t, "zone": zone, "price_eur_mwh": float(v), "source": "ENTSOE"} for t, v in zip(idx, cur)]
    db.upsert_df(eng, db.prices_actual, pd.DataFrame(rows), ["timestamp", "zone"])
    return eng


def test_decoupling_with_near_zero_price_is_explained(tmp_path):
    hu = np.full(96, 250.0)
    ro = hu.copy()
    ro[36:40] = [90, 57, 1, 0]                                       # 09:00-10:00, ca pe 19 nov 2026
    ev = I.day_events(_engine(tmp_path, ro, hu, prev_level=250.0), DAY)
    assert [e["kind"] for e in ev] == ["decouple_low"]
    assert ev[0]["start"] == "09:00" and ev[0]["end"] == "10:00"
    assert "ieftină" in ev[0]["title"] and "orice preț" in ev[0]["text"]


def test_regional_spike_and_day_shift(tmp_path):
    base = np.full(96, 200.0)
    base[70:74] = 700.0                                               # varf in toata regiunea
    ev = I.day_events(_engine(tmp_path, base, base, prev_level=120.0), DAY, forecast_mae=80.0)
    kinds = {e["kind"] for e in ev}
    assert {"spike", "day_shift", "big_miss"} <= kinds and "decouple_low" not in kinds
    spike = next(e for e in ev if e["kind"] == "spike")
    assert "regiunea" in spike["text"]


def test_quiet_day_has_no_events_and_missing_prices_give_none(tmp_path):
    flat = np.full(96, 200.0)
    assert I.day_events(_engine(tmp_path, flat, flat), DAY) == []
    assert I.day_events(_engine(tmp_path, flat, flat), "2030-01-01") == []


def test_segments_merge_short_gaps():
    m = pd.Series([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert I._segments(m) == [(1, 5), (9, 10)]
