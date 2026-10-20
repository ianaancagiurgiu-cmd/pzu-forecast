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
    assert list(X.columns) == F.FEATURES + F.EXPERIMENTAL + F.GEN_FEATURES + F.OUTAGE_FEATURES and len(X) == 96
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


def _doc(created, start, end, mw=100.0, zone="RO", ptype="Nuclear", business="Unplanned outage", mrid="m1", rev="1", status=None):
    return {"mrid": mrid, "revision": rev, "resource_id": "r", "start": pd.Timestamp(start, tz="UTC"),
            "end": pd.Timestamp(end, tz="UTC"), "zone": zone, "created": pd.Timestamp(created, tz="Europe/Bucharest"),
            "plant_type": ptype, "business_type": business, "doc_status": status, "nominal_mw": mw, "available_mw": 0.0}


def test_outages_only_known_documents_count():
    day = pd.Timestamp("2026-03-10")
    idx = day_index(day)                         # ziua tinta; D = 9 martie, cutoff 9 martie 06:00 local
    docs = pd.DataFrame([
        _doc("2026-03-09 05:59", "2026-03-09", "2026-03-12", mw=100, mrid="known"),
        _doc("2026-03-09 06:01", "2026-03-09", "2026-03-12", mw=500, mrid="late"),          # creat dupa 06:00
        _doc("2026-03-08 10:00", "2026-03-09", "2026-03-12", mw=700, mrid="cancel", status="Cancelled"),
        _doc("2026-03-05 10:00", "2026-03-01", "2026-03-08", mw=900, mrid="over"),          # deja terminat
    ])
    f = F.outage_features(docs, idx)
    assert (f["outage_ro_total_mw"] == 100).all() and (f["outage_ro_nuclear_mw"] == 100).all()
    assert (f["outage_ro_unplanned_mw"] == 100).all()


def test_outages_use_latest_revision_known_at_cutoff():
    idx = day_index("2026-03-10")
    docs = pd.DataFrame([_doc("2026-03-08 10:00", "2026-03-09", "2026-03-12", mw=100, rev="1"),
                         _doc("2026-03-09 05:00", "2026-03-09", "2026-03-12", mw=300, rev="2"),
                         _doc("2026-03-09 08:00", "2026-03-09", "2026-03-12", mw=50, rev="3")])
    assert (F.outage_features(docs, idx)["outage_ro_total_mw"] == 300).all()


def test_outage_window_respects_start_and_end():
    idx = day_index("2026-03-10")                # 00:00 local = 22:00 UTC pe 9 martie
    docs = pd.DataFrame([_doc("2026-03-01 10:00", "2026-03-10 00:00", "2026-03-10 06:00", mw=80, ptype="Fossil Gas",
                              business="Planned maintenance")])
    f = F.outage_features(docs, idx)
    # documentul acopera 00:00-06:00 UTC = 02:00-08:00 local (UTC+2 in martie), capatul final exclus
    th = f["outage_ro_thermal_mw"]
    assert th.iloc[0] == 0 and th.iloc[3 * 4] == 80 and th.iloc[9 * 4] == 0
    assert f["outage_ro_unplanned_mw"].sum() == 0


def test_features_with_extras_pass_dynamic_leakage_check(synth):
    inputs, prices = synth
    gen = pd.DataFrame({"gen_nuclear_mw": 1400.0}, index=prices.index)
    docs = pd.DataFrame([_doc("2026-01-10 10:00", "2026-01-12", "2026-01-30", mw=700)])
    F.verify_no_leakage(inputs, prices, "2026-01-20", gen=gen, outages=docs)
    X = F.build_features(inputs, prices, day_index("2026-01-20"), gen=gen, outages=docs)
    assert (X["gen_nuclear_lag24h"] == 1400).all() and (X["outage_ro_nuclear_mw"] == 700).all()


def test_cancellation_known_at_cutoff_removes_the_outage_but_later_one_does_not():
    idx = day_index("2026-03-10")
    orig = _doc("2026-03-01 10:00", "2026-03-09", "2026-03-12", mw=100, rev="1")
    early_cancel = _doc("2026-03-08 10:00", "2026-03-09", "2026-03-12", mw=100, rev="2", status="Cancelled")
    late_cancel = _doc("2026-03-09 09:00", "2026-03-09", "2026-03-12", mw=100, rev="2", status="Cancelled")
    assert (F.outage_features(pd.DataFrame([orig, early_cancel]), idx)["outage_ro_total_mw"] == 0).all()
    assert (F.outage_features(pd.DataFrame([orig, late_cancel]), idx)["outage_ro_total_mw"] == 100).all()
