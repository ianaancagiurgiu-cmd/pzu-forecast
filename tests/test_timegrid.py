import pandas as pd

from pzu.timegrid import day_index, to_15min


def test_normal_day_has_96_intervals():
    idx = day_index("2026-06-10")
    assert len(idx) == 96 and str(idx.tz) == "UTC"


def test_dst_days():
    assert len(day_index("2026-03-29")) == 92   # trecerea la ora de vara
    assert len(day_index("2026-10-25")) == 100  # trecerea la ora de iarna


def test_hourly_to_15min_extends_last_hour():
    s = pd.Series([1.0, 2.0], index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"))
    out = to_15min(s)
    assert len(out) == 8
    assert out.iloc[:4].eq(1.0).all() and out.iloc[4:].eq(2.0).all()


def test_single_point_series_is_returned_unchanged():
    s = pd.Series([1.0], index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    assert len(to_15min(s)) == 1


def test_mixed_resolution_hourly_values_fill_four_quarters():
    idx = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 01:15",
                            "2026-01-01 01:30", "2026-01-01 01:45"], tz="UTC")
    out = to_15min(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx))
    assert out.iloc[:4].eq(1.0).all()      # ora 00:00 acopera 4 sferturi
    assert list(out.iloc[4:]) == [2.0, 3.0, 4.0, 5.0]


def test_long_gaps_stay_missing():
    idx = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 03:00"], tz="UTC")
    out = to_15min(pd.Series([1.0, 2.0, 3.0], index=idx))
    assert out.loc["2026-01-01 01:30"] != out.loc["2026-01-01 01:30"]  # NaN: gol > 45 min


def test_complete_15min_series_unchanged():
    idx = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    s = pd.Series(range(8), index=idx, dtype=float)
    pd.testing.assert_series_equal(to_15min(s), s, check_freq=False)
