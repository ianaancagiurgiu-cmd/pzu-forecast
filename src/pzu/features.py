"""Feature engineering pentru intervalele de 15 min.

Regula anti-leakage: la ora prognozei (dimineata zilei D) stim preturile pana la sfarsitul zilei D
(publicate in D-1 dupa 12:45) si forecast-urile D+1 (solar/eolian/load/capacitati/vreme).
NU stim preturile D+1, nici ale Romaniei, nici ale vecinilor. De aceea toate feature-urile de pret
sunt lag-uri >= 24h, iar preturile vecinilor intra ca lag-uri, nu ca valori D+1.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from pzu.calendar_ro import is_holiday
from pzu.config import NEIGHBORS
from pzu.timegrid import day_index, to_local

H24 = pd.Timedelta(hours=24)

# nume -> (tip, lag minim in ore). Tipuri: 'forecast' (cunoscut inainte de gate), 'calendar', 'lag'.
FEATURE_SPECS: dict[str, tuple[str, int]] = {
    "solar_forecast_mw": ("forecast", 0),
    "wind_forecast_mw": ("forecast", 0),
    "load_forecast_mw": ("forecast", 0),
    "residual_load": ("forecast", 0),
    # capacity_hu_ro_mw / capacity_ro_hu_mw lipsesc intentionat: ENTSO-E nu publica NTC pentru
    # granita RO-HU (zona Core, flow-based). Se pot adauga daca apare o alta sursa.
    "capacity_bg_ro_mw": ("forecast", 0),
    "capacity_ro_bg_mw": ("forecast", 0),
    "temp_c": ("forecast", 0),
    "wind_speed_ms": ("forecast", 0),
    "radiation_wm2": ("forecast", 0),
    "hour": ("calendar", 0),
    "quarter": ("calendar", 0),
    "day_of_week": ("calendar", 0),
    "is_holiday": ("calendar", 0),
    "price_lag_24h": ("lag", 24),
    "price_lag_7d": ("lag", 168),
    "price_rolling_mean_7d": ("lag", 24),
    **{f"price_{z.lower()}_lag_24h": ("lag", 24) for z in NEIGHBORS},
    **{f"price_{z.lower()}_lag_7d": ("lag", 168) for z in NEIGHBORS},
}
FEATURES = list(FEATURE_SPECS)

# Experimentale: calculate mereu, dar NEfolosite de modelul de productie. In scripts/experiments.py nu au
# imbunatatit valoarea pe baterie nici pe perioada de dezvoltare, nici pe holdout (vezi reports/experiments.json).
EXPERIMENTAL_SPECS: dict[str, tuple[str, int]] = {
    "price_lag_48h": ("lag", 48),
    "price_lag_14d": ("lag", 336),
    # statistici ale zilei precedente (D): toata ziua D e cunoscuta la ora prognozei
    "prev_day_mean": ("lag", 24),
    "prev_day_min": ("lag", 24),
    "prev_day_max": ("lag", 24),
    "prev_day_std": ("lag", 24),
    "lag24_dev": ("lag", 24),   # pretul de ieri la aceeasi ora fata de media zilei de ieri (forma zilei)
}
EXPERIMENTAL = list(EXPERIMENTAL_SPECS)

# Date suplimentare ENTSO-E (productie pe tipuri + indisponibilitati). Pentru productie: lag 24h.
# Pentru indisponibilitati: doar documentele create inainte de 06:00 din ziua D (D = ziua dinaintea celei prognozate).
GEN_FEATURES = [c.replace("_mw", "_lag24h") for c in ("gen_nuclear_mw", "gen_hydro_res_mw", "gen_hydro_ror_mw",
                                                      "gen_gas_mw", "gen_coal_mw")]
OUTAGE_FEATURES = ["outage_ro_total_mw", "outage_ro_nuclear_mw", "outage_ro_hydro_mw", "outage_ro_thermal_mw",
                   "outage_ro_unplanned_mw", "outage_hu_total_mw", "outage_bg_total_mw"]
OUTAGE_CUTOFF_HOURS = 6   # ora locala din ziua D pana la care documentele sunt considerate cunoscute
INPUT_FEATURES = [f for f, (kind, _) in FEATURE_SPECS.items() if kind == "forecast" and f != "residual_load"]


def assert_no_leakage(specs: dict[str, tuple[str, int]] = {**FEATURE_SPECS, **EXPERIMENTAL_SPECS}) -> None:
    """Verificare statica: niciun feature de pret nu se uita mai aproape de 24h."""
    bad = [n for n, (kind, lag) in specs.items() if kind == "lag" and lag < 24]
    if bad:
        raise AssertionError(f"Feature-uri cu risc de data leakage (lag < 24h): {bad}")


def _lag(s: pd.Series | None, index: pd.DatetimeIndex, delta: pd.Timedelta) -> np.ndarray:
    if s is None:
        return np.full(len(index), np.nan)
    return s.reindex(index - delta).to_numpy(dtype=float)


def _prev_day_stats(ro: pd.Series | None, local_index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """mean/min/max/std ale pretului RO din ziua locala precedenta fiecarui interval."""
    n = len(local_index)
    if ro is None or ro.dropna().empty:
        return {k: np.full(n, np.nan) for k in ("mean", "min", "max", "std")}
    r = ro.dropna()
    by_day = r.groupby(pd.Index(r.index.tz_convert(local_index.tz).date)).agg(["mean", "min", "max", "std"])
    prev = [d - timedelta(days=1) for d in local_index.date]
    got = by_day.reindex(prev)
    return {k: got[k].to_numpy(dtype=float) for k in ("mean", "min", "max", "std")}


def _category(plant_type: str | None, business: str | None) -> set[str]:
    cats = set()
    pt = (plant_type or "").lower()
    if "nuclear" in pt:
        cats.add("nuclear")
    if "hydro" in pt:
        cats.add("hydro")
    if "fossil" in pt:
        cats.add("thermal")
    if "unplanned" in (business or "").lower():
        cats.add("unplanned")
    return cats


def outage_features(outages: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    """MW indisponibili la fiecare interval, folosind DOAR documentele create inainte de 06:00 din ziua D
    (D = ziua locala dinaintea zilei intervalului); la mai multe revizii se ia ultima cunoscuta atunci."""
    cols = {c: np.full(len(index), np.nan) for c in OUTAGE_FEATURES}
    if outages is None or outages.empty:
        return pd.DataFrame(cols, index=index)
    docs = outages.copy()
    bad = docs["doc_status"].fillna("").str.lower().str.contains("cancel|withdraw")
    docs = docs[~bad].copy()
    docs["unavail"] = (docs["nominal_mw"] - docs["available_mw"]).clip(lower=0).fillna(0)
    docs["rev"] = pd.to_numeric(docs["revision"], errors="coerce").fillna(0)
    docs["cats"] = [_category(p, b) for p, b in zip(docs["plant_type"], docs["business_type"])]
    local = to_local(index)
    pos = np.arange(len(index))
    days = pd.Series(local.date, index=pos)
    for day, ix in days.groupby(days).groups.items():
        day_start = pd.Timestamp(day, tz=local.tz)
        cutoff = day_start - pd.Timedelta(hours=24 - OUTAGE_CUTOFF_HOURS)
        t = index[list(ix)]
        known = docs[(docs["created"] <= cutoff) & (docs["end"] > t.min()) & (docs["start"] <= t.max())]
        if known.empty:
            for c in cols.values():
                c[list(ix)] = 0.0
            continue
        known = known[known["rev"] == known.groupby("mrid")["rev"].transform("max")]
        acc = {c: np.zeros(len(t)) for c in OUTAGE_FEATURES}
        for r in known.itertuples():
            m = (t >= r.start) & (t < r.end)
            if not m.any():
                continue
            z = r.zone.lower()
            if z == "ro":
                acc["outage_ro_total_mw"][m] += r.unavail
                for cat in r.cats:
                    acc[f"outage_ro_{cat}_mw"][m] += r.unavail
            elif z in ("hu", "bg"):
                acc[f"outage_{z}_total_mw"][m] += r.unavail
        for c, v in acc.items():
            cols[c][list(ix)] = v
    return pd.DataFrame(cols, index=index)


def build_features(inputs: pd.DataFrame, prices: pd.DataFrame, index: pd.DatetimeIndex,
                   gen: pd.DataFrame | None = None, outages: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matricea de feature-uri pentru `index` (UTC). `prices`: format lat, coloane = zone."""
    assert_no_leakage()
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)
    for col in INPUT_FEATURES:
        out[col] = inputs[col].reindex(index) if col in inputs else np.nan
    out["residual_load"] = out["load_forecast_mw"] - out["solar_forecast_mw"] - out["wind_forecast_mw"]

    local = to_local(index)
    out["hour"] = local.hour
    out["quarter"] = local.minute // 15
    out["day_of_week"] = local.dayofweek
    out["is_holiday"] = is_holiday(index).to_numpy().astype(int)

    ro = prices["RO"] if "RO" in prices else None
    out["price_lag_24h"] = _lag(ro, index, H24)
    out["price_lag_7d"] = _lag(ro, index, 7 * H24)
    if ro is not None:
        stack = np.vstack([_lag(ro, index, k * H24) for k in range(1, 8)])
        valid = (~np.isnan(stack)).sum(axis=0)
        mean = np.nanmean(np.where(np.isnan(stack), 0, stack), axis=0) * 7 / np.maximum(valid, 1)
        out["price_rolling_mean_7d"] = np.where(valid >= 3, mean, np.nan)
    else:
        out["price_rolling_mean_7d"] = np.nan
    out["price_lag_48h"] = _lag(ro, index, 2 * H24)
    out["price_lag_14d"] = _lag(ro, index, 14 * H24)
    stats = _prev_day_stats(ro, local)
    for name in ("mean", "min", "max", "std"):
        out[f"prev_day_{name}"] = stats[name]
    out["lag24_dev"] = out["price_lag_24h"] - out["prev_day_mean"]
    for z in NEIGHBORS:
        s = prices[z] if z in prices else None
        out[f"price_{z.lower()}_lag_24h"] = _lag(s, index, H24)
        out[f"price_{z.lower()}_lag_7d"] = _lag(s, index, 7 * H24)
    for col in GEN_FEATURES:
        src = col.replace("_lag24h", "_mw")
        out[col] = _lag(gen[src] if gen is not None and src in gen else None, index, H24)
    out = out.join(outage_features(outages, index))
    return out[FEATURES + EXPERIMENTAL + GEN_FEATURES + OUTAGE_FEATURES]


def verify_no_leakage(inputs: pd.DataFrame, prices: pd.DataFrame, target_day, **extra) -> None:
    """Test dinamic: matricea pentru ziua tinta trebuie sa fie identica daca ascundem
    toate preturile de la inceputul zilei tinta incolo."""
    idx = day_index(target_day)
    full = build_features(inputs, prices, idx, **extra)
    cut_extra = dict(extra)
    if extra.get("gen") is not None:
        cut_extra["gen"] = extra["gen"][extra["gen"].index < idx[0]]
    cut = build_features(inputs, prices[prices.index < idx[0]], idx, **cut_extra)
    pd.testing.assert_frame_equal(full, cut)


def build_training_set(inputs: pd.DataFrame, prices: pd.DataFrame, dropna_target: bool = True,
                       gen: pd.DataFrame | None = None, outages: pd.DataFrame | None = None):
    """(X, y) pe toate intervalele unde exista pret RO si forecast-uri de intrare."""
    idx = inputs.index
    if "RO" not in prices:
        raise ValueError("Lipsesc preturile RO")
    y = prices["RO"].reindex(idx)
    X = build_features(inputs, prices, idx, gen=gen, outages=outages)
    mask = y.notna() & X["residual_load"].notna() & X["price_lag_24h"].notna()
    if not dropna_target:
        return X, y
    return X[mask], y[mask]
