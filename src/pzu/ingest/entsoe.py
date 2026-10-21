"""Descarcare ENTSO-E (prin entsoe-py). Toate iesirile: UTC, 15 min.

Documente: A44 preturi day-ahead, A69 forecast solar/eolian, A65 forecast load,
A61 capacitate transfer (NTC) day-ahead.
Nota: preturile vecinilor pentru D+1 NU sunt folosite ca input (se publica dupa gate closure);
aici descarcam doar istoricul lor, din care se calculeaza lag-uri.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

import pandas as pd

from pzu.config import NEIGHBORS, TZ
from pzu.timegrid import to_15min

log = logging.getLogger(__name__)

# Perechi (din, in) pentru coloanele de capacitate
CAPACITY_PAIRS = {
    "capacity_hu_ro_mw": ("HU", "RO"),
    "capacity_ro_hu_mw": ("RO", "HU"),
    "capacity_bg_ro_mw": ("BG", "RO"),
    "capacity_ro_bg_mw": ("RO", "BG"),
}


class _Client(Protocol):  # subset din entsoe.EntsoePandasClient
    def query_day_ahead_prices(self, country_code, start, end): ...
    def query_wind_and_solar_forecast(self, country_code, start, end, **kw): ...
    def query_load_forecast(self, country_code, start, end): ...
    def query_net_transfer_capacity_dayahead(self, country_code_from, country_code_to, start, end): ...


def make_client(token: str) -> _Client:
    from entsoe import EntsoePandasClient
    if not token:
        raise RuntimeError("ENTSOE_TOKEN lipseste (vezi .env.example)")
    return EntsoePandasClient(api_key=token)


def _span(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    s = s.tz_localize(TZ) if s.tzinfo is None else s.tz_convert(TZ)
    e = e.tz_localize(TZ) if e.tzinfo is None else e.tz_convert(TZ)
    return s, e


def _safe(fn, *args, what: str, **kw):
    """Datele lipsa (NoMatchingDataError etc.) nu trebuie sa opreasca pipeline-ul."""
    try:
        return fn(*args, **kw)
    except Exception as exc:  # entsoe-py ridica NoMatchingDataError, HTTP errors
        msg = re.sub(r"securityToken=[^&\s)'\"]+", "securityToken=***", str(exc))
        log.warning("ENTSO-E %s indisponibil: %s", what, msg)
        return None


def fetch_prices(client: _Client, zone: str, start, end) -> pd.Series:
    s, e = _span(start, end)
    raw = _safe(client.query_day_ahead_prices, zone, start=s, end=e, what=f"preturi {zone}")
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name=zone)
    return to_15min(raw.astype(float)).rename(zone)


def fetch_generation_forecast(client: _Client, start, end, zone: str = "RO") -> pd.DataFrame:
    s, e = _span(start, end)
    raw = _safe(client.query_wind_and_solar_forecast, zone, start=s, end=e, psr_type=None,
                what="forecast solar/eolian")
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["solar_forecast_mw", "wind_forecast_mw"])
    out = pd.DataFrame(index=raw.index)
    out["solar_forecast_mw"] = raw["Solar"] if "Solar" in raw else 0.0
    wind_cols = [c for c in raw.columns if str(c).startswith("Wind")]
    out["wind_forecast_mw"] = raw[wind_cols].sum(axis=1) if wind_cols else 0.0
    return to_15min(out.astype(float))


def fetch_load_forecast(client: _Client, start, end, zone: str = "RO") -> pd.DataFrame:
    s, e = _span(start, end)
    raw = _safe(client.query_load_forecast, zone, start=s, end=e, what="forecast load")
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["load_forecast_mw"])
    ser = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
    return to_15min(ser.astype(float).rename("load_forecast_mw")).to_frame()


def fetch_capacities(client: _Client, start, end) -> pd.DataFrame:
    s, e = _span(start, end)
    cols = {}
    for name, (a, b) in CAPACITY_PAIRS.items():
        raw = _safe(client.query_net_transfer_capacity_dayahead, a, b, start=s, end=e,
                    what=f"capacitate {a}->{b}")
        if raw is not None and len(raw):
            cols[name] = to_15min(raw.astype(float).rename(name))
    return pd.DataFrame(cols)


def fetch_all_prices(client: _Client, start, end, zones=("RO", *NEIGHBORS)) -> pd.DataFrame:
    cols = [fetch_prices(client, z, start, end) for z in zones]
    return pd.concat(cols, axis=1) if cols else pd.DataFrame()


def fetch_inputs(client: _Client, start, end) -> pd.DataFrame:
    """Forecast solar/eolian/load + capacitati pentru intervalul dat (date D+1)."""
    parts = [
        fetch_generation_forecast(client, start, end),
        fetch_load_forecast(client, start, end),
        fetch_capacities(client, start, end),
    ]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


# ---------------------------------------------------------------- extra: mix de productie si indisponibilitati
GEN_MAP = {
    "gen_nuclear_mw": ["Nuclear"],
    "gen_hydro_res_mw": ["Hydro Water Reservoir"],
    "gen_hydro_ror_mw": ["Hydro Run-of-river and pondage"],
    "gen_gas_mw": ["Fossil Gas"],
    "gen_coal_mw": ["Fossil Brown coal/Lignite", "Fossil Hard coal"],
}


def fetch_generation_mix(client, start, end, zone: str = "RO") -> pd.DataFrame:
    """Productia REALA pe tipuri (MW). Se publica cu intarziere: in modele se foloseste doar cu lag >= 24h."""
    s, e = _span(start, end)
    raw = _safe(client.query_generation, zone, start=s, end=e, what="productie pe tipuri")
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=list(GEN_MAP))
    if isinstance(raw.columns, pd.MultiIndex):   # (tip, "Actual Aggregated") + (tip, "Actual Consumption")
        raw = raw.loc[:, [c for c in raw.columns if c[1] != "Actual Consumption"]]
        raw.columns = raw.columns.get_level_values(0)
    out = pd.DataFrame(index=raw.index)
    for col, names in GEN_MAP.items():
        have = [n for n in names if n in raw.columns]
        out[col] = raw[have].sum(axis=1, min_count=1) if have else float("nan")
    return to_15min(out.astype(float))


OUTAGE_COLS = {"mrid": "mrid", "revision": "revision", "production_resource_id": "resource_id", "start": "start",
               "end": "end", "plant_type": "plant_type", "businesstype": "business_type", "docstatus": "doc_status",
               "nominal_power": "nominal_mw", "avail_qty": "available_mw"}


def fetch_outages(client, start, end, zone: str = "RO") -> pd.DataFrame:
    """Documente de indisponibilitate a unitatilor de productie (cu momentul crearii)."""
    s, e = _span(start, end)
    raw = _safe(client.query_unavailability_of_generation_units, zone, start=s, end=e, what=f"indisponibilitati {zone}")
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=[*OUTAGE_COLS.values(), "created", "zone"])
    df = raw.rename(columns=OUTAGE_COLS)
    df["created"] = pd.to_datetime(raw.index, utc=True)
    df["zone"] = zone
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["revision"] = df["revision"].astype(str)
    df["doc_status"] = df["doc_status"].astype(object).where(df["doc_status"].notna(), None)
    keep = ["mrid", "revision", "resource_id", "start", "zone", "created", "end", "plant_type", "business_type",
            "doc_status", "nominal_mw", "available_mw"]
    return df[keep].drop_duplicates(["mrid", "revision", "resource_id", "start"]).reset_index(drop=True)


def fetch_actual_solar_wind(client, start, end, zone: str = "RO") -> pd.DataFrame:
    """Productia REALA solara si eoliana (MW). Folosita doar pentru antrenarea modelului vreme -> productie
    si pentru teste de limita superioara; nu exista la ora prognozei."""
    s, e = _span(start, end)
    out = {}
    for col, psr in (("gen_solar_mw", "B16"), ("gen_wind_mw", "B19")):
        raw = _safe(client.query_generation, zone, start=s, end=e, psr_type=psr, what=f"productie reala {psr}")
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.loc[:, [c for c in raw.columns if c[1] != "Actual Consumption"]]
        out[col] = to_15min(raw.iloc[:, 0].astype(float).rename(col))
    return pd.DataFrame(out)
