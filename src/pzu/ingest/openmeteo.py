"""Open-Meteo: vreme pentru D+1 (forecast) si istoric (archive). Fara cheie API.

Media se face pe cateva puncte reprezentative pentru Romania. Gratuit doar pentru uz non-comercial.
"""
from __future__ import annotations

import time

import pandas as pd
import httpx

from pzu.timegrid import to_15min

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# (lat, lon, pondere) - Dobrogea (eolian), sud/sud-est (solar), Bucuresti (consum), Transilvania
POINTS = [
    (44.18, 28.65, 0.30),  # Constanta / Dobrogea
    (44.43, 26.10, 0.30),  # Bucuresti
    (44.32, 23.80, 0.20),  # Craiova / sud-vest
    (46.77, 23.60, 0.20),  # Cluj
]
HOURLY = "temperature_2m,wind_speed_10m,shortwave_radiation"
RENAME = {
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_speed_ms",
    "shortwave_radiation": "radiation_wm2",
}


def _parse(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    idx = pd.to_datetime(h["time"]).tz_localize("UTC")  # timezone=UTC la cerere
    df = pd.DataFrame({new: h[old] for old, new in RENAME.items()}, index=idx)
    return df.astype(float)


def _get(client: httpx.Client, url: str, params: dict, tries: int = 6) -> httpx.Response:
    """GET cu pauze crescatoare la 429 (limita de cereri Open-Meteo)."""
    for k in range(tries):
        try:
            r = client.get(url, params=params)
        except httpx.TransportError:  # timeout / conexiune intrerupta
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1) if SLEEP else 0)
            continue
        if r.status_code != 429 or k == tries - 1:
            r.raise_for_status()
            return r
        time.sleep(min(15 * 2 ** k, 300) if SLEEP else 0)


SLEEP = True


def fetch_weather(start, end, *, archive: bool = False, client: httpx.Client | None = None) -> pd.DataFrame:
    """start/end: date calendaristice (inclusiv). Returneaza media ponderata pe 15 min, UTC."""
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        frames, weights = [], []
        for lat, lon, w in POINTS:
            params = {
                "latitude": lat, "longitude": lon, "hourly": HOURLY,
                "wind_speed_unit": "ms", "timezone": "UTC",
                "start_date": pd.Timestamp(start).date().isoformat(),
                "end_date": pd.Timestamp(end).date().isoformat(),
            }
            r = _get(client, ARCHIVE_URL if archive else FORECAST_URL, params)
            frames.append(_parse(r.json()))
            weights.append(w)
    finally:
        if own:
            client.close()
    total = sum(weights)
    avg = sum(f * (w / total) for f, w in zip(frames, weights))
    return to_15min(avg, how="interpolate")
