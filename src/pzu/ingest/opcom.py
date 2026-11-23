"""OPCOM (bursa care organizeaza PZU): sursa oficiala a pretului Romaniei, folosita cand ENTSO-E nu publica.

Raportul public "Rezultate cuplare PZU" (fara cont) da, pe intervale de 15 minute: pretul RO in EUR/MWh, fluxurile
transfrontaliere RO<->HU si RO<->BG si capacitatea disponibila (ATC) pe granita RO-BG. Intervalele sunt numerotate in
ora pietei (CET/CEST): intervalul 1 incepe la 00:00 CET, adica 01:00 ora Romaniei.
"""
from __future__ import annotations

import csv
import io
import logging
import time

import httpx
import pandas as pd

log = logging.getLogger("opcom")
_blocked = False   # OPCOM refuza unele retele (ex. serverele GitHub Actions, HTTP 403): nu mai insistam in aceeasi rulare
URL = "https://www.opcom.ro/rapoarte-pzu-raportMarketResults-export-csv/{d:%d/%m/%Y}/ro"
MARKET_TZ = "Europe/Brussels"
COLS = ["price_eur", "flow_ro_hu", "flow_hu_ro", "flow_ro_bg", "flow_bg_ro", "atc_ro_bg", "atc_bg_ro"]


def _num(x: str) -> float:
    x = (x or "").strip().replace(",", "")
    return float(x) if x else float("nan")


def parse_market_results(text: str, day) -> pd.DataFrame:
    """CSV-ul OPCOM -> DataFrame cu index UTC (inceputul fiecarui interval) si coloanele COLS."""
    start = pd.Timestamp(pd.Timestamp(day).date(), tz=MARKET_TZ)
    rows = {}
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 8 or not r[0].strip().isdigit():
            continue
        n = int(r[0])
        rows[(start + pd.Timedelta(minutes=15 * (n - 1))).tz_convert("UTC")] = [_num(v) for v in r[1:8]]
    if not rows:
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLS).sort_index().rename_axis("timestamp")


def fetch_market_results(day, client: httpx.Client | None = None, tries: int = 3) -> pd.DataFrame:
    """Rezultatele PZU pentru ziua de livrare `day` (dupa publicare, ~13:00 ora Romaniei in ziua dinainte)."""
    global _blocked
    if _blocked:
        return pd.DataFrame(columns=COLS)
    own = client is None
    client = client or httpx.Client(timeout=40, follow_redirects=True)
    try:
        for k in range(tries):
            try:
                r = client.get(URL.format(d=pd.Timestamp(day)))
                if r.status_code == 200:
                    return parse_market_results(r.text, day)
                log.warning("OPCOM %s: HTTP %s", pd.Timestamp(day).date(), r.status_code)
                if r.status_code == 403:   # acces refuzat pentru reteaua aceasta: nu ocolim, renuntam
                    _blocked = True
                    break
            except httpx.TransportError as exc:
                log.warning("OPCOM %s: %s", pd.Timestamp(day).date(), exc)
            time.sleep(5 * (k + 1))
    finally:
        if own:
            client.close()
    return pd.DataFrame(columns=COLS)


SNAPSHOT = "data/opcom_results.csv"


def load_snapshot(path: str = SNAPSHOT) -> pd.DataFrame:
    """Rezultate OPCOM salvate in repository (descarcate de unde OPCOM permite accesul); gol daca lipsesc."""
    import os
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")[COLS].sort_index()
