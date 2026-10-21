"""Descarca preturile pietelor din regiune (AT, DE, CZ, HR, SI, GR) si prognoza de consum D+1 a vecinilor.

  python scripts/backfill_region.py --months 24
"""
import argparse

import pandas as pd

from pzu import db
from pzu.config import NEIGHBORS, REGION_ZONES, get_settings
from pzu.ingest import entsoe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--prices", default=",".join(REGION_ZONES), help="zone de pret (virgula); gol = niciuna")
    ap.add_argument("--loads", default=",".join(NEIGHBORS), help="zone pentru prognoza de consum; gol = niciuna")
    a = ap.parse_args()
    eng = db.get_engine()
    db.init_db(eng)
    client = entsoe.make_client(get_settings().entsoe_token)
    today = pd.Timestamp.now(tz="Europe/Bucharest").normalize().tz_localize(None)
    start = today - pd.DateOffset(months=a.months)
    chunks = [(s, min(s + pd.Timedelta(days=30), today + pd.Timedelta(days=2))) for s in pd.date_range(start, today, freq="30D")]
    for code in [z for z in a.prices.split(',') if z]:
        entsoe_code = REGION_ZONES[code]
        parts = [entsoe.fetch_prices(client, entsoe_code, s, e) for s, e in chunks]
        ser = pd.concat([p for p in parts if len(p)])
        ser = ser[~ser.index.duplicated(keep="last")]
        long = ser.rename("price_eur_mwh").rename_axis("timestamp").reset_index()
        long["zone"], long["source"] = code, "ENTSOE"
        print(f"pret {code}: {db.upsert_df(eng, db.prices_actual, long, ['timestamp', 'zone'])} randuri", flush=True)
    for z in [z for z in a.loads.split(',') if z]:
        parts = [entsoe.fetch_load_forecast(client, s, e, zone=z) for s, e in chunks]
        df = pd.concat([p for p in parts if len(p)])
        df = df[~df.index.duplicated(keep="last")]
        long = df["load_forecast_mw"].rename_axis("timestamp").reset_index()
        long["zone"] = z
        print(f"consum {z}: {db.upsert_df(eng, db.neighbor_load, long, ['timestamp', 'zone'])} randuri", flush=True)


if __name__ == "__main__":
    main()
