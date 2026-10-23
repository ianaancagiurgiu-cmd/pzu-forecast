"""Productia reala solara/eoliana a zonelor din regiune (pentru testul de limita superioara).

  python scripts/backfill_region_generation.py --zones HU,BG --months 24
"""
import argparse

import pandas as pd

from pzu import db
from pzu.config import REGION_ZONES, get_settings
from pzu.ingest import entsoe

CODES = {"HU": "HU", "BG": "BG", "RS": "RS", "SK": "SK", **REGION_ZONES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", required=True)
    ap.add_argument("--months", type=int, default=24)
    a = ap.parse_args()
    eng = db.get_engine()
    db.init_db(eng)
    client = entsoe.make_client(get_settings().entsoe_token)
    today = pd.Timestamp.now(tz="Europe/Bucharest").normalize().tz_localize(None)
    start = today - pd.DateOffset(months=a.months)
    for z in a.zones.split(","):
        parts = []
        for s in pd.date_range(start, today, freq="30D"):
            e = min(s + pd.Timedelta(days=30), today + pd.Timedelta(days=1))
            g = entsoe.fetch_actual_solar_wind(client, s, e, zone=CODES[z])
            if len(g):
                parts.append(g)
        if not parts:
            print(f"{z}: fara date", flush=True)
            continue
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="last")].rename(columns={"gen_solar_mw": "solar_mw", "gen_wind_mw": "wind_mw"})
        long = df.rename_axis("timestamp").reset_index()
        long["zone"] = z
        print(f"{z}: {db.upsert_df(eng, db.region_generation, long, ['timestamp', 'zone'])} randuri", flush=True)


if __name__ == "__main__":
    main()
