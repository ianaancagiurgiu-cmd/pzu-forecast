"""Descarca istoricul (implicit 24 luni: minim 12 pentru a acoperi toate sezoanele) si il salveaza in DB.

  python scripts/backfill.py --months 24           # date reale (necesita ENTSOE_TOKEN)
  python scripts/backfill.py --synthetic --days 200 # date sintetice, pentru dezvoltare
"""
import argparse
import json
import logging

import pandas as pd

from pzu import db, quality
from pzu.config import get_settings
from pzu.ingest import entsoe, openmeteo
from pzu.timegrid import day_index

log = logging.getLogger("backfill")


def to_long_prices(wide: pd.DataFrame, source: str) -> pd.DataFrame:
    long = wide.stack().rename("price_eur_mwh").reset_index()
    long.columns = ["timestamp", "zone", "price_eur_mwh"]
    long["source"] = source
    return long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--chunk-days", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = db.get_engine()
    db.init_db(engine)
    today = pd.Timestamp.now(tz="Europe/Bucharest").normalize().tz_localize(None)

    if args.synthetic:
        from pzu.synthetic import generate
        start = today - pd.Timedelta(days=args.days)
        inputs, prices = generate(str(start.date()), args.days)
        source = "SYNTHETIC"
    else:
        start = today - pd.DateOffset(months=args.months)
        client = entsoe.make_client(get_settings().entsoe_token)
        in_parts, pr_parts = [], []
        for s in pd.date_range(start, today, freq=f"{args.chunk_days}D"):
            e = min(s + pd.Timedelta(days=args.chunk_days), today + pd.Timedelta(days=1))
            log.info("ENTSO-E %s -> %s", s.date(), e.date())
            in_parts.append(entsoe.fetch_inputs(client, s, e))
            pr_parts.append(entsoe.fetch_all_prices(client, s, e))
        inputs = pd.concat([p for p in in_parts if not p.empty]).sort_index()
        prices = pd.concat([p for p in pr_parts if not p.empty]).sort_index()
        inputs = inputs[~inputs.index.duplicated(keep="last")]
        prices = prices[~prices.index.duplicated(keep="last")]
        weather = openmeteo.fetch_weather(start, today - pd.Timedelta(days=1), archive=True)
        inputs = inputs.join(weather, how="outer")
        source = "ENTSOE"

    db.upsert_df(engine, db.prices_actual, to_long_prices(prices, source), ["timestamp", "zone"])
    inputs = inputs[[c for c in db.INPUT_COLS if c in inputs.columns]]
    db.upsert_df(engine, db.forecasts_input, inputs, ["timestamp"])

    last = prices.index.max().tz_convert("Europe/Bucharest").tz_localize(None).normalize()
    print(json.dumps(quality.summary(prices, start, last), indent=2, default=str))


if __name__ == "__main__":
    main()
