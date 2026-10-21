"""Descarca productia REALA solara si eoliana (ENTSO-E) pentru istoric, in generation_actual.

  python scripts/backfill_production.py --months 24
"""
import argparse

import pandas as pd

from pzu import db
from pzu.config import get_settings
from pzu.ingest import entsoe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    a = ap.parse_args()
    eng = db.get_engine()
    db.init_db(eng)
    client = entsoe.make_client(get_settings().entsoe_token)
    today = pd.Timestamp.now(tz="Europe/Bucharest").normalize().tz_localize(None)
    start = today - pd.DateOffset(months=a.months)
    parts = []
    for s in pd.date_range(start, today, freq="30D"):
        e = min(s + pd.Timedelta(days=30), today + pd.Timedelta(days=1))
        print("productie solar/eolian", s.date(), flush=True)
        g = entsoe.fetch_actual_solar_wind(client, s, e)
        if not g.empty:
            parts.append(g)
    gen = pd.concat(parts)
    gen = gen[~gen.index.duplicated(keep="last")]
    existing = db.read_df(eng, db.generation_actual)
    merged = gen.combine_first(existing[[c for c in existing.columns if c not in gen.columns]]) if not existing.empty else gen
    # pastram coloanele existente (mix) si adaugam solar/eolian
    full = existing.join(gen, how="outer", rsuffix="_new") if not existing.empty else gen
    for c in gen.columns:
        if c + "_new" in full:
            full[c] = full[c + "_new"].combine_first(full[c]) if c in full else full[c + "_new"]
            full = full.drop(columns=c + "_new")
    print("generation_actual:", db.upsert_df(eng, db.generation_actual, full, ["timestamp"]), "randuri")
    print(full[["gen_solar_mw", "gen_wind_mw"]].describe().round(0).to_string())


if __name__ == "__main__":
    main()
