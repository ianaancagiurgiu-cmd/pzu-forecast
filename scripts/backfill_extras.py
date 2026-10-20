"""Descarca mixul de productie si indisponibilitatile (RO + vecini) pentru istoric.

  python scripts/backfill_extras.py --months 24
"""
import argparse

import pandas as pd

from pzu import db
from pzu.config import NEIGHBORS, get_settings
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
        print("mix productie", s.date(), flush=True)
        g = entsoe.fetch_generation_mix(client, s, e)
        if not g.empty:
            parts.append(g)
    gen = pd.concat(parts)
    gen = gen[~gen.index.duplicated(keep="last")]
    print("generation_actual:", db.upsert_df(db.get_engine(), db.generation_actual, gen, ["timestamp"]), "randuri")
    # indisponibilitati: documente care se suprapun cu fereastra (inclusiv pornite inainte)
    for zone in ["RO", *NEIGHBORS]:
        docs = entsoe.fetch_outages(client, start - pd.Timedelta(days=60), today + pd.Timedelta(days=60), zone)
        n = db.upsert_df(eng, db.outages, docs, ["mrid", "revision", "resource_id", "start"]) if len(docs) else 0
        print(f"outages {zone}: {n} documente", flush=True)


if __name__ == "__main__":
    main()
