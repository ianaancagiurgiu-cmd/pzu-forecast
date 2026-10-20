"""Copiaza datele dintr-o DB in alta (ex. SQLite local -> Postgres/Supabase).

  python scripts/migrate_db.py --src sqlite:///pzu.db --dst postgresql+psycopg://user:pass@host:5432/db
"""
import argparse

import pandas as pd
from sqlalchemy import select

from pzu import db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sqlite:///pzu.db")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = db.get_engine(a.src), db.get_engine(a.dst)
    db.init_db(dst)
    keys = {"prices_actual": ["timestamp", "zone"], "forecasts_input": ["timestamp"],
            "price_forecast": ["timestamp"], "forecast_accuracy": ["forecast_date"]}
    for table in (db.prices_actual, db.forecasts_input, db.price_forecast, db.forecast_accuracy):
        with src.connect() as conn:
            df = pd.read_sql(select(table), conn)
        for col in df.columns:
            if "timestamp" in col or col.endswith("_at"):
                df[col] = pd.to_datetime(df[col], utc=True)
        n = db.upsert_df(dst, table, df, keys[table.name])
        print(f"{table.name}: {n} randuri")
    row = db.load_model_bytes(src)
    if row:
        db.save_model_bytes(dst, row[0], row[1])
        print(f"model: {row[0]}")


if __name__ == "__main__":
    main()
