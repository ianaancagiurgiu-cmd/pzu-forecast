"""Documentele de indisponibilitate (unitati de productie A80 + centrale A77) pentru o zona, ~26 de luni -> outages_<zona>.csv.
Se pastreaza momentul crearii fiecarei revizii, ca testul sa foloseasca doar ce se stia la ora prognozei.
Rulat in GitHub Actions (are cheia ENTSO-E).

  python scripts/fetch_outages_csv.py RO --months 26
"""
import argparse
import time

import pandas as pd
from entsoe import EntsoePandasClient

from pzu.config import get_settings

KEEP = {"mrid": "mrid", "revision": "revision", "production_resource_id": "resource_id", "production_resource_name": "name",
        "start": "start", "end": "end", "plant_type": "plant_type", "businesstype": "business_type",
        "docstatus": "doc_status", "nominal_power": "nominal_mw", "avail_qty": "available_mw"}


def query(fn, zone, s, e, tries=4):
    for k in range(tries):
        try:
            return fn(zone, start=s, end=e)
        except Exception as ex:
            if type(ex).__name__ == "NoMatchingDataError":
                return None
            print("  reincerc:", type(ex).__name__, str(ex)[:100], flush=True)
            time.sleep(20 * (k + 1))
    return None


def norm(raw, zone, kind):
    if raw is None or len(raw) == 0:
        return None
    df = raw.copy()
    created = df["created_doc_time"] if "created_doc_time" in df else df.index.to_series()
    df = df.rename(columns=KEEP)
    df["created"] = pd.to_datetime(created.to_numpy(), utc=True)
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["zone"], df["kind"] = zone, kind
    cols = [c for c in (*KEEP.values(), "created", "zone", "kind") if c in df]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zone")
    ap.add_argument("--months", type=int, default=26)
    a = ap.parse_args()
    c = EntsoePandasClient(api_key=get_settings().entsoe_token)
    end = pd.Timestamp.now(tz="Europe/Bucharest").normalize() + pd.Timedelta(days=60)
    start = end - pd.DateOffset(months=a.months)
    parts = []
    for s in pd.date_range(start, end, freq="90D"):
        e = min(s + pd.Timedelta(days=90), end)
        for kind, fn in (("A80", c.query_unavailability_of_generation_units), ("A77", c.query_unavailability_of_production_units)):
            r = norm(query(fn, a.zone, s, e), a.zone, kind)
            if r is not None:
                parts.append(r)
                if len(parts) == 1:
                    print("coloane:", list(r.columns), flush=True)
        print(a.zone, s.date(), sum(len(p) for p in parts), flush=True)
    if not parts:
        print(a.zone, "fara documente"); pd.DataFrame().to_csv(f"outages_{a.zone}.csv"); return
    df = pd.concat(parts).drop_duplicates(["mrid", "revision", "resource_id", "start", "kind"])
    df.to_csv(f"outages_{a.zone}.csv", index=False)
    print(a.zone, "documente", len(df), "| tipuri:", df["plant_type"].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
