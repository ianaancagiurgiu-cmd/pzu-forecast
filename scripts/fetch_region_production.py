"""Productia REALA eoliana + solara si prognoza de consum D+1 pentru o zona, orar, ~25 de luni -> prod_<zona>.csv.
Pentru testul in doi pasi (vreme -> productie -> pret). Rulat in GitHub Actions (are cheia ENTSO-E).

  python scripts/fetch_region_production.py DE --months 25
"""
import argparse
import time

import pandas as pd
from entsoe import EntsoePandasClient

from pzu.config import get_settings

CODES = {"RO": "RO", "DE": "DE_LU", "PL": "PL", "HU": "HU", "AT": "AT", "CZ": "CZ", "BG": "BG", "SK": "SK"}
PSR = {"solar": "B16", "wind_off": "B18", "wind_on": "B19"}


def retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:   # NoMatchingDataError = nu exista (ex. fara offshore); restul: reincercam
            if type(e).__name__ == "NoMatchingDataError":
                return None
            print("  reincerc:", type(e).__name__, str(e)[:80], flush=True)
            time.sleep(20 * (k + 1))
    return None


def hourly(x):
    if x is None or len(x) == 0:
        return None
    if isinstance(x, pd.DataFrame):
        if isinstance(x.columns, pd.MultiIndex):   # (tip, Actual Aggregated / Actual Consumption)
            x = x.loc[:, [c for c in x.columns if c[1] != "Actual Consumption"]]
        x = x.iloc[:, 0]
    return x.astype(float).tz_convert("UTC").resample("1h").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zone")
    ap.add_argument("--months", type=int, default=25)
    a = ap.parse_args()
    c = EntsoePandasClient(api_key=get_settings().entsoe_token)
    code = CODES[a.zone]
    end = pd.Timestamp.now(tz="Europe/Bucharest").normalize() + pd.Timedelta(days=2)
    start = end - pd.DateOffset(months=a.months)
    cols = {k: [] for k in (*PSR, "load_fc")}
    for s in pd.date_range(start, end, freq="30D"):
        e = min(s + pd.Timedelta(days=30), end)
        for name, psr in PSR.items():
            r = hourly(retry(lambda: c.query_generation(code, start=s, end=e, psr_type=psr)))
            if r is not None:
                cols[name].append(r)
        r = hourly(retry(lambda: c.query_load_forecast(code, start=s, end=e)))
        if r is not None:
            cols["load_fc"].append(r)
        print(a.zone, s.date(), {k: sum(len(v) for v in vs) for k, vs in cols.items()}, flush=True)
    df = pd.DataFrame({k: pd.concat(v)[lambda x: ~x.index.duplicated(keep="last")] for k, v in cols.items() if v})
    df["wind"] = df.get("wind_on", 0).fillna(0) + (df["wind_off"].fillna(0) if "wind_off" in df else 0)
    df.rename_axis("ts_utc").to_csv(f"prod_{a.zone}.csv")
    print(a.zone, "randuri", len(df), df.index.min(), df.index.max(), df.notna().mean().round(2).to_dict())


if __name__ == "__main__":
    main()
