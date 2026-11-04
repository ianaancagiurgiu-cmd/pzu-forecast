"""Descarca istoricul zilnic al pretului gazului TTF (Yahoo Finance, gratuit) -> gas_ttf.csv.
Rulat in GitHub Actions (sesiunea de dezvoltare nu are acces la acest domeniu)."""
import sys

import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=history"


def fetch(sym: str = "TTF=F") -> pd.Series:
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    r = requests.get(URL.format(sym=sym, p1=now - 3 * 365 * 86400, p2=now + 86400), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("Europe/Bucharest").normalize().tz_localize(None)
    return pd.Series(res["indicators"]["quote"][0]["close"], index=idx, name="ttf_eur_mwh").dropna()


if __name__ == "__main__":
    s = fetch()
    s.rename_axis("date").to_csv("gas_ttf.csv")
    print(len(s), s.index.min().date(), s.index.max().date(), "ultima:", round(float(s.iloc[-1]), 1))
    sys.exit(0 if len(s) > 400 else 1)
