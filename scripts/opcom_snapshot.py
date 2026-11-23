"""Descarca rezultatele PZU de la OPCOM (pret RO in EUR, fluxuri, ATC) si le adauga in data/opcom_results.csv.

Ruleaza zilnic de unde OPCOM permite accesul (serverele GitHub Actions primesc HTTP 403). Pipeline-ul din GitHub
foloseste apoi fisierul ca sa completeze preturile RO pe care ENTSO-E nu le-a publicat.

  python scripts/opcom_snapshot.py [--days 7]

Iesire: un rand pe zi de piata (CET), cu numarul de intervale; codul de iesire 2 daca ziua de maine lipseste.
"""
import argparse
import sys
from datetime import timedelta

import pandas as pd

from pzu.ingest import opcom


def expected(day) -> int:
    """Intervale intr-o zi de piata CET (92/100 la schimbarea orei)."""
    a = pd.Timestamp(day, tz=opcom.MARKET_TZ)
    b = pd.Timestamp(pd.Timestamp(day) + timedelta(days=1), tz=opcom.MARKET_TZ)
    return len(pd.date_range(a, b, freq="15min", inclusive="left"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="cate zile in urma se verifica (pe langa ziua de maine)")
    a = ap.parse_args()
    today = pd.Timestamp.now(tz="Europe/Bucharest").date()
    days = [today - timedelta(days=k) for k in range(a.days, -2, -1)]   # ... azi, maine
    old = opcom.load_snapshot()
    parts, status = [old] if not old.empty else [], {}
    for d in days:
        res = opcom.fetch_market_results(d)
        n, exp = len(res.dropna(subset=["price_eur"])), expected(d)
        status[d] = (n, exp)
        if n:
            parts.append(res)
    if parts:
        allr = pd.concat(parts)
        allr = allr[~allr.index.duplicated(keep="last")].sort_index()
        out = allr.reset_index()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(opcom.SNAPSHOT, index=False)
    for d, (n, exp) in status.items():
        print(f"{d}: {n}/{exp} intervale" + ("" if n == exp else "  <-- INCOMPLET"))
    tomorrow = today + timedelta(days=1)
    n, exp = status[tomorrow]
    if n < exp:
        print(f"ATENTIE: rezultatele PZU pentru {tomorrow} nu sunt (complet) publicate pe OPCOM")
        return 2
    print(f"OK: rezultatele pentru {tomorrow} sunt publicate (medie {pd.concat(parts).loc[pd.Timestamp(tomorrow, tz=opcom.MARKET_TZ).tz_convert('UTC'):].price_eur.head(exp).mean():.1f} EUR/MWh)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
