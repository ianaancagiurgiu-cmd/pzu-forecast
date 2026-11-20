"""Diagnostic: cate intervale publica ENTSO-E pentru fiecare zona, pe zile (folosit cand lipsesc preturi oficiale)."""
import sys
import pandas as pd
from entsoe import EntsoePandasClient
from pzu.config import get_settings
from pzu.timegrid import day_index

c = EntsoePandasClient(api_key=get_settings().entsoe_token)
days = sys.argv[1:] or ["2026-11-19", "2026-11-20", "2026-11-21"]
zones = {"RO": "RO", "HU": "HU", "BG": "BG", "RS": "RS", "SK": "SK", "AT": "AT", "DE": "DE_LU", "HR": "HR", "SI": "SI", "GR": "GR", "CZ": "CZ"}
for d in days:
    idx = day_index(d)
    start, end = pd.Timestamp(d, tz="Europe/Bucharest"), pd.Timestamp(d, tz="Europe/Bucharest") + pd.Timedelta(days=1)
    row = {}
    for name, code in zones.items():
        try:
            s = c.query_day_ahead_prices(code, start=start, end=end)
            n = int(s.reindex(idx).notna().sum()) if len(s) else 0
            row[name] = f"{n}/{len(idx)}" + (f" ({s.index.min().tz_convert('Europe/Bucharest'):%H:%M}-{s.index.max().tz_convert('Europe/Bucharest'):%H:%M}, pas {pd.Series(s.index).diff().dropna().mode().iloc[0]})" if len(s) else "")
        except Exception as e:
            row[name] = f"EROARE {type(e).__name__}: {str(e)[:70]}"
    print(d)
    for k, v in row.items():
        print("  ", k, v)
