"""Ajuta informatia din regiune? Piete mai departe (AT, DE, CZ, HR, SI, GR) si prognoza de consum a vecinilor.

  M   modelul de dimineata (RO + lag-uri HU, BG, RS, SK)
  R1  + lag-uri de pret (24h, 7 zile) pentru AT, DE, CZ, HR, SI, GR + media regionala
  R2  + prognoza de consum D+1 a vecinilor (HU, BG, RS, SK)
  R3  R1 + R2

  python scripts/region_experiment.py [--trees 400]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pzu import backtest, db, features as F, value as V
from pzu.config import NEIGHBORS, REGION_ZONES

H24 = pd.Timedelta(hours=24)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    prices = db.read_prices(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), prices)
    idx = X.index
    region = list(REGION_ZONES)
    for z in region:
        s = prices[z] if z in prices else None
        X[f"price_{z.lower()}_lag_24h"] = F._lag(s, idx, H24)
        X[f"price_{z.lower()}_lag_7d"] = F._lag(s, idx, 7 * H24)
    lag24 = X[[f"price_{z.lower()}_lag_24h" for z in region + NEIGHBORS if f"price_{z.lower()}_lag_24h" in X]]
    X["region_mean_lag24h"] = lag24.mean(axis=1)
    X["region_std_lag24h"] = lag24.std(axis=1)
    X["ro_minus_region_lag24h"] = X["price_lag_24h"] - X["region_mean_lag24h"]
    nl = db.read_df(eng, db.neighbor_load, index=None)
    nl["timestamp"] = pd.to_datetime(nl["timestamp"], utc=True)
    nl = nl.pivot(index="timestamp", columns="zone", values="load_forecast_mw")
    for z in NEIGHBORS:
        X[f"load_{z.lower()}"] = nl[z].reindex(idx) if z in nl else np.nan
    R1 = ([f"price_{z.lower()}_{k}" for z in region for k in ("lag_24h", "lag_7d")]
          + ["region_mean_lag24h", "region_std_lag24h", "ro_minus_region_lag24h"])
    R2 = [f"load_{z.lower()}" for z in NEIGHBORS]
    P = F.BASE_PRODUCTION_FEATURES
    variants = {"M": P, "R1": P + R1, "R2": P + R2, "R3": P + R1 + R2}
    print("lipsuri: " + ", ".join(f"{c} {X[c].isna().mean():.2f}" for c in R2 + ["price_de_lag_24h", "price_at_lag_24h"]), flush=True)
    n_days = pd.Series(idx.tz_convert("Europe/Bucharest").date).nunique() - 62
    bt = None
    for name, cols in variants.items():
        part = backtest.walk_forward(X, y, 60, n_days, refit_every=7, cols=cols, n_estimators=a.trees)
        part = part[["y", "pred", "date", "naive_lag24h", "naive_mean7d"]].rename(columns={"pred": name})
        bt = part if bt is None else bt.join(part[name])
        print("gata", name, flush=True)
    bt = bt.dropna()
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)]
    for v in variants:
        bt[v + "+medie7"] = 0.5 * bt[v] + 0.5 * bt["naive_mean7d"]
    cols = ["naive_lag24h", "naive_mean7d", *variants, *[v + "+medie7" for v in variants]]
    daily = V.simulate_battery(bt, {**{c: c for c in cols}, "perfect": "y"}, V.Battery())
    daily.index = pd.to_datetime(daily.index)
    months = daily.index.month
    sel = {"tot": slice(None), "iarna": months.isin([12, 1, 2]), "vara": months.isin([6, 7, 8])}
    mon = pd.Series([d.month for d in pd.to_datetime(bt["date"])], index=bt.index)
    rows = {}
    for c in cols:
        e = (bt["y"] - bt[c]).abs()
        row = {"MAE": e.mean(), "MAE_iarna": e[mon.isin([12, 1, 2])].mean(), "MAE_vara": e[mon.isin([6, 7, 8])].mean()}
        for k, s in sel.items():
            t = daily[s].sum()
            row[f"baterie_%_{k}"] = 100 * t[c] / t["perfect"]
        rows[c] = row
    df = pd.DataFrame(rows).T
    print(f"\n{bt['date'].nunique()} zile")
    print(df.round(1).to_string())
    Path("reports/region_experiment.json").write_text(json.dumps(df.round(2).to_dict("index"), indent=2))


if __name__ == "__main__":
    main()
