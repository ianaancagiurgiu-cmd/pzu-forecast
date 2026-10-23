"""Limita superioara pentru PRODUCTIA REGIONALA: cat ar ajuta daca am sti solarul/eolianul de maine din toata regiunea?

  B   modelul de dimineata (productia actuala)
  G1  + productia reala solara/eoliana a vecinilor si a pietelor din regiune (fara Romania)   [ORACOL]
  G2  + si productia reala a Romaniei                                                       [ORACOL]

  python scripts/region_oracle.py [--trees 400]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pzu import backtest, db, features as F, value as V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng), nload=db.read_neighbor_load(eng))
    rg = db.read_df(eng, db.region_generation, index=None)
    rg["timestamp"] = pd.to_datetime(rg["timestamp"], utc=True)
    for c in ("solar_mw", "wind_mw"):
        rg[c] = pd.to_numeric(rg[c], errors="coerce")
    zones = sorted(rg["zone"].unique())
    cols_reg = []
    for kind in ("solar_mw", "wind_mw"):
        wide = rg.pivot(index="timestamp", columns="zone", values=kind).reindex(X.index)
        for z in zones:
            X[f"reg_{kind[:-3]}_{z}"] = wide[z]
            cols_reg.append(f"reg_{kind[:-3]}_{z}")
        X[f"reg_{kind[:-3]}_total"] = wide.sum(axis=1, min_count=1)
        cols_reg.append(f"reg_{kind[:-3]}_total")
    X["reg_res_total"] = X["reg_solar_total"].fillna(0) + X["reg_wind_total"].fillna(0)
    cols_reg.append("reg_res_total")
    gen = db.read_df(eng, db.generation_actual).apply(pd.to_numeric, errors="coerce")
    X["act_solar"] = gen["gen_solar_mw"].reindex(X.index)
    X["act_wind"] = gen["gen_wind_mw"].reindex(X.index)
    keep = X["act_solar"].notna() & X["act_wind"].notna()
    X, y = X[keep], y[keep]
    print("zone cu productie:", zones, "| acoperire:", {c: round(float(X[c].notna().mean()), 2) for c in ("reg_solar_total", "reg_wind_total")}, flush=True)
    base = [c for c in F.PRODUCTION_FEATURES if X[c].notna().mean() > 0.5]
    variants = {"B": base, "G1": base + cols_reg, "G2": base + cols_reg + ["act_solar", "act_wind"]}
    n_days = pd.Series(X.index.tz_convert("Europe/Bucharest").date).nunique() - 62
    bt = None
    for name, cols in variants.items():
        part = backtest.walk_forward(X, y, 60, n_days, refit_every=7, cols=cols, n_estimators=a.trees, n_jobs=a.threads)
        part = part[["y", "pred", "date", "naive_lag24h", "naive_mean7d"]].rename(columns={"pred": name})
        bt = part if bt is None else bt.join(part[name])
        print("gata", name, flush=True)
    bt = bt.dropna()
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)]
    for v in variants:
        bt[v + "+medie7"] = 0.5 * bt[v] + 0.5 * bt["naive_mean7d"]
    allc = ["naive_lag24h", "naive_mean7d", *variants, *[v + "+medie7" for v in variants]]
    daily = V.simulate_battery(bt, {**{c: c for c in allc}, "perfect": "y"}, V.Battery())
    daily.index = pd.to_datetime(daily.index)
    mon = daily.index.month
    sel = {"tot": slice(None), "iarna": mon.isin([12, 1, 2]), "vara": mon.isin([6, 7, 8])}
    mm = pd.Series([d.month for d in pd.to_datetime(bt["date"])], index=bt.index)
    rows = {}
    for c in allc:
        e = (bt["y"] - bt[c]).abs()
        row = {"MAE": e.mean(), "MAE_iarna": e[mm.isin([12, 1, 2])].mean(), "MAE_vara": e[mm.isin([6, 7, 8])].mean()}
        for k, s in sel.items():
            t = daily[s].sum()
            row[f"baterie_%_{k}"] = 100 * t[c] / t["perfect"]
        rows[c] = row
    df = pd.DataFrame(rows).T
    print(f"\n{bt['date'].nunique()} zile")
    print(df.round(1).to_string())
    Path("reports/region_oracle.json").write_text(json.dumps(df.round(2).to_dict("index"), indent=2))


if __name__ == "__main__":
    main()
