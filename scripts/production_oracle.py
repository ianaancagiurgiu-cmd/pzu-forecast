"""Limita superioara a lantului vreme -> productie -> pret: cat ar ajuta preturile daca am cunoaste productia?

  M  modelul de dimineata (productie curenta)
  A  + prognoza operatorilor (ENTSO-E) solar/eolian pentru ziua tinta  [nu exista dimineata]
  O  + productia REALA solara/eoliana a zilei tinta                    [ORACOL: informatie din viitor]

Daca nici oracolul nu aduce un castig mare, nicio prognoza de productie (din vreme sau altfel) nu poate ajuta mult.

  N  + preturile REALE ale vecinilor (HU, BG, RS, SK) din aceeasi zi                [ORACOL: apar dupa 12:00]

  python scripts/production_oracle.py [--trees 400] [--variants M,A,O,N]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pzu import backtest, db, features as F, metrics, value as V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--variants", default="M,A,O")
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    gen = db.read_df(eng, db.generation_actual).apply(pd.to_numeric, errors="coerce")
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng))
    X["act_solar"] = gen["gen_solar_mw"].reindex(X.index)
    X["act_wind"] = gen["gen_wind_mw"].reindex(X.index)
    X["act_resid"] = X["load_forecast_mw"] - X["act_solar"] - X["act_wind"]
    nb = db.read_prices(eng)
    for z in ("HU", "BG", "RS", "SK"):
        X[f"nb_{z}"] = nb[z].reindex(X.index) if z in nb else float("nan")
    keep = X["act_solar"].notna() & X["act_wind"].notna()
    X, y = X[keep], y[keep]
    n_days = pd.Series(X.index.tz_convert("Europe/Bucharest").date).nunique() - 62
    P = F.PRODUCTION_FEATURES
    variants = {"M": P, "A": P + ["solar_forecast_mw", "wind_forecast_mw", "residual_load"],
                "O": P + ["act_solar", "act_wind", "act_resid"],
                "N": P + [f"nb_{z}" for z in ("HU", "BG", "RS", "SK")]}
    variants = {k: v for k, v in variants.items() if k in a.variants.split(",")}
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
    Path(f"reports/production_oracle_{a.variants.replace(',', '')}.json").write_text(json.dumps(df.round(2).to_dict("index"), indent=2))


if __name__ == "__main__":
    main()
