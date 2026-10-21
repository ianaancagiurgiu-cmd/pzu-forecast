"""Testeaza imbunatatiri pe TOT istoricul (toate anotimpurile), cu antrenare pe 60 de zile si reantrenare la 7 zile.

Variante:
  V0  modelul de productie (nivel direct)
  V1  V0 + corectie cu erorile ultimelor 7 zile la aceeasi ora
  V2  modele separate pe zone orare (noapte, dimineata, pranz, seara, tarziu)
  V3  doua etape: nivelul zilei (regresie simpla) + forma zilei (LightGBM pe deviatia fata de media zilei)
  V4  forma zilei din V3 + nivelul dat de media pe 7 zile (testeaza doar partea de forma)

  python scripts/improvements.py [--trees 400]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pzu import db, features as F, metrics, model as M, value as V
from pzu.config import TZ

BLOCKS = {"noapte": (0, 6), "dimineata": (6, 10), "pranz": (10, 16), "seara": (16, 22), "tarziu": (22, 24)}
DEV_SRC = ["load_forecast_mw", "price_rolling_mean_7d", "price_lag_24h", "price_lag_7d"]
SHAPE_COLS = F.PRODUCTION_FEATURES + [c + "_dev" for c in DEV_SRC]
LEVEL_COLS = ["prev_day_mean", "mean7d_day", "load_day", "weekend", "is_holiday"]


def prepare(X: pd.DataFrame, y: pd.Series):
    loc = X.index.tz_convert(TZ)
    date = pd.Series(loc.date, index=X.index)
    X = X.copy()
    for c in DEV_SRC:   # deviatia fata de media zilei tinta (doar informatii cunoscute dimineata)
        X[c + "_dev"] = X[c] - X[c].groupby(date).transform("mean")
    X["mean7d_day"] = X["price_rolling_mean_7d"].groupby(date).transform("mean")
    X["load_day"] = X["load_forecast_mw"].groupby(date).transform("mean")
    X["weekend"] = (loc.dayofweek >= 5).astype(int)
    X["hour_local"] = loc.hour
    ydm = y.groupby(date).transform("mean")
    return X, date, ydm


def fit_all(X, y, ydm, date, train_days, trees):
    tr = date.isin(train_days).to_numpy()
    models = {"V0": M.fit(X[tr], y[tr], n_estimators=trees), "blocks": {}}
    h_tr = X.loc[tr, "hour_local"].to_numpy()
    for name, (a, b) in BLOCKS.items():   # V2: modele pe zone orare
        m = (h_tr >= a) & (h_tr < b)
        if m.sum() > 50:
            models["blocks"][name] = M.fit(X[tr][m], y[tr][m], n_estimators=trees)
    models["shape"] = M.fit(X[tr], (y - ydm)[tr], cols=SHAPE_COLS, n_estimators=trees)   # V3/V4: forma zilei
    daily = X[tr].groupby(date[tr]).first()[LEVEL_COLS].join(y[tr].groupby(date[tr]).mean().rename("y")).dropna()
    models["level"] = Ridge(alpha=10.0).fit(daily[LEVEL_COLS], daily["y"])
    return models


def predict_all(models, X, idx):
    Xt = X.loc[idx]
    out = {"V0": models["V0"].predict(Xt[F.PRODUCTION_FEATURES])}
    p2 = np.full(len(idx), np.nan)
    h = Xt["hour_local"].to_numpy()
    for name, (a, b) in BLOCKS.items():
        m = (h >= a) & (h < b)
        if name in models["blocks"] and m.any():
            p2[m] = models["blocks"][name].predict(Xt[F.PRODUCTION_FEATURES][m])
    out["V2"] = p2
    sh = models["shape"].predict(Xt[SHAPE_COLS])
    out["V3"] = models["level"].predict(Xt[LEVEL_COLS].iloc[[0]])[0] + sh
    out["V4"] = Xt["mean7d_day"].to_numpy() + sh
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--window", type=int, default=60)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng))
    X, date, ydm = prepare(X, y)
    days = sorted(date.unique())
    rows, models = [], None
    for k, day in enumerate(days[a.window:]):
        te = (date == day).to_numpy()
        if te.sum() < 92:
            continue
        if models is None or k % 7 == 0:   # reantrenare la 7 zile, pe ultimele `window` zile dinaintea zilei curente
            train_days = [d for d in days if d < day][-a.window:]
            models = fit_all(X, y, ydm, date, train_days, a.trees)
            print("reantrenat pentru", day, flush=True)
        idx = X.index[te]
        preds = predict_all(models, X, idx)
        part = pd.DataFrame({"y": y[te], "date": day, "hour": X.loc[te, "hour_local"],
                             "ieri": X.loc[te, "price_lag_24h"], "medie7": X.loc[te, "price_rolling_mean_7d"]}, index=idx)
        for name, p in preds.items():
            part[name] = p
        rows.append(part)
    bt = pd.concat(rows).dropna(subset=["y", "V0", "ieri", "medie7"])
    # V1: corectie cu reziduurile ultimelor 7 zile la aceeasi ora (doar zile anterioare)
    bt["slot"] = [f"{t.hour:02d}:{t.minute:02d}" for t in bt.index.tz_convert(TZ)]
    res = (bt["y"] - bt["V0"]).groupby([bt["date"], bt["slot"]]).mean().unstack("slot")
    adj = res.shift(1).rolling(7, min_periods=3).mean()
    bt["V1"] = bt["V0"] + adj.stack().reindex(pd.MultiIndex.from_arrays([bt["date"], bt["slot"]])).to_numpy()
    bt["V1"] = bt["V1"].fillna(bt["V0"])
    for v in ("V0", "V1", "V2", "V3", "V4"):
        bt[v + "+medie7"] = 0.5 * bt[v] + 0.5 * bt["medie7"]
    bt = bt.dropna(subset=["V2", "V3", "V4"])
    bt["month"] = [d.strftime("%Y-%m") for d in pd.to_datetime(bt["date"])]
    bt.to_csv("reports/improvements_predictions.csv")
    cols = ["ieri", "medie7", "V0", "V1", "V2", "V3", "V4", "V0+medie7", "V3+medie7", "V4+medie7"]
    daily = V.simulate_battery(bt.rename(columns={"y": "y"}), {**{c: c for c in cols}, "perfect": "y"}, V.Battery())
    daily.index = pd.to_datetime(daily.index)
    def pct(sel):
        t = daily[sel].sum()
        return {c: 100 * t[c] / t["perfect"] for c in cols}
    winter = daily.index.month.isin([12, 1, 2]); summer = daily.index.month.isin([6, 7, 8])
    table = {}
    for c in cols:
        e = (bt["y"] - bt[c]).abs()
        table[c] = {"MAE": e.mean(), "MAE_pranz": e[(bt.hour >= 10) & (bt.hour < 16)].mean(),
                    "MAE_seara": e[(bt.hour >= 16) & (bt.hour < 22)].mean(),
                    "baterie_%_tot": pct(slice(None))[c], "baterie_%_iarna": pct(winter)[c], "baterie_%_vara": pct(summer)[c]}
    df = pd.DataFrame(table).T
    print(f"\n{bt['date'].nunique()} zile, {bt.index.min().date()} .. {bt.index.max().date()}")
    print(df.round(1).to_string())
    Path("reports/improvements.json").write_text(json.dumps(df.round(2).to_dict("index"), indent=2))


if __name__ == "__main__":
    main()
