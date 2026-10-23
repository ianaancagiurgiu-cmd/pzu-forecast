"""Zilele recente conteaza mai mult la antrenare? Testat pe TOT istoricul (668 zile, toate anotimpurile).

Pondere = 0.5 ** (varsta_in_zile / half_life). Varianta (60, None) = productia actuala (toate zilele la fel).

  python scripts/recency_weights.py [--trees 400] [--variants 60:none,60:20,...]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pzu import backtest, db, features as F, value as V

DEFAULT = "60:none,60:20,60:30,60:45,120:30,120:60,180:30,180:60"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--variants", default=DEFAULT)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng), nload=db.read_neighbor_load(eng))
    cols = [c for c in F.PRODUCTION_FEATURES if X[c].notna().mean() > 0.5]
    n_days = pd.Series(X.index.tz_convert("Europe/Bucharest").date).nunique() - 62   # aceleasi zile de test pentru toate
    bt, names = None, []
    for v in a.variants.split(","):
        w, hl = v.split(":")
        name = f"fereastra {w}z, half-life {hl}" if hl != "none" else f"fereastra {w}z, uniform"
        part = backtest.walk_forward(X, y, int(w), n_days, refit_every=7, cols=cols,
                                     half_life_days=None if hl == "none" else float(hl),
                                     n_estimators=a.trees, n_jobs=a.threads)
        part = part[["y", "pred", "date", "naive_lag24h", "naive_mean7d"]].rename(columns={"pred": name})
        bt = part if bt is None else bt.join(part[name], how="inner")
        names.append(name)
        print("gata", name, flush=True)
    bt = bt.dropna()
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)]
    for n in names:
        bt[n + " + mix"] = 0.5 * bt[n] + 0.5 * bt["naive_mean7d"]
    allc = ["naive_mean7d", *names, *[n + " + mix" for n in names]]
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
    Path("reports/recency_weights.json").write_text(json.dumps(df.round(2).to_dict("index"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
