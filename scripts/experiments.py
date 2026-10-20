"""Compara variante de model pe doua perioade: dezvoltare (mai veche) si holdout (ultimele zile).
Alegem varianta pe dezvoltare si o confirmam O SINGURA DATA pe holdout (fara ajustari dupa holdout).

  python scripts/experiments.py [--days 90] [--trees 500]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pzu import backtest, db, features as F, metrics, value as V

OLD = list(F.FEATURES)
NEW = OLD + F.EXPERIMENTAL
VARIANTS = {
    "A. nivel, variabile vechi": dict(cols=OLD),
    "B. nivel, variabile noi": dict(cols=NEW),
    "C. corectie la media 7z, variabile noi": dict(target="residual", cols=NEW),
    "D. corectie, obiectiv L1 (robust la varfuri)": dict(target="residual", cols=NEW, objective="l1"),
    "E. corectie, obiectiv Huber": dict(target="residual", cols=NEW, objective="huber", alpha=25.0),
}


def evaluate(bt: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)].dropna(subset=cols + ["y", "naive_mean7d"])
    perfect = V.simulate_battery(bt, {"p": "y"}, V.Battery())["p"].sum()
    rows = {}
    for c in cols:
        rev = V.simulate_battery(bt, {"r": c}, V.Battery())["r"].sum()
        m = metrics.summarize(bt["y"], bt[c])
        rows[c] = {"MAE": m["mae"], "RMSE": m["rmse"], "bateria_EUR": rev, "pct_plafon": 100 * rev / perfect}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--trees", type=int, default=500)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng))
    out = {}
    for period, offset in (("dezvoltare", a.days), ("holdout", 0)):
        bt = None
        cols = []
        for name, kw in VARIANTS.items():
            part = backtest.walk_forward(X, y, 60, a.days, refit_every=7, end_offset_days=offset,
                                         n_estimators=a.trees, **kw)[["y", "pred"]].rename(columns={"pred": name})
            bt = part if bt is None else bt.join(part[name])
            cols.append(name)
        base = backtest.walk_forward(X, y, 60, a.days, refit_every=10**6, end_offset_days=offset, n_estimators=5)
        bt = bt.join(base[["naive_lag24h", "naive_mean7d", "date"]])
        for name in cols:   # mix cu media pe 7 zile (ponderi fixe 50/50)
            bt[name + " + media7z 50/50"] = 0.5 * bt[name] + 0.5 * bt["naive_mean7d"]
        allc = ["naive_lag24h", "naive_mean7d", *cols, *[c + " + media7z 50/50" for c in cols]]
        res = evaluate(bt, allc)
        out[period] = res
        print(f"\n=== {period}: {bt.index.min().date()} .. {bt.index.max().date()} ===")
        print(res.round(1).to_string())
    Path("reports/experiments.json").write_text(json.dumps({k: v.round(2).to_dict("index") for k, v in out.items()},
                                                          indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
