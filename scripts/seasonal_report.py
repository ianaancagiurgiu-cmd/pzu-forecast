"""Backtest pe TOT istoricul disponibil (toate anotimpurile), cu rezultate pe luni.

  python scripts/seasonal_report.py [--trees 500]

Antrenare pe ultimele 60 de zile, test pe ziua urmatoare, reantrenare la 7 zile. Perioada dinainte de oct. 2025 are
preturi orare (replicate pe 4 sferturi), deci forma intra-orara din acea perioada nu e reprezentativa.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pzu import backtest, db, features as F, metrics, value as V
from pzu.conformal import rolling_intervals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--window", type=int, default=60)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng),
                                nload=db.read_neighbor_load(eng))
    n_days = pd.Series(X.index.tz_convert("Europe/Bucharest").date).nunique() - a.window - 2
    bt = backtest.walk_forward(X, y, a.window, n_days, refit_every=7, n_estimators=a.trees)
    bt = bt.dropna(subset=["y", "pred", "naive_lag24h", "naive_mean7d"])
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)]
    bt["month"] = [d.strftime("%Y-%m") for d in pd.to_datetime(bt["date"])]
    bt["blend"] = 0.5 * bt["pred"] + 0.5 * bt["naive_mean7d"]
    ci = rolling_intervals(bt, 0.10)
    rows = []
    for m, g in bt.groupby("month"):
        daily = V.simulate_battery(g, {"model": "pred", "ieri": "naive_lag24h", "medie7": "naive_mean7d", "perfect": "y"}, V.Battery())
        tot = daily.sum()
        rows.append({"luna": m, "zile": g["date"].nunique(), "pret_mediu": g["y"].mean(),
                     "MAE_model": metrics.mae(g["y"], g["pred"]), "MAE_ieri": metrics.mae(g["y"], g["naive_lag24h"]),
                     "MAE_medie7": metrics.mae(g["y"], g["naive_mean7d"]),
                     "baterie_model_%": 100 * tot["model"] / tot["perfect"], "baterie_medie7_%": 100 * tot["medie7"] / tot["perfect"],
                     "baterie_ieri_%": 100 * tot["ieri"] / tot["perfect"]})
    df = pd.DataFrame(rows).set_index("luna")
    print(df.round(1).to_string())
    tot = {"zile": int(bt["date"].nunique()), "MAE_model": metrics.mae(bt["y"], bt["pred"]),
           "MAE_ieri": metrics.mae(bt["y"], bt["naive_lag24h"]), "MAE_medie7": metrics.mae(bt["y"], bt["naive_mean7d"]),
           "coverage_90": float(((ci["y"] >= ci["ci_lower"]) & (ci["y"] <= ci["ci_upper"])).mean())}
    err = (bt["y"] - bt["pred"]).abs()
    tot.update({"mediana_eroare": float(err.median()), "p90_eroare": float(err.quantile(0.9)),
                "sub_20": float((err < 20).mean()), "sub_40": float((err < 40).mean())})
    print({k: round(v, 3) for k, v in tot.items()})
    Path("reports").mkdir(exist_ok=True)
    Path("reports/seasonal_report.json").write_text(json.dumps({"total": tot, "luni": df.round(2).to_dict("index")}, indent=2))


if __name__ == "__main__":
    main()
