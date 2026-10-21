"""Cat conteaza frecventa reantrenarii? Zilnic vs saptamanal vs lunar, pe ultimele N zile (model de productie).

  python scripts/refit_frequency.py [--days 200] [--trees 400]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pzu import backtest, db, features as F, metrics, value as V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--trees", type=int, default=400)
    a = ap.parse_args()
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng))
    res = {}
    for name, every in (("zilnic", 1), ("saptamanal", 7), ("lunar", 28)):
        bt = backtest.walk_forward(X, y, 60, a.days, refit_every=every, n_estimators=a.trees)
        bt = bt.dropna(subset=["y", "pred", "naive_mean7d"])
        full = bt.groupby("date").size()
        bt = bt[bt["date"].isin(full[full >= 92].index)]
        bt["blend"] = 0.5 * bt["pred"] + 0.5 * bt["naive_mean7d"]
        daily = V.simulate_battery(bt, {"model": "pred", "mix": "blend", "perfect": "y"}, V.Battery())
        tot = daily.sum()
        res[name] = {"MAE": metrics.mae(bt["y"], bt["pred"]), "MAE_mix": metrics.mae(bt["y"], bt["blend"]),
                     "bias": float((bt["pred"] - bt["y"]).mean()),
                     "baterie_model_%": 100 * tot["model"] / tot["perfect"], "baterie_mix_%": 100 * tot["mix"] / tot["perfect"]}
        print(name, {k: round(v, 2) for k, v in res[name].items()}, f"({bt['date'].nunique()} zile)", flush=True)
    Path("reports/refit_frequency.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
