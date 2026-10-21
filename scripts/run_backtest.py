"""Backtest walk-forward pe date REALE, cu comparatie de ferestre de antrenare.

  python scripts/run_backtest.py --windows 60,180,365 --days 90
  python scripts/run_backtest.py --smoke-test     # doar verificare de cod, pe date sintetice
"""
import argparse
import json
from pathlib import Path

from pzu import backtest, db, features as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="60,180,365")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--smoke-test", action="store_true",
                    help="date sintetice generate in memorie; nu scrie nimic, rezultatele nu au valoare")
    ap.add_argument("--out", default="reports/backtest.json")
    a = ap.parse_args()

    if a.smoke_test:
        from pzu.synthetic import generate
        inputs, prices = generate("2026-01-01", 120)
        windows, n_days, overrides = [30], 20, {"n_estimators": 100}
        print("!!! SMOKE TEST pe date sintetice: rezultatele NU reflecta piata reala !!!")
    else:
        eng = db.get_engine()
        db.ensure_real_data(eng)
        inputs, prices = db.read_df(eng, db.forecasts_input), db.read_prices(eng)
        windows, n_days, overrides = [int(w) for w in a.windows.split(",")], a.days, {}

    X, y = F.build_training_set(inputs, prices, nload=None if a.smoke_test else db.read_neighbor_load(eng))
    results = {}
    for w in windows:
        bt = backtest.walk_forward(X, y, w, n_days, **overrides)
        res = backtest.evaluate(bt)
        res["targets_met"] = backtest.check_targets(res)
        results[f"window_{w}d"] = res
        m = res["lightgbm"]
        print(f"fereastra {w:>3}d | MAE {m['mae']:.2f} RMSE {m['rmse']:.2f} sMAPE {m['smape']:.1f}% "
              f"hit {m['directional_hit_rate']:.2f} cov90 {m.get('coverage_90', float('nan')):.2f} "
              f"| baseline lag24h MAE {res['naive_lag24h']['mae']:.2f}")
    if not a.smoke_test:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
