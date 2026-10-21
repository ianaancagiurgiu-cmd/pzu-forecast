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
EXTRAS_SETS = {
    "A. referinta (setul de productie)": dict(cols=OLD),
    "F. + productie pe tipuri (lag 24h)": dict(cols=OLD + F.GEN_FEATURES),
    "G. + indisponibilitati RO/HU/BG": dict(cols=OLD + F.OUTAGE_FEATURES),
    "H. + productie + indisponibilitati": dict(cols=OLD + F.GEN_FEATURES + F.OUTAGE_FEATURES),
    "I. H, corectie la media 7z": dict(cols=OLD + F.GEN_FEATURES + F.OUTAGE_FEATURES, target="residual"),
}
# Model "de dimineata": prognozele ENTSO-E solar/eolian pentru D+1 apar dupa inchiderea pietei, deci nu pot fi folosite.
NO_RES = [f for f in OLD if f not in ("solar_forecast_mw", "wind_forecast_mw", "residual_load")]
WEATHER = ["temp_c", "wind_speed_ms", "radiation_wm2"]
NO_WEATHER = [f for f in NO_RES if f not in WEATHER]
CAPACITY = ["capacity_bg_ro_mw", "capacity_ro_bg_mw"]
NO_CAP = [f for f in NO_WEATHER if f not in CAPACITY]
MORNING_SETS = {
    "A. model actual (foloseste solar/eolian D+1, INDISPONIBIL dimineata)": dict(cols=OLD),
    "M0. dimineata: fara solar/eolian ENTSO-E, cu vreme (vreme REALA din arhiva = optimist)": dict(cols=NO_RES),
    "M1. dimineata: fara solar/eolian si fara vreme (complet onest)": dict(cols=NO_WEATHER),
    "M2. ca M1, si fara capacitati de interconexiune": dict(cols=NO_CAP),
}
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
    ap.add_argument("--set", choices=["base", "extras", "morning"], default="base")
    a = ap.parse_args()
    variants = {"base": VARIANTS, "extras": EXTRAS_SETS, "morning": MORNING_SETS}[a.set]
    eng = db.get_engine()
    db.ensure_real_data(eng)
    kw = {}
    if a.set == "extras":
        kw["gen"] = db.read_df(eng, db.generation_actual)
        o = db.read_df(eng, db.outages, index=None)
        for c in ("start", "end", "created"):
            o[c] = pd.to_datetime(o[c], utc=True)
        kw["outages"] = o
        print(f"productie: {len(kw['gen'])} intervale; indisponibilitati: {len(o)} documente")
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng), **kw)
    out = {}
    for period, offset in (("dezvoltare", a.days), ("holdout", 0)):
        bt = None
        cols = []
        for name, vkw in variants.items():
            part = backtest.walk_forward(X, y, 60, a.days, refit_every=7, end_offset_days=offset,
                                         n_estimators=a.trees, **vkw)[["y", "pred"]].rename(columns={"pred": name})
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
    Path(f"reports/experiments_{a.set}.json").write_text(json.dumps({k: v.round(2).to_dict("index") for k, v in out.items()},
                                                          indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
