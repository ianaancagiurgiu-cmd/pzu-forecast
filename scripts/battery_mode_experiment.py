"""Mod baterie: ce semnal de pret alege cel mai bine orele de incarcare/descarcare? (walk-forward, fara date din viitor)

Candidati: modelul v3, media pe 7 zile, combinatii, "nivel + forma" (nivelul zilei din model, forma zilei din ultimele
7 zile sau dintr-un model antrenat special pe forma) si o regula "nu face nimic" cand castigul prognozat e mic.
Scorul: castigul simulat al bateriei la preturile REALE (% din optim si EUR/zi), nu eroarea medie.

  python scripts/battery_mode_experiment.py <cale pzu.db> [zile]
"""
import sys
import numpy as np
import pandas as pd

from pzu import backtest, db, features as F, model as M, pipeline, value as V
from pzu.config import TZ

path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
eng = db.get_engine(f"sqlite:///{path}")
X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng),
                            nload=db.read_neighbor_load(eng), weather=db.read_weather(eng))
cols = pipeline._main_cols(X, slice(None))
B = V.Battery()   # 10 MW / 20 MWh, randament 85%, 1 ciclu/zi, 4 EUR/MWh uzura

bt = backtest.walk_forward(X, y, M.WINDOW_DAYS, N, cols=cols, refit_every=7, pair=True).dropna(subset=["y", "pred"])
day = pd.Series(X.index.tz_convert(TZ).date, index=X.index)
ys = y - y.groupby(day).transform("mean")                       # forma zilei: abaterea fata de media zilei
sh = backtest.walk_forward(X, ys, M.WINDOW_DAYS, N, cols=cols, refit_every=7).dropna(subset=["pred"])
bt["shape_model"] = sh["pred"].reindex(bt.index)
full = bt.groupby("date").size(); bt = bt[bt.date.isin(full[full >= 92].index)].dropna(subset=["naive_mean7d", "shape_model"])

g = bt.groupby("date")
lvl = g["pred"].transform("mean")
m7 = bt["naive_mean7d"]; sh7 = m7 - g["naive_mean7d"].transform("mean")
shm = bt["shape_model"] - g["shape_model"].transform("mean")
shp = bt["pred"] - lvl
C = {
    "model v3": bt["pred"],
    "media 7 zile": m7,
    "mix 50/50": 0.5 * bt["pred"] + 0.5 * m7,
    "mix 25/75 (mai mult media)": 0.25 * bt["pred"] + 0.75 * m7,
    "nivel model + forma 7 zile": lvl + sh7,
    "nivel model + forma model-forma": lvl + shm,
    "nivel + forma (model-forma + 7 zile)/2": lvl + 0.5 * (shm + sh7),
    "nivel + forma (v3 + model-forma + 7 zile)/3": lvl + (shp + shm + sh7) / 3,
}
for k, v in C.items():
    bt[k] = v

def score(df, strategies, label=""):
    r = V.simulate_battery(df, {**strategies, "perfect": "y"}, B)
    return r

S = {k: k for k in C}
res = score(bt, S)
tot = res.sum(); nd = len(res); h = nd // 2
print(f"zile: {nd} ({bt.date.min()} - {bt.date.max()}) | castig maxim posibil (prognoza perfecta): {tot.perfect/nd:,.0f} EUR/zi")
print(f"{'semnal':45s} {'% din optim':>11s} {'EUR/zi':>8s} {'prima jum.':>10s} {'a doua':>8s} {'ult. 30z':>9s}")
for k in C:
    pct = 100 * tot[k] / tot.perfect
    p1 = 100 * res[k].iloc[:h].sum() / res.perfect.iloc[:h].sum(); p2 = 100 * res[k].iloc[h:].sum() / res.perfect.iloc[h:].sum()
    l30 = 100 * res[k].tail(30).sum() / res.perfect.tail(30).sum()
    print(f"{k:45s} {pct:10.1f}% {tot[k]/nd:8,.0f} {p1:9.1f}% {p2:7.1f}% {l30:8.1f}%")
# perechi: cat de sigur e castigul fata de media pe 7 zile si fata de v3
for ref in ("media 7 zile", "model v3"):
    for k in C:
        if k == ref: continue
        d = (res[k] - res[ref]); se = d.std() / np.sqrt(len(d))
        if abs(d.mean()) > 1.96 * se:
            print(f"  semnificativ: {k} fata de {ref}: {d.mean():+,.0f} EUR/zi ± {1.96*se:,.0f}")
# regula "nu face nimic": cicleaza doar daca castigul PROGNOZAT (pe semnalul ales) depaseste un prag
best = max(C, key=lambda k: tot[k])
print(f"\ncel mai bun semnal: {best}")
pred_rev = {}
for d_, gg in bt.groupby("date"):
    c, dch = V.dispatch(gg[best].to_numpy(), B)
    pred_rev[d_] = V.battery_revenue(c, dch, gg[best].to_numpy(), B)
pred_rev = pd.Series(pred_rev)
for thr in (0, 200, 500, 1000):
    take = pred_rev.reindex(res.index) > thr
    got = res[best].where(take, 0.0)
    print(f"  prag castig prognozat > {thr:5d} EUR: {100*got.sum()/tot.perfect:5.1f}% din optim | {got.sum()/nd:7,.0f} EUR/zi | zile active {int(take.sum())}/{nd} | zile cu pierdere evitate {int(((res[best]<0)&~take).sum())}")
