"""Test: capacitatea indisponibila (centrale oprite) ca variabila a modelului de pret v3.
Pentru fiecare zi prognozata se folosesc DOAR documentele create pana la ora prognozei (06:00 sau 11:00 in ziua dinainte),
ultima revizie cunoscuta atunci, fara cele anulate. Varianta 'oracol' foloseste tot ce s-a aflat (limita superioara).
  python scripts/outage_experiment.py <dir cu pzu.db si outages_*.csv> [zile]
Fisierele outages_<zona>.csv vin din workflow-ul "Fetch outages" (scripts/fetch_outages_csv.py).
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
from pzu import backtest, db, features as F, pipeline, model as M, value as V
from pzu.config import TZ

D = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
eng = db.get_engine(f"sqlite:///{D}/pzu.db")

def cat(pt):
    pt = str(pt).lower()
    if "nuclear" in pt: return "nuclear"
    if "hydro" in pt: return "hydro"
    if any(k in pt for k in ("coal", "lignite", "gas", "oil", "fossil")): return "thermal"
    return "other"

docs = []
for z in ("RO", "HU", "BG", "RS"):
    p = f"{D}/outages_{z}.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 10: continue
    d = pd.read_csv(p)
    if d.empty: continue
    if (d["kind"] == "A80").any():   # A80 (unitati) si A77 (centrale) descriu adesea aceeasi oprire: folosim una
        d = d[d["kind"] == "A80"]
    for c in ("start", "end", "created"):
        d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    d["unavail"] = (pd.to_numeric(d["nominal_mw"], errors="coerce") - pd.to_numeric(d["available_mw"], errors="coerce")).clip(lower=0).fillna(0)
    d["rev"] = pd.to_numeric(d["revision"], errors="coerce").fillna(0)
    d["cancel"] = d["doc_status"].fillna("").astype(str).str.lower().str.contains("cancel|withdraw|A09|A13")
    d["cat"] = d["plant_type"].map(cat)
    d["unpl"] = d["business_type"].astype(str).str.contains("A54|nplanned|Unplanned", case=False)
    docs.append(d)
    print(z, "documente", len(d), "| creat", d.created.min().date(), "->", d.created.max().date(), "| pe categorii (MW medii):",
          d.groupby("cat").unavail.mean().round(0).to_dict(), flush=True)
docs = pd.concat(docs, ignore_index=True)

X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng),
                            nload=db.read_neighbor_load(eng), weather=db.read_weather(eng))
base = pipeline._main_cols(X, slice(None))
loc = X.index.tz_convert(TZ)
dates = pd.Series(loc.date, index=X.index)
FEATS = [f"out_{z}_{c}" for z in ("RO", "HU", "BG", "RS") for c in ("total", "nuclear", "hydro", "thermal", "unpl")]

def features(cutoff_hour):
    """cutoff_hour: ora locala din ziua dinainte (6, 11); None = oracol (tot ce s-a aflat vreodata)."""
    out = {c: np.zeros(len(X)) for c in FEATS}
    starts, ends = docs.start.values, docs.end.values
    for day, ix in dates.groupby(dates).groups.items():
        t = X.index[dates.index.get_indexer(ix)] if False else pd.DatetimeIndex(ix)
        d0, d1 = t.min(), t.max()
        k = docs[(docs.end > d0) & (docs.start <= d1)]
        if cutoff_hour is not None:
            cut = (pd.Timestamp(day, tz=TZ) - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)).tz_convert("UTC")
            k = k[k.created <= cut]
        if k.empty: continue
        k = k[k.rev == k.groupby("mrid").rev.transform("max")]
        k = k[~k.cancel]
        pos = X.index.get_indexer(t)
        tv = t.values
        for r in k.itertuples():
            m = (tv >= r.start.to_datetime64()) & (tv < r.end.to_datetime64())
            if not m.any(): continue
            z = r.zone
            out[f"out_{z}_total"][pos[m]] += r.unavail
            if r.cat in ("nuclear", "hydro", "thermal"): out[f"out_{z}_{r.cat}"][pos[m]] += r.unavail
            if r.unpl: out[f"out_{z}_unpl"][pos[m]] += r.unavail
    return pd.DataFrame(out, index=X.index)

t = time.time()
Fa = features(6); Fb = features(None)
print(f"variabile construite ({time.time()-t:.0f}s); medii MW (la 06:00):", Fa.mean().round(0)[Fa.mean() > 0].to_dict(), flush=True)
ro = [c for c in FEATS if c.startswith("out_RO")]
X1 = X.join(Fa); X2 = X.join(Fb.add_suffix("_or"))
days = {}
def run(name, Xv, cols):
    t = time.time()
    bt = backtest.walk_forward(Xv, y, M.WINDOW_DAYS, N, cols=cols, refit_every=7, pair=True).dropna(subset=["y", "pred"])
    full = bt.groupby("date").size(); bt = bt[bt.date.isin(full[full >= 92].index)]
    d = (bt.y - bt.pred).abs().groupby(bt.date).mean(); days[name] = d
    ok20 = ((bt.pred - bt.y).abs() <= np.maximum(0.2 * bt.y.abs(), 20)).mean()
    print(f"{name:44s} MAE {d.mean():6.2f} | ult30 {d.tail(30).mean():6.2f} | sloturi corecte {100*ok20:5.1f}% | {time.time()-t:.0f}s", flush=True)
run("V0 model actual (v3)", X, base)
run("V1 + indisponibilitati Romania", X1, base + ro)
run("V2 + indisponibilitati RO, HU, BG, RS", X1, base + FEATS)
run("V3 limita: tot ce s-a aflat ulterior", X2, base + [c + "_or" for c in FEATS])
Dd = pd.DataFrame(days).dropna(); ref = "V0 model actual (v3)"
for k in Dd.columns:
    if k == ref: continue
    diff = Dd[ref] - Dd[k]; se = diff.std() / np.sqrt(len(diff)); h = len(diff) // 2
    print(f"{k:44s} castig {diff.mean():+5.2f} ± {1.96*se:.2f} €/MWh | zile mai bune {int((diff>0).sum())}/{len(diff)} | jumatati {diff.iloc[:h].mean():+.2f}/{diff.iloc[h:].mean():+.2f}")
pickle.dump(days, open(f"{D}/outage_days.pkl", "wb"))
print("DONE")
