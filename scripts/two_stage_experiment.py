"""Test in doi pasi: (1) vreme -> productie eoliana/solara pe zone, (2) productie + consum -> pret.
Pasul 1 e out-of-sample in timp (reantrenat la 14 zile doar pe trecut), deci la fiecare zi de test
modelul de pret vede doar estimari facute cu ce se stia atunci.
  python scripts/two_stage_experiment.py <dir_cu_pzu.db_si_prod_csv> [zile_test] [variante...]
Fisierele prod_<zona>.csv vin din workflow-ul "Fetch region production" (scripts/fetch_region_production.py).
"""
import os, sys, time, json, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from pzu import backtest, db, features as F, pipeline, model as M, value as V
from pzu.config import TZ

D = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ONLY = sys.argv[3:]
eng = db.get_engine(f"sqlite:///{D}/pzu.db")

# ---------- vremea pe puncte (prognoza arhivata, orar UTC) ----------
w = pd.read_sql("select * from weather_fc", eng, parse_dates=["timestamp"])
w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
WP = {v: w.pivot(index="timestamp", columns="point", values=v).sort_index() for v in ("wind_ms", "rad_wm2", "cloud_pct", "temp_c")}

POINTS = {  # zona: (puncte vant, puncte soare)
    "RO": (["RO_dobrogea", "RO_buc"], ["RO_buc", "RO_craiova", "RO_cluj", "RO_dobrogea"]),
    "DE": (["DE_nord", "DE_centru", "DK", "NL"], ["DE_sud", "DE_centru", "DE_nord"]),
    "PL": (["PL", "DE_nord", "DK"], ["PL", "DE_centru"]),
    "HU": (["HU", "CZ_AT"], ["HU"]),
    "AT": (["CZ_AT", "HU"], ["CZ_AT", "DE_sud"]),
    "CZ": (["CZ_AT", "DE_centru"], ["CZ_AT", "DE_centru"]),
    "BG": (["BG", "RO_dobrogea"], ["BG", "GR"]),
}

def load_prod(z):
    p = f"{D}/prod_{z}.csv"
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["ts_utc"]).set_index("ts_utc")
        df.index = pd.to_datetime(df.index, utc=True)
        if "wind_on" in df:   # vant = uscat + larg; lipsa la uscat = lipsa (nu doar offshore)
            df["wind"] = df["wind_on"] + (df["wind_off"].fillna(0) if "wind_off" in df else 0)
        return df
    if z == "RO":   # rezerva: productia RO din baza de date
        g = pd.read_sql("select timestamp, gen_solar_mw solar, gen_wind_mw wind from generation_actual", eng, parse_dates=["timestamp"])
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
        return g.set_index("timestamp").resample("1h").mean()
    return None

def stage1_features(idx, pts_w, pts_s):
    loc = idx.tz_convert(TZ)
    f = pd.DataFrame(index=idx)
    f["h"] = loc.hour; f["doy_s"] = np.sin(2 * np.pi * loc.dayofyear / 365.25); f["doy_c"] = np.cos(2 * np.pi * loc.dayofyear / 365.25)
    f["t"] = (idx - pd.Timestamp("2024-01-01", tz="UTC")).days   # crestere de capacitate in timp
    for p in pts_w:
        v = WP["wind_ms"][p].reindex(idx); f[f"w_{p}"] = v; f[f"w3_{p}"] = np.clip(v, 0, 13) ** 3
    for p in pts_s:
        f[f"r_{p}"] = WP["rad_wm2"][p].reindex(idx); f[f"c_{p}"] = WP["cloud_pct"][p].reindex(idx)
    f["temp"] = WP["temp_c"][pts_s].reindex(idx).mean(axis=1)
    return f

def stage1(z):
    """Estimari out-of-sample (orar) pentru vant si soare; reantrenare la 14 zile pe ultimele 365 zile."""
    prod = load_prod(z)
    if prod is None:
        return None
    pts_w, pts_s = POINTS[z]
    idx = WP["wind_ms"].index
    Xs = stage1_features(idx, pts_w, pts_s)
    out = pd.DataFrame(index=idx)
    for tgt in ("wind", "solar"):
        if tgt not in prod or prod[tgt].notna().sum() < 24 * 120:
            continue
        y = prod[tgt].reindex(idx)
        pred = pd.Series(np.nan, index=idx)
        days = pd.date_range(idx.min().normalize() + pd.Timedelta(days=90), idx.max(), freq="14D")
        for d0 in days:
            tr = (idx < d0) & (idx >= d0 - pd.Timedelta(days=365)) & y.notna().to_numpy()
            te = (idx >= d0) & (idx < d0 + pd.Timedelta(days=14))
            if tr.sum() < 24 * 60 or te.sum() == 0:
                continue
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbosity=-1)
            m.fit(Xs[tr], y[tr])
            pred[te] = np.clip(m.predict(Xs[te]), 0, None)
        out[f"{tgt}_hat"] = pred
        ok = y.notna() & pred.notna()
        r = np.corrcoef(y[ok], pred[ok])[0, 1] if ok.sum() > 100 else np.nan
        mae = (y[ok] - pred[ok]).abs().mean(); print(f"  pas 1 {z} {tgt}: corelatie estimare/real {r:.2f}, eroare medie {mae:.0f} MW (medie real {y[ok].mean():.0f} MW)", flush=True)
        out[f"{tgt}_act"] = y
    if "load_fc" in prod:
        out["load"] = prod["load_fc"].reindex(idx)
    return out

t0 = time.time()
S1 = {}
for z in POINTS:
    r = stage1(z)
    if r is not None:
        S1[z] = r
print("zone cu productie:", list(S1), f"({time.time()-t0:.0f}s)", flush=True)

# ---------- pasul 2: modelul de pret ----------
X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng),
                            nload=db.read_neighbor_load(eng), weather=db.read_weather(eng))
base = pipeline._main_cols(X, slice(None))
day = pd.Series(X.index.tz_convert(TZ).date, index=X.index)

def to15(s):
    return s.resample("15min").interpolate("time", limit=8).reindex(X.index)

def add_feats(kind):
    """kind = 'hat' (estimari) sau 'act' (productie reala = limita superioara)."""
    cols = []
    tot_ren, tot_load = 0, 0
    for z, d in S1.items():
        w_, s_ = d.get(f"wind_{kind}"), d.get(f"solar_{kind}")
        ren = (to15(w_) if w_ is not None else 0) + (to15(s_) if s_ is not None else 0)
        if z == "RO":
            load = X["load_forecast_mw"]
        elif "load" in d:
            load = to15(d["load"])
        else:
            load = None
        X[f"p2_{z}_ren_{kind}"] = ren; cols.append(f"p2_{z}_ren_{kind}")
        if load is not None:
            X[f"p2_{z}_res_{kind}"] = load - ren; cols.append(f"p2_{z}_res_{kind}")
            X[f"p2_{z}_share_{kind}"] = ren / load.replace(0, np.nan); cols.append(f"p2_{z}_share_{kind}")
            if z != "RO":
                tot_ren = tot_ren + ren; tot_load = tot_load + load
    if not isinstance(tot_load, int):
        X[f"p2_REG_res_{kind}"] = tot_load - tot_ren; X[f"p2_REG_share_{kind}"] = tot_ren / tot_load
        X[f"p2_REG_res_zi_{kind}"] = X[f"p2_REG_res_{kind}"].groupby(day).transform("mean")
        X[f"p2_REG_res_min_{kind}"] = X[f"p2_REG_res_{kind}"].groupby(day).transform("min")
        cols += [f"p2_REG_res_{kind}", f"p2_REG_share_{kind}", f"p2_REG_res_zi_{kind}", f"p2_REG_res_min_{kind}"]
    return cols

hat = add_feats("hat"); act = add_feats("act")
ro_hat = [c for c in hat if c.startswith("p2_RO_")]
print("variabile noi:", len(hat), "| acoperire estimari in ultimele 400 zile:",
      round(float(X.loc[day >= sorted(day.unique())[-N], hat].notna().mean().mean()), 2), flush=True)

days = {}
def run(name, cols):
    t = time.time()
    bt = backtest.walk_forward(X, y, M.WINDOW_DAYS, N, cols=cols, refit_every=7, pair=True).dropna(subset=["y", "pred"])
    full = bt.groupby("date").size(); bt = bt[bt.date.isin(full[full >= 92].index)]
    d = (bt.y - bt.pred).abs().groupby(bt.date).mean(); days[name] = d
    bat = V.simulate_battery(bt, {"m": "pred", "perfect": "y"}, V.Battery()).sum()
    ok20 = ((bt.pred - bt.y).abs() <= np.maximum(0.2 * bt.y.abs(), 20)).mean()
    print(f"{name:42s} MAE {d.mean():6.2f} | ult30 {d.tail(30).mean():6.2f} | sloturi corecte (±20%/±20€) {100*ok20:5.1f}% | baterie {100*bat.m/bat.perfect:5.1f}% | {time.time()-t:.0f}s", flush=True)
    pickle.dump(days, open(f"{D}/ts_days.pkl", "wb"))

VAR = {"V0 model actual (v3)": base,
       "V1 + productie estimata Romania": base + ro_hat,
       "V2 + productie estimata regiune": base + hat,
       "V3 limita: productie REALA regiune": base + act}
for k, v in VAR.items():
    if not ONLY or any(o in k for o in ONLY):
        run(k, v)
Dd = pd.DataFrame(days).dropna(); ref = "V0 model actual (v3)"
if ref in Dd:
    for k in Dd.columns:
        if k == ref: continue
        diff = Dd[ref] - Dd[k]; se = diff.std() / np.sqrt(len(diff)); h = len(diff) // 2
        print(f"{k:42s} castig {diff.mean():+5.2f} ± {1.96*se:.2f} €/MWh | zile mai bune {int((diff>0).sum())}/{len(diff)} | jumatati {diff.iloc[:h].mean():+.2f}/{diff.iloc[h:].mean():+.2f}")
print("DONE")
