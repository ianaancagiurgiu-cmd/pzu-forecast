"""Walk-forward: antrenam pe ultimele `window_days` zile, prezicem ziua urmatoare, avansam o zi."""
from __future__ import annotations

import pandas as pd

from pzu import features as F
from pzu import model as M
from pzu.baseline import BASELINES
from pzu.conformal import rolling_intervals
from pzu.metrics import summarize
from pzu.timegrid import to_local


def _local_dates(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(to_local(idx).date, index=idx)


def walk_forward(X: pd.DataFrame, y: pd.Series, window_days: int = 60, n_days: int = 90,
                 **model_overrides) -> pd.DataFrame:
    """Rezultat: un rand pe interval (y, pred, date, + baseline-uri). Se testeaza ultimele
    `n_days` zile; fiecare zi e prezisa cu un model antrenat doar pe zilele anterioare."""
    dates = _local_dates(X.index)
    days = sorted(dates.unique())
    # sare peste ziua incompleta de la capat; ultimele n_days zile devin zile de test
    test_days = days[-n_days:] if n_days < len(days) else days[1:]
    rows = []
    for day in test_days:
        train_days = [d for d in days if d < day][-window_days:]
        if len(train_days) < min(window_days, 14):
            continue
        tr = dates.isin(train_days).to_numpy()
        te = (dates == day).to_numpy()
        if te.sum() == 0:
            continue
        m = M.fit(X[tr], y[tr], **model_overrides)
        part = pd.DataFrame({"y": y[te], "pred": m.predict(X.loc[te, F.FEATURES])}, index=X.index[te])
        part["date"] = day
        for name, col in BASELINES.items():
            part[name] = X.loc[te, col]
        rows.append(part)
    return pd.concat(rows) if rows else pd.DataFrame(columns=["y", "pred", "date", *BASELINES])


def evaluate(bt: pd.DataFrame, alpha: float = 0.10, by_hour: bool = False) -> dict:
    """Metrici LightGBM + baseline-uri (pe aceleasi intervale) + coverage cu calibrare 'onesta'."""
    out = {"n_days": int(bt["date"].nunique()), "n_intervals": int(len(bt))}
    ok = bt.dropna(subset=["y", "pred"] + list(BASELINES))
    out["lightgbm"] = summarize(ok["y"], ok["pred"])
    for name in BASELINES:
        out[name] = summarize(ok["y"], ok[name])
    ci = rolling_intervals(bt.dropna(subset=["y", "pred"]), alpha=alpha, by_hour=by_hour)
    if len(ci):
        out["lightgbm"] = summarize(ci["y"], ci["pred"], ci["ci_lower"], ci["ci_upper"])
        out["coverage_days"] = int(ci["date"].nunique())
    return out


TARGETS = {"mae": 20.0, "rmse": 30.0, "smape": 15.0}


def check_targets(metrics: dict) -> dict[str, bool]:
    m = metrics["lightgbm"]
    res = {k: m[k] < v for k, v in TARGETS.items()}
    res["directional_hit_rate"] = m["directional_hit_rate"] > 0.60
    if "coverage_90" in m:
        res["coverage_90"] = 0.85 <= m["coverage_90"] <= 0.95
    return res
