"""Intervale de incredere prin conformal prediction (split conformal, simetric).

NOTA fata de spec: pentru un interval de 90% se foloseste cuantila 90% a |eroare| (nu 95%;
cuantila 95% ar da ~95% acoperire). Se aplica corectia de esantion finit ceil((n+1)(1-a))/n.
Optional, cuantile separate pe ora a zilei (erorile de seara sunt mai mari decat noaptea).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pzu.timegrid import to_local


def _quantile(abs_err: np.ndarray, alpha: float) -> float:
    n = len(abs_err)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_err, level, method="higher"))


@dataclass
class ConformalInterval:
    alpha: float = 0.10
    by_hour: bool = True
    min_per_hour: int = 100
    q_global: float = float("nan")
    q_hour: dict[int, float] = field(default_factory=dict)

    def fit(self, errors: pd.Series) -> "ConformalInterval":
        """errors = real - prognoza, index = timestamp UTC."""
        e = errors.dropna().abs()
        self.q_global = _quantile(e.to_numpy(), self.alpha)
        self.q_hour = {}
        if self.by_hour:
            hours = to_local(e.index).hour
            for h in range(24):
                vals = e[hours == h].to_numpy()
                if len(vals) >= self.min_per_hour:
                    self.q_hour[h] = _quantile(vals, self.alpha)
        return self

    def half_width(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = to_local(index).hour
        return np.array([self.q_hour.get(h, self.q_global) for h in hours])

    def interval(self, pred: pd.Series) -> tuple[pd.Series, pd.Series]:
        q = self.half_width(pred.index)
        return pred - q, pred + q


def rolling_intervals(bt: pd.DataFrame, alpha: float = 0.10, min_days: int = 14,
                      lookback_days: int = 60, by_hour: bool = False) -> pd.DataFrame:
    """Intervale 'oneste' pe backtest: pentru fiecare zi, calibrare doar pe erorile zilelor anterioare.
    bt: coloane date, y, pred. Zilele fara suficient istoric de erori sunt omise."""
    days = sorted(bt["date"].unique())
    parts = []
    for k, day in enumerate(days):
        if k < min_days:
            continue
        past = bt[bt["date"].isin(days[max(0, k - lookback_days):k])]
        ci = ConformalInterval(alpha, by_hour=by_hour, min_per_hour=20).fit(past["y"] - past["pred"])
        cur = bt[bt["date"] == day].copy()
        cur["ci_lower"], cur["ci_upper"] = ci.interval(cur["pred"])
        parts.append(cur)
    return pd.concat(parts) if parts else bt.iloc[0:0].assign(ci_lower=np.nan, ci_upper=np.nan)
