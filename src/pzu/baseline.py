"""Baseline-uri naive: tinta pe care LightGBM trebuie s-o bata."""
import pandas as pd

from pzu.metrics import summarize

BASELINES = {
    "naive_lag24h": "price_lag_24h",
    "naive_lag7d": "price_lag_7d",
    "naive_mean7d": "price_rolling_mean_7d",
}


def evaluate_baselines(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = {}
    for name, col in BASELINES.items():
        ok = X[col].notna() & y.notna()
        rows[name] = summarize(y[ok], X.loc[ok, col])
    return pd.DataFrame(rows).T
