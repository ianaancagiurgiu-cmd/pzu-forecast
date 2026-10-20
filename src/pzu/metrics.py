import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y, p) -> float:
    """sMAPE in procente. Intervalele cu |y|+|p| = 0 sunt tratate ca eroare 0."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(y - p), den, out=np.zeros_like(den), where=den > 0)
    return float(100 * ratio.mean())


def directional_hit_rate(y, p) -> float:
    """% din intervale unde directia fata de intervalul precedent (sus/jos) e ghicita corect.
    Valorile trebuie sa fie in ordine cronologica; se compara doar intervale consecutive."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2:
        return float("nan")
    dy, dp = np.sign(np.diff(y)), np.sign(np.diff(p))
    return float(np.mean(dy == dp))


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def summarize(y, p, lo=None, hi=None) -> dict[str, float]:
    out = {"mae": mae(y, p), "rmse": rmse(y, p), "smape": smape(y, p),
           "directional_hit_rate": directional_hit_rate(y, p)}
    if lo is not None and hi is not None:
        out["coverage_90"] = coverage(y, lo, hi)
    return out


def summarize_by_day(y: pd.Series, p: pd.Series, tz: str = "Europe/Bucharest") -> pd.DataFrame:
    """Metrici pe zi calendaristica locala."""
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    days = df.index.tz_convert(tz).date
    rows = {d: summarize(g["y"], g["p"]) for d, g in df.groupby(days)}
    return pd.DataFrame(rows).T
