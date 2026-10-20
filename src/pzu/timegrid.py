"""Grila de timp: toate datele se normalizeaza la intervale de 15 min, indexate in UTC,
iar ziua de livrare se defineste in Europe/Bucharest (92/96/100 intervale)."""
import pandas as pd

from pzu.config import TZ

FREQ = "15min"


def day_index(day: pd.Timestamp | str) -> pd.DatetimeIndex:
    """Intervalele de 15 min ale unei zile calendaristice locale, in UTC."""
    d = pd.Timestamp(day)
    start = pd.Timestamp(d.date(), tz=TZ)
    end = pd.Timestamp((d + pd.Timedelta(days=1)).date(), tz=TZ)
    return pd.date_range(start, end, freq=FREQ, inclusive="left").tz_convert("UTC")


def to_15min(obj: pd.Series | pd.DataFrame, how: str = "ffill"):
    """Aduce o serie orara (sau mixta) la 15 min, in UTC.

    how='ffill' pentru preturi/capacitati (valoare constanta pe ora),
    how='interpolate' pentru vreme (variabile continue).
    """
    if obj.empty:
        return obj
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index).tz_convert("UTC")
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    if len(obj) < 2:
        return obj
    diffs = obj.index.to_series().diff().dropna()
    step = diffs.min()
    if step <= pd.Timedelta(FREQ):
        out = obj.resample(FREQ).mean()
        # serie cu rezolutii amestecate (ex. tranzitia orar -> 15 min): valorile orare se extind
        # pe maximum 3 sferturi; o serie pur 15-min ramane nefiltrata (golurile raman goluri)
        return out.ffill(limit=3) if diffs.max() > pd.Timedelta(FREQ) else out
    # extinde ultima ora pe 4 sferturi
    new_idx = pd.date_range(obj.index[0], obj.index[-1] + step - pd.Timedelta(FREQ), freq=FREQ)
    if how == "interpolate":
        return obj.reindex(new_idx).interpolate(limit_area="inside").ffill(limit=3)
    return obj.reindex(new_idx).ffill()


def to_local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).tz_convert(TZ)
