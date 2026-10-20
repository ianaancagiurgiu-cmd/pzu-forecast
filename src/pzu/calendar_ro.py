import holidays
import pandas as pd

from pzu.config import TZ


def is_holiday(idx: pd.DatetimeIndex) -> pd.Series:
    """Sarbatori legale Romania, dupa data locala."""
    local = pd.DatetimeIndex(idx).tz_convert(TZ)
    years = sorted(set(local.year))
    cal = holidays.country_holidays("RO", years=years)
    return pd.Series([d in cal for d in local.date], index=idx, dtype=bool)
