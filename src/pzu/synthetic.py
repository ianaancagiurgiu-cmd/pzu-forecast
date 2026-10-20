"""Date sintetice cu structura plauzibila, pentru dezvoltare si teste fara token ENTSO-E.
NU reprezinta piata reala: backtest-ul cu aceste date nu spune nimic despre acuratetea reala."""
import numpy as np
import pandas as pd

from pzu.config import NEIGHBORS, TZ
from pzu.timegrid import day_index


def generate(start: str, days: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returneaza (inputs, prices) pe `days` zile pornind de la `start` (ora locala)."""
    rng = np.random.default_rng(seed)
    idx = day_index(start)
    for k in range(1, days):
        idx = idx.append(day_index(pd.Timestamp(start) + pd.Timedelta(days=k)))
    n = len(idx)
    local = idx.tz_convert(TZ)
    h = (local.hour + local.minute / 60).to_numpy()
    doy = local.dayofyear.to_numpy()
    dow = local.dayofweek.to_numpy()
    day_no = np.floor(np.arange(n) / 96).astype(int)

    def ar1(rho, scale, size):
        x = np.zeros(size)
        e = rng.normal(0, scale, size)
        for i in range(1, size):
            x[i] = rho * x[i - 1] + e[i]
        return x

    season = np.cos(2 * np.pi * (doy - 200) / 365)          # +1 vara, -1 iarna
    temp = 10 + 12 * season + 4 * np.sin(2 * np.pi * (h - 9) / 24) + ar1(0.999, 0.15, n)
    cloud = np.clip(0.75 + ar1(0.998, 0.02, n), 0.15, 1.0)
    sun = np.clip(np.sin(np.pi * (h - 6) / 14), 0, None)
    radiation = 850 * sun * cloud * (0.6 + 0.4 * season)
    wind_speed = np.clip(6 + ar1(0.998, 0.15, n), 0.5, None)
    solar = 1800 * radiation / 850 * 1.0
    wind = np.clip(60 * wind_speed ** 2, 0, 3200)
    load = (6200 + 1100 * np.sin(2 * np.pi * (h - 14) / 24 + 0.8) - 120 * (dow >= 5) * 6
            - 40 * (temp - 15).clip(max=0) + 25 * (temp - 22).clip(min=0) + rng.normal(0, 60, n))
    residual = load - solar - wind

    daily_shock = rng.normal(0, 12, days)[day_no]
    price_ro = (70 + 0.03 * (residual - 4500) + daily_shock
                + 25 * ((h >= 18) & (h < 22)) + rng.normal(0, 6, n))
    price_ro = np.clip(price_ro, -20, 400)
    prices = pd.DataFrame({"RO": price_ro}, index=idx)
    for z in NEIGHBORS:
        prices[z] = price_ro + rng.normal(0, 7, n) + rng.normal(0, 4, days)[day_no]

    inputs = pd.DataFrame({
        "solar_forecast_mw": solar * (1 + rng.normal(0, 0.05, n)),
        "wind_forecast_mw": wind * (1 + rng.normal(0, 0.08, n)),
        "load_forecast_mw": load + rng.normal(0, 40, n),
        "capacity_hu_ro_mw": 500 + 100 * rng.random(n),
        "capacity_ro_hu_mw": 600 + 100 * rng.random(n),
        "capacity_bg_ro_mw": 400 + 100 * rng.random(n),
        "capacity_ro_bg_mw": 450 + 100 * rng.random(n),
        "temp_c": temp, "wind_speed_ms": wind_speed, "radiation_wm2": radiation,
    }, index=idx)
    return inputs, prices
