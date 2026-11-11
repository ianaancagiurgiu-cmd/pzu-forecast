"""Vremea din toata Europa ca indicator pentru pret (prognoza D+1, disponibila dimineata).

Pretul Romaniei se formeaza regional: vantul din Europa de Nord-Vest si soarele din Europa Centrala conteaza mai mult
decat vremea de la noi. Testat walk-forward pe 479 de zile (README): cu prognoze arhivate, MAE 25,76 -> 24,96, cu
vremea doar din Romania a iesit mai rau (26,27). Punctele sunt agregate in cateva grupe (medii).
"""
from __future__ import annotations

import pandas as pd

from pzu.config import TZ

# nume: (lat, lon)
POINTS = {
    "RO_dobrogea": (44.18, 28.65), "RO_buc": (44.43, 26.10), "RO_craiova": (44.32, 23.80), "RO_cluj": (46.77, 23.60),
    "HU": (47.2, 19.0), "BG": (42.7, 25.5), "RS": (44.5, 20.5), "GR": (39.0, 22.5),
    "DE_nord": (54.0, 9.5), "DE_centru": (51.0, 10.0), "DE_sud": (48.4, 11.5), "PL": (52.0, 19.5),
    "CZ_AT": (48.5, 15.0), "DK": (56.0, 10.0), "NL": (52.5, 5.0), "NO": (60.0, 10.0), "SE": (62.0, 15.0),
    "FR": (46.5, 2.5), "ES": (40.0, -3.5), "IT_nord": (45.2, 9.5), "IT_sud": (40.5, 16.5), "GB": (53.5, -1.5),
    "IE": (53.0, -8.0),
}
# grup: (variabila, puncte)
GROUPS = {
    "wind_nw": ("wind_ms", ["DE_nord", "DK", "NL", "GB", "IE", "PL"]),
    "wind_s": ("wind_ms", ["IT_sud", "ES", "GR", "BG", "RO_dobrogea"]),
    "rad_c": ("rad_wm2", ["DE_sud", "DE_centru", "CZ_AT", "FR", "IT_nord", "HU"]),
    "rad_se": ("rad_wm2", ["RO_buc", "RO_craiova", "BG", "RS", "GR"]),
    "t_eu": ("temp_c", ["DE_centru", "PL", "FR", "CZ_AT", "IT_nord", "HU", "RO_buc", "BG", "GB", "SE"]),
    "t_ro": ("temp_c", ["RO_dobrogea", "RO_buc", "RO_craiova", "RO_cluj"]),
    "t_nord": ("temp_c", ["SE", "NO", "DK", "PL"]),
    "cloud_c": ("cloud_pct", ["DE_sud", "DE_centru", "CZ_AT", "FR", "PL"]),
}
VARIABLES = ["temp_c", "wind_ms", "rad_wm2", "cloud_pct"]
FEATURE_NAMES = ([f"wx_{g}" for g in GROUPS] + [f"wx_{g}_zi" for g in GROUPS]
                 + [f"wx_{c}_chg" for c in ("wind_nw", "rad_c", "t_eu")] + ["wx_wind_nw_max"])


def group_means(points: pd.DataFrame) -> pd.DataFrame:
    """points: format lung (timestamp UTC, point, temp_c, wind_ms, rad_wm2, cloud_pct) -> medii pe grupe, orar."""
    if points.empty:
        return pd.DataFrame()
    out = {}
    for g, (var, names) in GROUPS.items():
        sub = points[points["point"].isin(names)]
        if sub.empty:
            continue
        out[g] = sub.pivot(index="timestamp", columns="point", values=var).mean(axis=1)
    return pd.DataFrame(out).sort_index()


def weather_features(groups: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Variabilele pentru fiecare interval din `idx` (UTC, 15 min). Mediile zilnice (ora locala) si schimbarea fata de
    ziua precedenta se calculeaza pe toata perioada disponibila, apoi se aleg intervalele cerute."""
    out = pd.DataFrame(index=idx, columns=FEATURE_NAMES, dtype=float)
    if groups.empty or any(g not in groups for g in GROUPS):
        return out
    w15 = groups.resample("15min").interpolate("time", limit=8)
    day = pd.Series(w15.index.tz_convert(TZ).date, index=w15.index)
    full = pd.DataFrame(index=w15.index)
    for g in GROUPS:
        full[f"wx_{g}"] = w15[g]
        full[f"wx_{g}_zi"] = w15[g].groupby(day).transform("mean")
    for c in ("wind_nw", "rad_c", "t_eu"):
        dmean = w15[c].groupby(day).mean()
        full[f"wx_{c}_chg"] = (day.map(dmean) - day.map(dmean.shift(1))).astype(float)
    full["wx_wind_nw_max"] = w15["wind_nw"].groupby(day).transform("max")
    return full[FEATURE_NAMES].reindex(idx)
