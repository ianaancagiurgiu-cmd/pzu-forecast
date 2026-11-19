"""Situatii neobisnuite intr-o zi de livrare si posibile explicatii, deduse din date.

Regulile sunt simple si transparente (praguri fixe); textul spune "posibila explicatie", nu o certitudine:
nu avem ofertele individuale si nici capacitatea liniilor pentru fiecare ora.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from pzu import db
from pzu.config import TZ
from pzu.timegrid import day_index

ZONE_NAMES = {"RO": "România", "HU": "Ungaria", "BG": "Bulgaria", "RS": "Serbia", "SK": "Slovacia", "AT": "Austria",
              "DE": "Germania", "CZ": "Cehia", "HR": "Croația", "SI": "Slovenia", "GR": "Grecia"}
ZONES = list(ZONE_NAMES)

DECOUPLE_EUR = 40.0      # |RO - HU| peste acest prag, cel putin 2 intervale la rand = preturi separate
NEAR_ZERO_EUR = 10.0
SPIKE_EUR = 400.0
DAY_SHIFT = 0.25         # schimbare a pretului mediu fata de ziua precedenta (25%) ...
DAY_SHIFT_EUR = 50.0     # ... si de cel putin 50 EUR/MWh
BIG_MISS_EUR = 50.0      # eroare medie a zilei peste care ziua e "greu de prognozat"


@dataclass
class Event:
    kind: str            # decouple_low | decouple_high | near_zero | spike | day_shift | big_miss
    severity: str        # info | warn
    start: str | None    # HH:MM (ora Romaniei); None = toata ziua
    end: str | None
    title: str
    text: str


def _f(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.0f}".replace(",", ".")


def _hhmm(ts) -> str:
    return pd.Timestamp(ts).tz_convert(TZ).strftime("%H:%M")


def region_prices(engine: Engine, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Preturile tuturor zonelor intre `start` si `end` (inclusiv), format lat, index UTC."""
    with engine.connect() as conn:
        df = pd.read_sql(select(db.prices_actual.c.timestamp, db.prices_actual.c.zone, db.prices_actual.c.price_eur_mwh)
                         .where(db.prices_actual.c.timestamp >= start.to_pydatetime(),
                                db.prices_actual.c.timestamp <= end.to_pydatetime()), conn)
    if df.empty:
        return pd.DataFrame(columns=ZONES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    wide = df.pivot_table(index="timestamp", columns="zone", values="price_eur_mwh", aggfunc="last").sort_index()
    return wide.reindex(columns=[z for z in ZONES if z in wide.columns])


def _range(engine: Engine, table, cols, start, end) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(select(table.c.timestamp, *[table.c[c] for c in cols]).where(
            table.c.timestamp >= start.to_pydatetime(), table.c.timestamp <= end.to_pydatetime()), conn)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index().astype(float)


def _weather_daily(engine: Engine, days: list) -> pd.DataFrame:
    """Medii zilnice (ora locala) ale vremii pe grupe: vant NV Europa, soare Europa Centrala, temperatura UE."""
    from pzu import weather as W
    start = pd.Timestamp(min(days), tz=TZ).tz_convert("UTC")
    end = (pd.Timestamp(max(days), tz=TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    with engine.connect() as conn:
        df = pd.read_sql(select(db.weather_fc).where(db.weather_fc.c.timestamp >= start.to_pydatetime(),
                                                     db.weather_fc.c.timestamp < end.to_pydatetime()), conn)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    g = W.group_means(df)
    if g.empty:
        return g
    return g.groupby(g.index.tz_convert(TZ).date).mean()


def _context(engine: Engine, a: pd.Timestamp, b: pd.Timestamp) -> str:
    """Ce se intampla in Romania intre a si b: solar, eolian (real) si consum (prognoza)."""
    gen = _range(engine, db.generation_actual, ["gen_solar_mw", "gen_wind_mw"], a, b)
    load = _range(engine, db.forecasts_input, ["load_forecast_mw"], a, b)
    parts = []
    if len(gen) and gen["gen_solar_mw"].fillna(0).max() >= 20:
        s = gen["gen_solar_mw"].dropna()
        parts.append(f"solar {_f(s.iloc[0])} → {_f(s.iloc[-1])} MW" if len(s) > 1 else f"solar {_f(s.iloc[0])} MW")
    if len(gen) and gen["gen_wind_mw"].notna().any():
        parts.append(f"eolian ~{_f(gen['gen_wind_mw'].mean())} MW")
    if len(load) and load["load_forecast_mw"].notna().any():
        l = load["load_forecast_mw"].dropna()
        parts.append(f"consum {_f(l.iloc[0])} → {_f(l.iloc[-1])} MW" if len(l) > 1 else f"consum {_f(l.iloc[0])} MW")
    return ("În România, în acest interval: " + ", ".join(parts) + ".") if parts else ""


def _segments(mask: pd.Series, min_len: int = 2, gap: int = 2) -> list[tuple[int, int]]:
    """Secvente True (indici inclusivi) de lungime >= min_len; pauzele de cel mult `gap` intervale se unesc."""
    runs, start = [], None
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        if (not v or i == len(vals) - 1) and start is not None:
            runs.append([start, i if v else i - 1])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a + 1 >= min_len]


def day_events(engine: Engine, day, forecast_mae: float | None = None) -> list[dict]:
    """Situatiile neobisnuite ale zilei `day` (dupa ce preturile oficiale sunt publicate)."""
    idx = day_index(day)
    step = pd.Timedelta(minutes=15)
    prev_idx = day_index(pd.Timestamp(day) - pd.Timedelta(days=1))
    p = region_prices(engine, prev_idx[0], idx[-1])
    if p.empty or "RO" not in p or p["RO"].reindex(idx).isna().sum() > 4:
        return []
    cur = p.reindex(idx)
    ro = cur["RO"]
    events: list[Event] = []

    # 1) Romania se separa de Ungaria (de obicei au acelasi pret: liniile dintre ele nu sunt pline)
    if "HU" in cur:
        diff = ro - cur["HU"]
        for sign, kind in ((-1, "decouple_low"), (1, "decouple_high")):
            for i, j in _segments((sign * diff >= DECOUPLE_EUR).fillna(False)):
                a, b = idx[i], idx[j]
                seg = cur.iloc[i:j + 1]
                r, h = seg["RO"].mean(), seg["HU"].mean()
                others = {z: seg[z].mean() for z in ("BG", "RS") if z in seg and seg[z].notna().any()}
                ctx = _context(engine, a, b)
                if sign < 0:
                    title = f"România mult mai ieftină decât Ungaria ({_f(r)} față de {_f(h)} €/MWh)"
                    text = ("Prețul României s-a separat de cel al Ungariei. Posibilă explicație: România avea mai multă "
                            "energie decât putea consuma, iar liniile de export spre vest erau pline, deci surplusul nu "
                            "putea fi vândut la prețul vecinilor.")
                    if seg["RO"].min() <= NEAR_ZERO_EUR:
                        text += (f" Prețul a coborât la {_f(seg['RO'].min())} €/MWh: au rămas doar ofertele "
                                 "producătorilor care vând la orice preț (regenerabile, centrale care nu se pot opri).")
                    if others and all(v < h - DECOUPLE_EUR for v in others.values()):
                        text += " Bulgaria/Serbia au fost și ele ieftine (" + ", ".join(
                            f"{ZONE_NAMES[z]} {_f(v)}" for z, v in others.items()) + "): surplus în tot sud-estul."
                else:
                    title = f"România mult mai scumpă decât Ungaria ({_f(r)} față de {_f(h)} €/MWh)"
                    text = ("Prețul României s-a separat în sus de cel al Ungariei. Posibilă explicație: importul era la "
                            "limită (consum mare sau producție internă mică), deci prețul s-a format din ofertele "
                            "interne, mai scumpe.")
                events.append(Event(kind, "warn", _hhmm(a), _hhmm(b + step), title, (text + " " + ctx).strip()))

    # 2) pret aproape de zero sau negativ, fara separare de Ungaria = surplus in toata regiunea
    low = ro <= NEAR_ZERO_EUR
    if "HU" in cur:
        low &= (ro - cur["HU"]).abs() < DECOUPLE_EUR
    for i, j in _segments(low.fillna(False), min_len=1):
        a, b = idx[i], idx[j]
        events.append(Event("near_zero", "warn", _hhmm(a), _hhmm(b + step),
                            f"Preț aproape de zero ({_f(ro.iloc[i:j + 1].min())} €/MWh) în toată regiunea",
                            ("Și vecinii au avut prețuri foarte mici. Posibilă explicație: producție mare din soare și vânt "
                             "în regiune, peste consum. " + _context(engine, a, b)).strip()))

    # 3) varfuri de pret
    med = ro.median()
    spike = (ro >= max(SPIKE_EUR, 1.6 * med)).fillna(False)
    for i, j in _segments(spike, min_len=1):
        a, b = idx[i], idx[j]
        seg = cur.iloc[i:j + 1]
        regional = "HU" in seg and (seg["HU"] >= 0.8 * seg["RO"]).all()
        events.append(Event("spike", "warn", _hhmm(a), _hhmm(b + step),
                            f"Vârf de preț: {_f(seg['RO'].max())} €/MWh",
                            (("Vârful a fost în toată regiunea (Ungaria " + _f(seg["HU"].max()) + "), deci cauza e regională: "
                              "cerere mare și producție regenerabilă mică în Europa. " if regional else
                              "Vârful a fost mai ales în România. Posibilă explicație: consum mare sau producție internă "
                              "redusă, cu importul la limită. ") + _context(engine, a, b)).strip()))

    # 4) schimbare mare a nivelului fata de ziua precedenta (de obicei regionala, legata de vreme)
    prev = p["RO"].reindex(prev_idx)
    if prev.notna().sum() > 90:
        m0, m1 = prev.mean(), ro.mean()
        if abs(m1 - m0) >= DAY_SHIFT_EUR and abs(m1 - m0) / max(m0, 1) >= DAY_SHIFT:
            up = m1 > m0
            de0 = p["DE"].reindex(prev_idx).mean() if "DE" in p else np.nan
            de1 = cur["DE"].mean() if "DE" in cur else np.nan
            text = f"Prețul mediu a {'crescut' if up else 'scăzut'} de la {_f(m0)} la {_f(m1)} €/MWh față de ziua precedentă."
            if pd.notna(de0) and pd.notna(de1):
                text += f" În Germania: de la {_f(de0)} la {_f(de1)}."
            wx = _weather_daily(engine, [prev_idx[0].tz_convert(TZ).date(), pd.Timestamp(day).date()])
            d0, d1 = prev_idx[0].tz_convert(TZ).date(), pd.Timestamp(day).date()
            if not wx.empty and d0 in wx.index and d1 in wx.index:
                w0, w1 = wx.loc[d0], wx.loc[d1]
                hints = []
                if "wind_nw" in wx and abs(w1["wind_nw"] - w0["wind_nw"]) >= 1.0:
                    hints.append(f"vântul din nord-vestul Europei {'a scăzut' if w1['wind_nw'] < w0['wind_nw'] else 'a crescut'} "
                                 f"de la {w0['wind_nw']:.1f} la {w1['wind_nw']:.1f} m/s")
                if "rad_c" in wx and abs(w1["rad_c"] - w0["rad_c"]) >= 15:
                    hints.append(f"soarele din Europa Centrală {'a scăzut' if w1['rad_c'] < w0['rad_c'] else 'a crescut'} "
                                 f"({w0['rad_c']:.0f} → {w1['rad_c']:.0f} W/m²)")
                if "t_eu" in wx and abs(w1["t_eu"] - w0["t_eu"]) >= 2.0:
                    hints.append(f"temperatura medie în Europa {'a scăzut' if w1['t_eu'] < w0['t_eu'] else 'a crescut'} "
                                 f"({w0['t_eu']:.1f} → {w1['t_eu']:.1f} °C)")
                if hints:
                    text += (" Posibilă explicație, din prognoza de vreme: " + "; ".join(hints) + ". Mai puțin vânt sau soare "
                             "și frig înseamnă mai multă producție pe gaz, deci prețuri mai mari, și invers.")
            events.append(Event("day_shift", "info", None, None,
                                f"Prețul mediu {'a crescut' if up else 'a scăzut'} brusc ({_f(m0)} → {_f(m1)} €/MWh)", text))

    # 5) zi greu de prognozat
    if forecast_mae is not None and forecast_mae >= BIG_MISS_EUR:
        events.append(Event("big_miss", "info", None, None, f"Zi greu de prognozat (eroare medie {_f(forecast_mae)} €/MWh)",
                            "Eroarea a fost mult peste obișnuit. Cauzele probabile sunt situațiile de mai sus: ele depind "
                            "de oferte și de capacitatea liniilor, care nu se cunosc dimineața, la ora prognozei."))
    return [asdict(e) for e in sorted(events, key=lambda e: e.start or "99")]
