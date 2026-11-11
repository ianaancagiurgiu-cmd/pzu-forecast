"""Istoric de vreme pentru testul vreme -> pret: prognoze ARHIVATE (ce se stia dinainte) si vreme REALA (ERA5),
pe puncte din Romania si din regiune (Germania conteaza pentru pretul regional). Open-Meteo, gratuit.
Rulat in GitHub Actions (IP-ul sesiunii de dezvoltare e partajat si atinge limita zilnica).

  python scripts/fetch_weather_history.py --group ro|centru|vest --months 25   # scrie weather_<grup>.csv
"""
import argparse
import time

import httpx
import pandas as pd

HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARS = "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"
# nume: (lat, lon). RO ca in config; DE_N = vant (Marea Nordului/Baltica), DE_S = solar, HU = vecin
GROUPS = {
    # Romania si vecini apropiati (consum, solar, eolian Dobrogea)
    "ro": {"RO_dobrogea": (44.18, 28.65), "RO_buc": (44.43, 26.10), "RO_craiova": (44.32, 23.80), "RO_cluj": (46.77, 23.60),
           "HU": (47.2, 19.0), "BG": (42.7, 25.5), "RS": (44.5, 20.5), "GR": (39.0, 22.5)},
    # Europa Centrala si de Nord (vant, solar, frig -> consum)
    "centru": {"DE_nord": (54.0, 9.5), "DE_centru": (51.0, 10.0), "DE_sud": (48.4, 11.5), "PL": (52.0, 19.5),
               "CZ_AT": (48.5, 15.0), "DK": (56.0, 10.0), "NL": (52.5, 5.0), "NO": (60.0, 10.0), "SE": (62.0, 15.0)},
    # Vest si Sud (Franta, Iberia, Italia, Regatul Unit)
    "vest": {"FR": (46.5, 2.5), "ES": (40.0, -3.5), "IT_nord": (45.2, 9.5), "IT_sud": (40.5, 16.5), "GB": (53.5, -1.5),
             "IE": (53.0, -8.0)},
}


def get(client, url, params, tries=8):
    for k in range(tries):
        try:
            r = client.get(url, params=params)
        except httpx.TransportError:
            time.sleep(10 * (k + 1)); continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:   # raspuns gol / HTML cu cod 200: reincercam
                print("raspuns invalid", r.text[:80], flush=True)
                time.sleep(15 * (k + 1)); continue
        print("HTTP", r.status_code, r.text[:120], flush=True)
        time.sleep(min(20 * 2 ** k, 600))
    raise RuntimeError(f"esec {url} {params}")


def fetch(client, url, lat, lon, start, end, prefix):
    parts = []
    for s in pd.date_range(start, end, freq="120D"):
        e = min(s + pd.Timedelta(days=119), end)
        j = get(client, url, {"latitude": lat, "longitude": lon, "hourly": VARS, "start_date": s.date().isoformat(),
                              "end_date": e.date().isoformat(), "timezone": "UTC", "wind_speed_unit": "ms"})
        h = j["hourly"]
        df = pd.DataFrame({k: h[k] for k in VARS.split(",")}, index=pd.to_datetime(h["time"]))
        parts.append(df.add_prefix(prefix))
        time.sleep(1)
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=25)
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    a = ap.parse_args()
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    start = end - pd.DateOffset(months=a.months)
    out = []
    with httpx.Client(timeout=60) as client:
        for name, (lat, lon) in GROUPS[a.group].items():
            fc = fetch(client, HIST_FC, lat, lon, start, end, "fc_")
            ac = fetch(client, ARCHIVE, lat, lon, start, end - pd.Timedelta(days=2), "act_")
            df = fc.join(ac, how="outer")
            df.insert(0, "point", name)
            out.append(df)
            print(name, len(df), flush=True)
    res = pd.concat(out).rename_axis("ts_utc")
    res.to_csv(f"weather_{a.group}.csv")
    print("total", len(res))


if __name__ == "__main__":
    main()
