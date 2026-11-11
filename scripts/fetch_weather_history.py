"""Istoric de vreme pentru testul vreme -> pret: prognoze ARHIVATE (ce se stia dinainte) si vreme REALA (ERA5),
pe puncte din Romania si din regiune (Germania conteaza pentru pretul regional). Open-Meteo, gratuit.
Rulat in GitHub Actions (IP-ul sesiunii de dezvoltare e partajat si atinge limita zilnica).

  python scripts/fetch_weather_history.py --months 25   # scrie weather_hist.csv
"""
import argparse
import time

import httpx
import pandas as pd

HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARS = "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"
# nume: (lat, lon). RO ca in config; DE_N = vant (Marea Nordului/Baltica), DE_S = solar, HU = vecin
POINTS = {
    "RO_dobrogea": (44.18, 28.65), "RO_buc": (44.43, 26.10), "RO_craiova": (44.32, 23.80), "RO_cluj": (46.77, 23.60),
    "DE_nord": (54.0, 9.5), "DE_centru": (51.0, 10.0), "DE_sud": (48.4, 11.5), "HU": (47.2, 19.0),
}


def get(client, url, params, tries=8):
    for k in range(tries):
        try:
            r = client.get(url, params=params)
        except httpx.TransportError:
            time.sleep(10 * (k + 1)); continue
        if r.status_code == 200:
            return r.json()
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
    a = ap.parse_args()
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    start = end - pd.DateOffset(months=a.months)
    out = []
    with httpx.Client(timeout=60) as client:
        for name, (lat, lon) in POINTS.items():
            fc = fetch(client, HIST_FC, lat, lon, start, end, "fc_")
            ac = fetch(client, ARCHIVE, lat, lon, start, end - pd.Timedelta(days=2), "act_")
            df = fc.join(ac, how="outer")
            df.insert(0, "point", name)
            out.append(df)
            print(name, len(df), flush=True)
    res = pd.concat(out).rename_axis("ts_utc")
    res.to_csv("weather_hist.csv")
    print("total", len(res))


if __name__ == "__main__":
    main()
