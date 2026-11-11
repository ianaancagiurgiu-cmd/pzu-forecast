"""Incarca in DB (weather_fc) prognozele de vreme arhivate descarcate de `fetch_weather_history.py`.

  python scripts/import_weather.py weather_ro.csv weather_centru.csv weather_vest.csv
"""
import sys

import pandas as pd

from pzu import db

RENAME = {"fc_temperature_2m": "temp_c", "fc_wind_speed_100m": "wind_ms", "fc_shortwave_radiation": "rad_wm2",
          "fc_cloud_cover": "cloud_pct"}


def main(paths: list[str]) -> None:
    eng = db.get_engine()
    db.init_db(eng)
    for path in paths:
        df = pd.read_csv(path, parse_dates=["ts_utc"])
        df = df.rename(columns={"ts_utc": "timestamp", **RENAME})[["timestamp", "point", *RENAME.values()]].dropna(how="any")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        n = db.upsert_df(eng, db.weather_fc, df, ["timestamp", "point"])
        print(path, n, "randuri", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
