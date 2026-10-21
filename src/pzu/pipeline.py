"""Job-uri zilnice: forecast D+1, evaluare, reantrenare.

  python -m pzu.pipeline daily    [--date YYYY-MM-DD] [--force] [--final]
  python -m pzu.pipeline evaluate [--date YYYY-MM-DD] [--force] [--final]
  python -m pzu.pipeline retrain

Toate sunt idempotente: pot rula de mai multe ori pe zi (reincercari) fara efecte duble.
`--final` = ultima incercare a zilei: daca nu a iesit, procesul se termina cu eroare (alerta in Actions).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from pzu import backtest, db, features as F, metrics, model as M
from pzu.config import TZ, get_settings
from pzu.conformal import ConformalInterval
from pzu.ingest import entsoe
from pzu.timegrid import day_index

log = logging.getLogger("pipeline")
# Singura intrare D+1 obligatorie: prognoza de consum. Solar/eolian ENTSO-E se publica dupa gate closure.
REQUIRED_INPUTS = ["load_forecast_mw"]


class NotReady(RuntimeError):
    """Datele necesare nu sunt (inca) disponibile; se poate reincerca mai tarziu."""


def local_today(now: pd.Timestamp | None = None) -> date:
    return (now or pd.Timestamp.now(tz=TZ)).tz_convert(TZ).date()


def target_day(now: pd.Timestamp | None = None) -> date:
    return local_today(now) + timedelta(days=1)


# ---------------------------------------------------------------- ingestie ----
def _long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.stack().rename("price_eur_mwh").reset_index()
    long.columns = ["timestamp", "zone", "price_eur_mwh"]
    long["source"] = "ENTSOE"
    return long


def ingest_recent(engine, client, through_day: date, lookback_days: int = 10) -> None:
    """Aduce in DB ultimele zile de preturi + forecast-urile pana la `through_day` inclusiv.
    Nu suprascrie niciodata o valoare existenta cu lipsa (merge)."""
    start = pd.Timestamp(through_day) - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(through_day) + pd.Timedelta(days=1)
    prices = entsoe.fetch_all_prices(client, start, end)
    if not prices.empty:
        db.upsert_df(engine, db.prices_actual, _long(prices.dropna(how="all")), ["timestamp", "zone"])
    new = entsoe.fetch_inputs(client, start, end)
    if new.empty:
        return
    new = new[[c for c in db.INPUT_COLS if c in new.columns]]
    old = db.read_df(engine, db.forecasts_input)
    old = old[(old.index >= new.index.min()) & (old.index <= new.index.max())]
    merged = new.combine_first(old) if not old.empty else new
    db.upsert_df(engine, db.forecasts_input, merged, ["timestamp"])


# -------------------------------------------------------------- reantrenare ----
CALIBRATION_MAX_AGE_DAYS = 7   # intervalele de incredere (backtest 90 zile) se recalculeaza saptamanal


def _local_date(ts: str | None, now: pd.Timestamp):
    return None if not ts else pd.Timestamp(ts).tz_convert(TZ).date()


def refresh_model(engine, bundle: M.ModelBundle, now: pd.Timestamp | None = None,
                  before: pd.Timestamp | None = None) -> M.ModelBundle:
    """Reantrenare zilnica: modelul se refit-uieste o data pe zi pe ultimele `window_days` zile (secunde).
    Calibrarea intervalelor (backtest 90 zile, ~2 min) se refoloseste si se recalculeaza o data pe saptamana.
    Idempotent: dupa prima reantrenare a zilei, incercarile urmatoare nu mai fac nimic."""
    now = now or pd.Timestamp.now(tz=TZ)
    today = local_today(now)
    if _local_date(bundle.trained_at, now) == today:
        return bundle
    cal_day = _local_date(getattr(bundle, "calibrated_at", None), now)
    if cal_day is None or (today - cal_day).days >= CALIBRATION_MAX_AGE_DAYS:
        retrain(engine, None, bundle.window_days, now=now, before=before)   # reantrenare completa + recalibrare
        stored = db.load_model_bytes(engine)
        return M.ModelBundle.from_bytes(stored[1])
    X, y = F.build_training_set(db.read_df(engine, db.forecasts_input), db.read_prices(engine))
    if before is not None:   # nu antrenam niciodata pe ziua pe care o prognozam
        X, y = X[X.index < before], y[y.index < before]
    dates = pd.Series(X.index.tz_convert(TZ).date, index=X.index)
    keep = dates.isin(sorted(dates.unique())[-bundle.window_days:]).to_numpy()
    final = M.fit(X[keep], y[keep], cols=bundle.features)
    version = f"v2-{today:%Y%m%d}"
    fresh = M.ModelBundle(final, version, bundle.window_days, bundle.conformal,
                          features=bundle.features, calibrated_at=bundle.calibrated_at)
    db.save_model_bytes(engine, version, fresh.to_bytes())
    return fresh


# ------------------------------------------------------------------- daily ----
def check_inputs_ready(inputs: pd.DataFrame, day) -> None:
    idx = day_index(day)
    sub = inputs.reindex(idx)
    missing = [c for c in REQUIRED_INPUTS if c not in sub or sub[c].isna().any()]
    if missing:
        raise NotReady(f"Lipsesc date pentru {pd.Timestamp(day).date()}: {missing}")


def daily(engine, client, day: date | None = None, force: bool = False,
          now: pd.Timestamp | None = None) -> str:
    day = day or target_day(now)
    idx = day_index(day)
    if not force and db.count_forecast_rows(engine, idx[0], idx[-1]) == len(idx):
        return f"exista deja forecast pentru {day}"
    db.ensure_real_data(engine)
    stored = db.load_model_bytes(engine)
    if stored is None:
        raise RuntimeError("Nu exista model antrenat: ruleaza `python -m pzu.pipeline retrain`")
    bundle = M.ModelBundle.from_bytes(stored[1])

    ingest_recent(engine, client, day)
    inputs, prices = db.read_df(engine, db.forecasts_input), db.read_prices(engine)
    check_inputs_ready(inputs, day)
    try:   # reantrenare zilnica; la esec se foloseste modelul existent
        bundle = refresh_model(engine, bundle, now, before=idx[0])
    except Exception as exc:
        log.warning("Reantrenarea zilnica a esuat, folosesc modelul existent: %s", exc)
    inputs, prices = db.read_df(engine, db.forecasts_input), db.read_prices(engine)
    # la ora prognozei nu exista preturile din ziua tinta; le taiem explicit (si pentru re-rulari tarzii)
    prices = prices[prices.index < idx[0]]
    out = M.forecast_day(bundle, inputs, prices, day)

    generated = pd.Timestamp.now(tz="UTC").to_pydatetime()
    rows = out.reset_index(names="timestamp")
    rows = rows.rename(columns={"predicted_price": "predicted_price"})
    rows["forecast_generated_at"] = generated
    rows["features_used"] = [{"features": bundle.features, "window_days": bundle.window_days}] * len(rows)
    db.upsert_df(engine, db.price_forecast, rows, ["timestamp"])
    return f"forecast {day} salvat ({len(rows)} intervale, model {bundle.version})"


# ---------------------------------------------------------------- evaluate ----
def evaluate(engine, client, day: date | None = None, force: bool = False,
             now: pd.Timestamp | None = None) -> str:
    day = day or target_day(now)
    idx = day_index(day)
    with engine.connect() as conn:
        from sqlalchemy import select
        done = conn.execute(select(db.forecast_accuracy.c.forecast_date)
                            .where(db.forecast_accuracy.c.forecast_date == day)).first()
    if done and not force:
        return f"evaluare existenta pentru {day}"
    fc = db.read_forecast(engine, idx)
    if len(fc) != len(idx):
        raise NotReady(f"Nu exista forecast complet pentru {day}")
    actual = entsoe.fetch_prices(client, "RO", pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1))
    actual = actual.reindex(idx)
    if actual.isna().any():
        raise NotReady(f"Preturile oficiale pentru {day} nu sunt (complet) publicate")
    db.upsert_df(engine, db.prices_actual, _long(actual.to_frame("RO")), ["timestamp", "zone"])
    m = metrics.summarize(actual, fc["predicted_price"], fc["ci_lower"], fc["ci_upper"])
    row = pd.DataFrame([{"forecast_date": day, **m, "model_version": fc["model_version"].iloc[0]}])
    db.upsert_df(engine, db.forecast_accuracy, row, ["forecast_date"])
    return f"evaluare {day}: MAE {m['mae']:.2f}, RMSE {m['rmse']:.2f}, coverage {m['coverage_90']:.2f}"


# ----------------------------------------------------------------- retrain ----
def retrain(engine, client=None, window_days: int = 60, calib_days: int = 90,
            version: str | None = None, now: pd.Timestamp | None = None,
            before: pd.Timestamp | None = None) -> str:
    """`before`: antrenarea/calibrarea folosesc doar date anterioare (ex. inceputul zilei prognozate)."""
    db.ensure_real_data(engine)
    if client is not None:
        ingest_recent(engine, client, local_today(now), lookback_days=14)
    X, y = F.build_training_set(db.read_df(engine, db.forecasts_input), db.read_prices(engine))
    if before is not None:
        X, y = X[X.index < before], y[y.index < before]
    if len(X) == 0:
        raise RuntimeError("Nu exista date de antrenare (ruleaza scripts/backfill.py)")
    bt = backtest.walk_forward(X, y, window_days, calib_days)
    conf = ConformalInterval(0.10, by_hour=True).fit(bt["y"] - bt["pred"])
    dates = pd.Series(X.index.tz_convert(TZ).date, index=X.index)
    keep = dates.isin(sorted(dates.unique())[-window_days:]).to_numpy()
    final = M.fit(X[keep], y[keep])
    version = version or f"v2-{local_today(now):%Y%m%d}"
    bundle = M.ModelBundle(final, version, window_days, conf, calibrated_at=pd.Timestamp.now(tz="UTC").isoformat())
    db.save_model_bytes(engine, version, bundle.to_bytes())
    m = backtest.evaluate(bt)["lightgbm"]
    return f"model {version} salvat (backtest {calib_days}z: MAE {m['mae']:.2f}, coverage {m.get('coverage_90', float('nan')):.2f})"


# --------------------------------------------------------------------- CLI ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pzu.pipeline")
    ap.add_argument("job", choices=["daily", "evaluate", "retrain"])
    ap.add_argument("--date", type=date.fromisoformat)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--final", action="store_true", help="ultima incercare: esec = cod de iesire 1")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = db.get_engine()
    db.init_db(engine)
    try:
        client = entsoe.make_client(get_settings().entsoe_token)
        if a.job == "daily":
            print(daily(engine, client, a.date, a.force))
        elif a.job == "evaluate":
            print(evaluate(engine, client, a.date, a.force))
        else:
            print(retrain(engine, client))
    except NotReady as exc:
        print(f"[inca nu e gata] {exc}")
        return 1 if a.final else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
