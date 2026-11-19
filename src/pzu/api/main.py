"""API REST + dashboard. `create_app(engine)` permite teste cu DB temporara."""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.engine import Engine

from pzu import db, insights
from pzu.config import TZ
from pzu.timegrid import day_index

STATIC = Path(__file__).parent / "static"
LOW_PRICE, HIGH_PRICE = 30.0, 150.0   # praguri pentru indicatori (EUR/MWh)


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ).isoformat()


def _round(x) -> float:
    return round(float(x), 2)


def _opt(row, key):
    v = row.get(key)
    return None if v is None or pd.isna(v) else round(float(v), 2)


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine or db.get_engine()
    db.init_db(engine)
    app = FastAPI(title="Forecast PZU Romania D+1", version="1.0")

    def load_day(day: date) -> pd.DataFrame:
        fc = db.read_forecast(engine, day_index(day))
        if fc.empty:
            raise HTTPException(404, f"Nu exista forecast pentru {day}")
        return fc

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/forecast/pzu")
    def forecast(date: date = Query(..., description="Ziua de livrare (D+1), YYYY-MM-DD")):
        fc = load_day(date)
        step = pd.Timedelta(minutes=15)
        actual = db.read_actual_ro(engine, day_index(date))   # NaN pana la publicarea pretului oficial
        meta = fc["features_used"].iloc[0] if "features_used" in fc else None
        if isinstance(meta, str):
            meta = json.loads(meta)
        hist = db.read_forecast_history(engine, day_index(date))
        return {
            "variant": (meta or {}).get("variant"),
            "load_estimated": bool((meta or {}).get("load_estimated")),
            "updates": int(hist["generated_at"].nunique()) if not hist.empty else 1,
            "date": date.isoformat(),
            "generated_at": _iso(pd.to_datetime(fc["forecast_generated_at"], utc=True).max()),
            "model_version": fc["model_version"].iloc[0],
            "intervals": [
                {"start": _iso(ts), "end": _iso(ts + step),
                 "predicted_price": _round(r.predicted_price),
                 "ci_90_lower": _round(r.ci_lower), "ci_90_upper": _round(r.ci_upper),
                 "mean_7d_price": _opt(r, "mean7d_price"), "blend_price": _opt(r, "blend_price"),
                 "actual_price": None if pd.isna(actual.get(ts)) else round(float(actual.get(ts)), 2)}
                for ts, r in fc.iterrows()
            ],
        }

    @app.get("/api/v1/forecast/pzu/versions")
    def versions(date: date):
        """Toate versiunile prognozei pentru zi (ora, varianta) si, daca exista pretul oficial, eroarea fiecareia."""
        idx = day_index(date)
        h = db.read_forecast_history(engine, idx)
        actual = db.read_actual_ro(engine, idx)
        out = []
        for gen, g in h.groupby("generated_at"):
            g = g.set_index("timestamp")
            row = {"generated_at": _iso(gen), "variant": g["variant"].iloc[0], "model_version": g["model_version"].iloc[0],
                   "mean_price": _round(g["predicted_price"].mean()), "mae": None, "mae_mix": None}
            a = actual.reindex(g.index)
            if len(g) == len(idx) and a.notna().all():
                row["mae"] = _round((g["predicted_price"] - a).abs().mean())
                if g["blend_price"].notna().all():
                    row["mae_mix"] = _round((g["blend_price"] - a).abs().mean())
            out.append(row)
        return {"date": date.isoformat(), "versions": out}

    @app.get("/api/v1/forecast/pzu/summary")
    def summary(date: date):
        fc = load_day(date)
        p = fc["predicted_price"]
        return {
            "date": date.isoformat(),
            "mean": _round(p.mean()), "min": _round(p.min()), "max": _round(p.max()),
            "spread": _round(p.max() - p.min()),
            "intervals_below_30": int((p < LOW_PRICE).sum()),
            "intervals_above_150": int((p > HIGH_PRICE).sum()),
            "hours_below_30": float((p < LOW_PRICE).sum() / 4),
            "hours_above_150": float((p > HIGH_PRICE).sum() / 4),
        }

    @app.get("/api/v1/forecast/pzu/export")
    def export(date: date, format: str = Query("csv", pattern="^(csv|xlsx)$")):
        fc = load_day(date)
        df = pd.DataFrame({
            "start": [_iso(ts) for ts in fc.index],
            "predicted_price": fc["predicted_price"].round(2).to_numpy(),
            "ci_90_lower": fc["ci_lower"].round(2).to_numpy(),
            "ci_90_upper": fc["ci_upper"].round(2).to_numpy(),
            "diff_vs_daily_mean": (fc["predicted_price"] - fc["predicted_price"].mean()).round(2).to_numpy(),
            "mean_7d_price": fc["mean7d_price"].round(2).to_numpy() if "mean7d_price" in fc else None,
            "blend_price": fc["blend_price"].round(2).to_numpy() if "blend_price" in fc else None,
        })
        name = f"forecast_pzu_{date}"
        if format == "csv":
            return Response(df.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
        buf = io.BytesIO()
        df.to_excel(buf, index=False, sheet_name="forecast")
        return Response(buf.getvalue(),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})

    @app.get("/api/v1/forecast/dates")
    def dates(limit: int = 30):
        with engine.connect() as conn:
            ts = pd.read_sql(select(db.price_forecast.c.timestamp), conn)["timestamp"]
        if ts.empty:
            return {"dates": []}
        days = sorted(set(pd.to_datetime(ts, utc=True).dt.tz_convert(TZ).dt.date), reverse=True)
        return {"dates": [d.isoformat() for d in days[:limit]]}

    def _baseline_errors(days: list) -> dict:
        """Eroarea medie a referintei simple (media pe 7 zile la aceeasi ora) pe zilele evaluate."""
        if not days:
            return {}
        start, end = day_index(min(days))[0], day_index(max(days))[-1]
        with engine.connect() as conn:
            fc = pd.read_sql(select(db.price_forecast.c.timestamp, db.price_forecast.c.mean7d_price).where(
                db.price_forecast.c.timestamp >= start.to_pydatetime(), db.price_forecast.c.timestamp <= end.to_pydatetime()), conn)
        if fc.empty:
            return {}
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        fc = fc.set_index("timestamp")["mean7d_price"].astype(float).sort_index()
        actual = db.read_actual_ro(engine, fc.index)
        err = (fc - actual.reindex(fc.index)).abs().dropna()
        by_day = err.groupby(err.index.tz_convert(TZ).date)
        return {d.isoformat(): _round(v.mean()) for d, v in by_day if len(v) >= 92}

    @app.get("/api/v1/accuracy")
    def accuracy(days: int = Query(30, ge=1, le=365)):
        acc = db.read_df(engine, db.forecast_accuracy, index=None)
        if acc.empty:
            return {"days": 0, "mean_mae": None, "items": []}
        acc["forecast_date"] = pd.to_datetime(acc["forecast_date"]).dt.date
        acc = acc[acc["forecast_date"] >= date.today() - timedelta(days=days)].sort_values("forecast_date")
        items = [{k: (v.isoformat() if isinstance(v, date) else (None if pd.isna(v) else v))
                  for k, v in r.items()} for r in acc.to_dict("records")]
        base = _baseline_errors(acc["forecast_date"].tolist())
        for it in items:
            it["mae_mean7d"] = base.get(it["forecast_date"])
        late = acc[acc["on_time"] == False] if "on_time" in acc else acc.iloc[0:0]   # noqa: E712
        ok = acc.drop(late.index)
        return {"days": len(items), "mean_mae": _round(acc["mae"].mean()) if len(acc) else None,
                "late_days": int(len(late)), "mean_mae_on_time": _round(ok["mae"].mean()) if len(ok) else None,
                "mean_coverage": _round(acc["coverage_90"].mean()) if len(acc) else None, "items": items}

    @app.get("/api/v1/region/prices")
    def region_prices(date: date):
        """Preturile oficiale ale Romaniei si ale vecinilor pentru o zi (dupa publicare), pe 15 minute."""
        idx = day_index(date)
        p = insights.region_prices(engine, idx[0], idx[-1]).reindex(idx)
        zones = [z for z in insights.ZONES if z in p and p[z].notna().any()]
        return {"date": date.isoformat(), "times": [_iso(ts)[11:16] for ts in idx],
                "zones": [{"code": z, "name": insights.ZONE_NAMES[z]} for z in zones],
                "prices": {z: [None if pd.isna(v) else round(float(v), 2) for v in p[z]] for z in zones},
                "daily_mean": {z: _round(p[z].mean()) for z in zones}}

    @app.get("/api/v1/region/daily")
    def region_daily(days: int = Query(90, ge=7, le=730)):
        """Pretul mediu zilnic al fiecarei zone, ultimele `days` zile."""
        end = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=2)
        p = insights.region_prices(engine, (end - pd.Timedelta(days=days + 2)).tz_convert("UTC"), end.tz_convert("UTC"))
        if p.empty:
            return {"dates": [], "zones": [], "means": {}}
        d = p.groupby(p.index.tz_convert(TZ).date)
        mean, n = d.mean(), d.count()
        mean = mean[n["RO"] >= 92] if "RO" in n else mean
        mean = mean.tail(days)
        zones = [z for z in insights.ZONES if z in mean]
        return {"dates": [x.isoformat() for x in mean.index],
                "zones": [{"code": z, "name": insights.ZONE_NAMES[z]} for z in zones],
                "means": {z: [None if pd.isna(v) else round(float(v), 1) for v in mean[z]] for z in zones}}

    def _mae(day) -> float | None:
        acc = db.read_df(engine, db.forecast_accuracy, index=None)
        if acc.empty:
            return None
        row = acc[pd.to_datetime(acc["forecast_date"]).dt.date == day]
        return None if row.empty else float(row["mae"].iloc[0])

    @app.get("/api/v1/insights")
    def day_insights(date: date):
        """Situatii neobisnuite ale zilei si posibile explicatii (dupa publicarea preturilor oficiale)."""
        return {"date": date.isoformat(), "events": insights.day_events(engine, date, forecast_mae=_mae(date))}

    @app.get("/api/v1/insights/recent")
    def recent_insights(days: int = Query(30, ge=1, le=120)):
        today = pd.Timestamp.now(tz=TZ).date()
        out = []
        for k in range(-1, days):
            d = today - timedelta(days=k)
            ev = insights.day_events(engine, d, forecast_mae=_mae(d))
            if ev:
                out.append({"date": d.isoformat(), "events": ev})
        return {"days": out}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = None  # `uvicorn pzu.api.main:create_app --factory`
