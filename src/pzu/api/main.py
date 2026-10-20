"""API REST + dashboard. `create_app(engine)` permite teste cu DB temporara."""
from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.engine import Engine

from pzu import db
from pzu.config import TZ
from pzu.timegrid import day_index

STATIC = Path(__file__).parent / "static"
LOW_PRICE, HIGH_PRICE = 30.0, 150.0   # praguri pentru indicatori (EUR/MWh)


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ).isoformat()


def _round(x) -> float:
    return round(float(x), 2)


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
        return {
            "date": date.isoformat(),
            "generated_at": _iso(pd.to_datetime(fc["forecast_generated_at"], utc=True).max()),
            "model_version": fc["model_version"].iloc[0],
            "intervals": [
                {"start": _iso(ts), "end": _iso(ts + step),
                 "predicted_price": _round(r.predicted_price),
                 "ci_90_lower": _round(r.ci_lower), "ci_90_upper": _round(r.ci_upper)}
                for ts, r in fc.iterrows()
            ],
        }

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

    @app.get("/api/v1/accuracy")
    def accuracy(days: int = Query(30, ge=1, le=365)):
        acc = db.read_df(engine, db.forecast_accuracy, index=None)
        if acc.empty:
            return {"days": 0, "mean_mae": None, "items": []}
        acc["forecast_date"] = pd.to_datetime(acc["forecast_date"]).dt.date
        acc = acc[acc["forecast_date"] >= date.today() - timedelta(days=days)].sort_values("forecast_date")
        items = [{k: (v.isoformat() if isinstance(v, date) else (None if pd.isna(v) else v))
                  for k, v in r.items()} for r in acc.to_dict("records")]
        return {"days": len(items), "mean_mae": _round(acc["mae"].mean()) if len(acc) else None,
                "items": items}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = None  # `uvicorn pzu.api.main:create_app --factory`
