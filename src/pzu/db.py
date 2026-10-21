"""Stocare: SQLAlchemy Core, acelasi cod pentru SQLite (dev) si PostgreSQL (productie).
Timestamp-urile se stocheaza in UTC; la citire se returneaza tz-aware UTC."""
import numpy as np
import pandas as pd
from sqlalchemy import (
    JSON, Column, Date, DateTime, Float, LargeBinary, MetaData, String, Table, create_engine, delete,
    inspect, select, text, tuple_,
)
from sqlalchemy.engine import Engine

from pzu.config import get_settings

metadata = MetaData()
TS = DateTime(timezone=True)

prices_actual = Table(
    "prices_actual", metadata,
    Column("timestamp", TS, primary_key=True),
    Column("zone", String(2), primary_key=True),
    Column("price_eur_mwh", Float),
    Column("source", String(20)),
)

INPUT_COLS = [
    "solar_forecast_mw", "wind_forecast_mw", "load_forecast_mw",
    "capacity_hu_ro_mw", "capacity_ro_hu_mw", "capacity_bg_ro_mw", "capacity_ro_bg_mw",
    "temp_c", "wind_speed_ms", "radiation_wm2",
]
forecasts_input = Table(
    "forecasts_input", metadata,
    Column("timestamp", TS, primary_key=True),
    *[Column(c, Float) for c in INPUT_COLS],
)

price_forecast = Table(
    "price_forecast", metadata,
    Column("timestamp", TS, primary_key=True),
    Column("forecast_generated_at", TS),
    Column("predicted_price", Float),
    Column("ci_lower", Float),
    Column("ci_upper", Float),
    Column("mean7d_price", Float),   # referinta simpla: media pe 7 zile la aceeasi ora
    Column("blend_price", Float),    # mix 50/50 model + media pe 7 zile
    Column("model_version", String(20)),
    Column("features_used", JSON),
)

forecast_accuracy = Table(
    "forecast_accuracy", metadata,
    Column("forecast_date", Date, primary_key=True),
    Column("mae", Float), Column("rmse", Float), Column("smape", Float),
    Column("directional_hit_rate", Float), Column("coverage_90", Float),
    Column("model_version", String(20)),
)


GEN_COLS = ["gen_nuclear_mw", "gen_hydro_res_mw", "gen_hydro_ror_mw", "gen_gas_mw", "gen_coal_mw"]
generation_actual = Table(
    "generation_actual", metadata,
    Column("timestamp", TS, primary_key=True),
    *[Column(c, Float) for c in GEN_COLS],
)

# Documente de indisponibilitate (ENTSO-E A77/A80): se pastreaza momentul crearii, ca sa putem
# reconstrui ce se stia la ora prognozei (fara informatii aparute dupa).
outages = Table(
    "outages", metadata,
    Column("mrid", String(40), primary_key=True),
    Column("revision", String(10), primary_key=True),
    Column("resource_id", String(40), primary_key=True),
    Column("start", TS, primary_key=True),
    Column("zone", String(2)),
    Column("created", TS),
    Column("end", TS),
    Column("plant_type", String(60)),
    Column("business_type", String(40)),
    Column("doc_status", String(20)),
    Column("nominal_mw", Float),
    Column("available_mw", Float),
)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Migrare usoara: coloanele noi (nullable) se adauga in tabelele existente (ex. DB din release)."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have and not col.primary_key:
                    ddl = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))


def upsert_df(engine: Engine, table: Table, df: pd.DataFrame, keys: list[str]) -> int:
    """Insereaza randurile, inlocuindu-le pe cele cu aceeasi cheie (portabil intre SQLite/Postgres)."""
    if df.empty:
        return 0
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis("timestamp").reset_index()
    df = df[[c.name for c in table.columns if c.name in df.columns]]
    df = df.astype(object).where(df.notna(), None)
    rows = df.to_dict("records")
    for r in rows:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = v.to_pydatetime()
    with engine.begin() as conn:
        for i in range(0, len(rows), 2000):
            chunk = rows[i:i + 2000]
            if len(keys) == 1:
                cond = table.c[keys[0]].in_([r[keys[0]] for r in chunk])
            else:
                cond = tuple_(*[table.c[k] for k in keys]).in_([tuple(r[k] for k in keys) for r in chunk])
            conn.execute(delete(table).where(cond))
            conn.execute(table.insert(), chunk)
    return len(rows)


def read_df(engine: Engine, table: Table, index: str | None = "timestamp") -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(select(table), conn)
    if index and index in df.columns:
        df[index] = pd.to_datetime(df[index], utc=True)
        df = df.set_index(index).sort_index()
    return df


def read_prices(engine: Engine) -> pd.DataFrame:
    """Preturi in format lat: index=timestamp UTC, coloane=zone."""
    df = read_df(engine, prices_actual, index=None)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.pivot(index="timestamp", columns="zone", values="price_eur_mwh").sort_index()


class SyntheticDataError(RuntimeError):
    pass


def ensure_real_data(engine: Engine) -> None:
    """Protectie: modelele de productie se antreneaza doar pe date reale."""
    with engine.connect() as conn:
        n = conn.execute(
            select(prices_actual.c.timestamp).where(prices_actual.c.source == "SYNTHETIC").limit(1)
        ).first()
    if n is not None:
        raise SyntheticDataError(
            "Baza de date contine date SYNTHETIC. Antrenarea pe date inventate e blocata; "
            "goleste baza si ruleaza backfill.py pe date reale (ENTSOE_TOKEN)."
        )


# --- registru de modele (modelul antrenat se pastreaza in DB, accesibil din GitHub Actions) ---
model_store = Table(
    "model_store", metadata,
    Column("name", String(20), primary_key=True),   # 'latest'
    Column("version", String(40)),
    Column("trained_at", TS),
    Column("payload", LargeBinary),
)


def save_model_bytes(engine: Engine, version: str, payload: bytes, name: str = "latest") -> None:
    from datetime import datetime, timezone
    with engine.begin() as conn:
        conn.execute(delete(model_store).where(model_store.c.name == name))
        conn.execute(model_store.insert(), [{"name": name, "version": version,
                                             "trained_at": datetime.now(timezone.utc), "payload": payload}])


def load_model_bytes(engine: Engine, name: str = "latest") -> tuple[str, bytes] | None:
    with engine.connect() as conn:
        row = conn.execute(select(model_store.c.version, model_store.c.payload)
                           .where(model_store.c.name == name)).first()
    return (row[0], bytes(row[1])) if row else None


def count_forecast_rows(engine: Engine, start, end) -> int:
    from sqlalchemy import func
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(price_forecast).where(
            price_forecast.c.timestamp >= start.to_pydatetime(),
            price_forecast.c.timestamp <= end.to_pydatetime())).scalar_one()


def read_forecast(engine: Engine, idx: pd.DatetimeIndex) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(select(price_forecast).where(
            price_forecast.c.timestamp >= idx[0].to_pydatetime(),
            price_forecast.c.timestamp <= idx[-1].to_pydatetime()), conn)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def read_actual_ro(engine: Engine, idx: pd.DatetimeIndex) -> pd.Series:
    """Preturile oficiale RO pentru intervalele date (NaN unde nu sunt publicate)."""
    with engine.connect() as conn:
        df = pd.read_sql(select(prices_actual.c.timestamp, prices_actual.c.price_eur_mwh).where(
            prices_actual.c.zone == "RO",
            prices_actual.c.timestamp >= idx[0].to_pydatetime(),
            prices_actual.c.timestamp <= idx[-1].to_pydatetime()), conn)
    if df.empty:
        return pd.Series(np.nan, index=idx)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["price_eur_mwh"].reindex(idx)
