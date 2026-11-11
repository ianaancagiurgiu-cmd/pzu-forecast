"""Model LightGBM + pachet salvat (model, feature-uri, calibrare conformala, versiune)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import io

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from pzu import features as F
from pzu.conformal import ConformalInterval
from pzu.timegrid import day_index

PARAMS = dict(
    objective="regression", metric="mae", n_estimators=1000, learning_rate=0.05,
    num_leaves=31, max_depth=7, min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    verbosity=-1, n_jobs=-1,
)


def make_model(**overrides) -> lgb.LGBMRegressor:
    # subsample_freq=1 e necesar ca `subsample=0.8` sa aiba efect in LightGBM
    return lgb.LGBMRegressor(**{**PARAMS, **overrides})


def fit(X: pd.DataFrame, y: pd.Series, cols: list[str] | None = None, sample_weight=None,
        **overrides) -> lgb.LGBMRegressor:
    return make_model(**overrides).fit(X[cols or F.PRODUCTION_FEATURES], y, sample_weight=sample_weight)


@dataclass
class ModelBundle:
    model: lgb.LGBMRegressor
    version: str
    window_days: int
    conformal: ConformalInterval
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    calibrated_at: str | None = None   # cand au fost calculate intervalele de incredere (backtest)
    fallback_model: lgb.LGBMRegressor | None = None      # model simplu, folosit daca lipsesc consumurile vecinilor
    fallback_features: list[str] | None = None
    features: list[str] = field(default_factory=lambda: list(F.PRODUCTION_FEATURES))
    nowx_model: lgb.LGBMRegressor | None = None          # fara vreme: folosit daca prognoza de vreme lipseste
    nowx_features: list[str] | None = None

    def save(self, directory: str | Path = "models") -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"pzu_{self.version}.joblib"
        joblib.dump(self, out)
        joblib.dump(self, path / "latest.joblib")
        return out

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(payload: bytes) -> "ModelBundle":
        # payload-ul vine din propria noastra DB (de incredere); joblib/pickle nu e sigur pe date straine
        return joblib.load(io.BytesIO(payload))

    @staticmethod
    def load(path: str | Path = "models/latest.joblib") -> "ModelBundle":
        return joblib.load(path)


def pick_variant(bundle, X: pd.DataFrame, target_day=None):
    """Cel mai complet model pentru care exista toate intrarile: principal (cu vreme) -> fara vreme -> de rezerva."""
    needs_nb = [c for c in bundle.features if c in F.NEIGHBOR_LOADS]
    nb_ok = not X[needs_nb].isna().any().any() if needs_nb else True
    needs_wx = [c for c in bundle.features if c.startswith("wx_")]
    wx_ok = all(c in X for c in needs_wx) and not (X[needs_wx].isna().any().any() if needs_wx else False)
    if nb_ok and wx_ok:
        return bundle.model, bundle.features, "main"
    if nb_ok and getattr(bundle, "nowx_model", None) is not None:   # lipseste prognoza de vreme
        return bundle.nowx_model, bundle.nowx_features, "noweather"
    if getattr(bundle, "fallback_model", None) is not None:   # lipsesc consumurile vecinilor: model fara ele
        return bundle.fallback_model, bundle.fallback_features, "fallback"
    raise ValueError(f"Lipsesc prognozele de consum ale vecinilor pentru {pd.Timestamp(target_day).date() if target_day is not None else '?'}")


def forecast_day(bundle: ModelBundle, inputs: pd.DataFrame, prices: pd.DataFrame, target_day,
                 nload: pd.DataFrame | None = None, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast pentru ziua `target_day` (D+1). `prices` poate contine doar date pana la D;
    verificam ca feature-urile nu depind de nimic de la target_day incolo."""
    F.verify_no_leakage(inputs, prices, target_day, nload=nload)
    idx = day_index(target_day)
    X = F.build_features(inputs, prices, idx, nload=nload)
    if weather is not None:
        from pzu.weather import weather_features
        X = X.join(weather_features(weather, idx))
    if X["load_forecast_mw"].isna().any():
        raise ValueError(f"Lipseste prognoza de consum pentru {pd.Timestamp(target_day).date()}")
    model, cols, variant = pick_variant(bundle, X, target_day)
    pred = pd.Series(model.predict(X[cols]), index=idx)
    lo, hi = bundle.conformal.interval(pred)
    mean7d = pd.Series(X["price_rolling_mean_7d"].to_numpy(), index=idx)
    blend = (0.5 * pred + 0.5 * mean7d).where(mean7d.notna(), pred)
    out = pd.DataFrame({"predicted_price": pred, "ci_lower": lo, "ci_upper": hi,
                        "mean7d_price": mean7d, "blend_price": blend})
    out["model_version"] = bundle.version
    out["variant"] = variant
    return out
