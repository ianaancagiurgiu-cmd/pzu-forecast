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


def forecast_day(bundle: ModelBundle, inputs: pd.DataFrame, prices: pd.DataFrame, target_day,
                 nload: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast pentru ziua `target_day` (D+1). `prices` poate contine doar date pana la D;
    verificam ca feature-urile nu depind de nimic de la target_day incolo."""
    F.verify_no_leakage(inputs, prices, target_day, nload=nload)
    idx = day_index(target_day)
    X = F.build_features(inputs, prices, idx, nload=nload)
    if X["load_forecast_mw"].isna().any():
        raise ValueError(f"Lipseste prognoza de consum pentru {pd.Timestamp(target_day).date()}")
    needs_nb = [c for c in F.NEIGHBOR_LOADS if c in bundle.features]
    use_main = not X[needs_nb].isna().any().any() if needs_nb else True
    if use_main:
        model, cols, variant = bundle.model, bundle.features, "main"
    elif getattr(bundle, "fallback_model", None) is not None:   # lipsesc consumurile vecinilor: model fara ele
        model, cols, variant = bundle.fallback_model, bundle.fallback_features, "fallback"
    else:
        raise ValueError(f"Lipsesc prognozele de consum ale vecinilor pentru {pd.Timestamp(target_day).date()}")
    pred = pd.Series(model.predict(X[cols]), index=idx)
    lo, hi = bundle.conformal.interval(pred)
    mean7d = pd.Series(X["price_rolling_mean_7d"].to_numpy(), index=idx)
    blend = (0.5 * pred + 0.5 * mean7d).where(mean7d.notna(), pred)
    out = pd.DataFrame({"predicted_price": pred, "ci_lower": lo, "ci_upper": hi,
                        "mean7d_price": mean7d, "blend_price": blend})
    out["model_version"] = bundle.version
    out["variant"] = variant
    return out
