"""Valoarea economica a forecast-ului: simulam decizii (baterie, parc solar/eolian) luate pe baza
prognozei si le evaluam la preturile REALE. Valoarea = castig fata de strategii simple.

Toate activele sunt de referinta (ipotetice); preturile sunt cele reale din perioada testata.
Se modeleaza doar venitul PZU: fara dezechilibre, taxe, scheme de sprijin sau contracte bilaterale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

DT = 0.25  # ore pe interval


@dataclass(frozen=True)
class Battery:
    power_mw: float = 10.0
    energy_mwh: float = 20.0          # 2 ore
    round_trip_eff: float = 0.85
    max_cycles_per_day: float = 1.0
    cost_per_mwh: float = 4.0         # uzura/operare, EUR per MWh descarcat+incarcat


def dispatch(price: np.ndarray, b: Battery) -> tuple[np.ndarray, np.ndarray]:
    """Program optim incarcare/descarcare (MW) pentru preturile date; stare initiala goala.
    Variabile: [c_0..c_{n-1}, d_0..d_{n-1}] (MW)."""
    n = len(price)
    eta = np.sqrt(b.round_trip_eff)
    # minimizam -(venit) = sum(p*(c-d)*dt) + cost*(c+d)*dt
    obj = np.concatenate([price * DT + b.cost_per_mwh * DT, -price * DT + b.cost_per_mwh * DT])
    # SoC_k = sum_{j<k}(eta*c_j - d_j/eta)*dt  in [0, E]  (k=1..n)
    tri = np.tril(np.ones((n, n)))
    soc = np.hstack([tri * eta * DT, -tri * DT / eta])
    A_ub = np.vstack([soc, -soc, np.concatenate([np.zeros(n), np.full(n, DT)])[None, :]])
    b_ub = np.concatenate([np.full(n, b.energy_mwh), np.zeros(n), [b.max_cycles_per_day * b.energy_mwh]])
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, b.power_mw)] * (2 * n), method="highs")
    if not res.success:
        raise RuntimeError(f"LP esuat: {res.message}")
    return res.x[:n], res.x[n:]


def battery_revenue(c: np.ndarray, d: np.ndarray, actual: np.ndarray, b: Battery) -> float:
    return float(np.sum((d - c) * actual * DT) - b.cost_per_mwh * np.sum(c + d) * DT)


def simulate_battery(bt: pd.DataFrame, strategies: dict[str, str], b: Battery) -> pd.DataFrame:
    """bt: coloane y (pret real), date, + coloana de prognoza per strategie. Returneaza venit pe zi."""
    rows = []
    for day, g in bt.groupby("date"):
        actual = g["y"].to_numpy()
        row = {"date": day}
        for name, col in strategies.items():
            c, d = dispatch(g[col].to_numpy() if col != "y" else actual, b)
            row[name] = battery_revenue(c, d, actual, b)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def simulate_curtailment(bt: pd.DataFrame, gen_mw: pd.Series, strategies: dict[str, str],
                         threshold: float = 0.0) -> pd.DataFrame:
    """Parc fara sprijin: vinde in PZU daca pretul PROGNOZAT > prag, altfel reduce productia (curtail).
    'always_sell' = nu reduce niciodata. Venit evaluat la pretul real."""
    g = gen_mw.reindex(bt.index)
    rows = []
    for day, idx in bt.groupby("date").groups.items():
        sub, gen = bt.loc[idx], g.loc[idx].fillna(0).to_numpy()
        actual = sub["y"].to_numpy()
        row = {"date": day, "always_sell": float(np.sum(gen * actual * DT))}
        for name, col in strategies.items():
            sell = (actual if col == "y" else sub[col].to_numpy()) > threshold
            row[name] = float(np.sum(gen * actual * DT * sell))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def summarize(daily: pd.DataFrame, reference: str, perfect: str | None = None) -> pd.DataFrame:
    tot = daily.sum()
    out = pd.DataFrame({"total_eur": tot, "per_day_eur": daily.mean(),
                        "vs_reference_eur": tot - tot[reference]})
    if perfect:
        gap = tot[perfect] - tot[reference]
        out["pct_of_gap_captured"] = (tot - tot[reference]) / gap * 100 if gap else np.nan
    return out
