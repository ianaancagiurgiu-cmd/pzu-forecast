"""Raport de valoare: cat ar fi castigat decizii luate cu forecast-ul vs. strategii simple, la preturi reale.

  python scripts/value_report.py [--days 90] [--out reports/value_report.html]

Active de referinta (ipotetice): baterie 10 MW (2h si 4h) si parc solar / eolian de 100 MW cu profilul
productiei nationale reale (ENTSO-E nu publica productia pe centrale individuale pentru Romania).
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pzu import backtest, db, features as F, value as V
from pzu.config import get_settings
from pzu.ingest import entsoe

CACHE_BT = Path("reports/backtest_predictions.csv")
STRATS = {"Prognoza modelului": "pred", "Prețul de ieri": "naive_lag24h", "Media pe 7 zile": "naive_mean7d",
          "Mix 50/50 model + media 7 zile": "blend", "Prevedere perfectă (plafon)": "y"}
REF = "Prețul de ieri"
PERFECT = "Prevedere perfectă (plafon)"
COLORS = {"Prognoza modelului": "#c1121f", "Prețul de ieri": "#4c78a8", "Media pe 7 zile": "#59a14f", "Mix 50/50 model + media 7 zile": "#f2a900",
          PERFECT: "#888888", "Fără prognoză (vinde mereu)": "#b07aa1"}


def load_backtest(days: int, window: int) -> pd.DataFrame:
    if CACHE_BT.exists():
        bt = pd.read_csv(CACHE_BT, index_col=0, parse_dates=True)
        bt["date"] = pd.to_datetime(bt["date"]).dt.date
        return bt
    eng = db.get_engine()
    db.ensure_real_data(eng)
    X, y = F.build_training_set(db.read_df(eng, db.forecasts_input), db.read_prices(eng),
                                nload=db.read_neighbor_load(eng))
    bt = backtest.walk_forward(X, y, window, days)
    CACHE_BT.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(CACHE_BT)
    return bt


def national_profile(bt: pd.DataFrame, psr: str) -> pd.Series:
    """Productia nationala reala (MW), normalizata la un parc de 100 MW (percentila 99,5 = capacitate)."""
    cache = Path(f"reports/_gen_{psr}.csv")
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        from entsoe import EntsoePandasClient
        c = EntsoePandasClient(api_key=get_settings().entsoe_token)
        start = bt.index.min().tz_convert("Europe/Bucharest").normalize()
        end = min(bt.index.max().tz_convert("Europe/Bucharest").normalize() + pd.Timedelta(days=1),
                  pd.Timestamp.now(tz="Europe/Bucharest").normalize())
        parts = []
        for s0 in pd.date_range(start, end, freq="30D", tz="Europe/Bucharest", inclusive="left"):
            try:
                g = c.query_generation("RO", start=s0, end=min(s0 + pd.Timedelta(days=30), end), psr_type=psr)
            except Exception as exc:  # NoMatchingDataError pentru segmente fara date
                print(f"productie {psr} {s0.date()}: {type(exc).__name__}")
                continue
            parts.append(g.iloc[:, 0] if isinstance(g, pd.DataFrame) else g)
        s = pd.concat(parts)
        s = s[~s.index.duplicated()].tz_convert("UTC").resample("15min").mean().interpolate(limit=4)
        s.to_csv(cache)
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    return (s / np.nanpercentile(s, 99.5) * 100).clip(0, 100)


def svg_cumulative(daily: pd.DataFrame, title: str) -> str:
    cum = daily.cumsum()
    W, H, P = 900, 280, 70
    lo, hi = float(cum.min().min()), float(cum.max().max())
    span = (hi - lo) or 1
    sx = lambda i: P + i * (W - 2 * P) / max(len(cum) - 1, 1)
    sy = lambda v: H - 30 - (v - lo) / span * (H - 30 - 24)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for k in range(5):
        v = lo + span * k / 4
        g.append(f'<line class="grid" x1="{P}" x2="{W-P+20}" y1="{sy(v):.0f}" y2="{sy(v):.0f}"/>'
                 f'<text x="{P-6}" y="{sy(v)+4:.0f}" text-anchor="end">{v/1000:,.0f}k</text>')
    for col in cum.columns:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(cum[col]))
        g.append(f'<polyline fill="none" stroke="{COLORS.get(col, "#000")}" stroke-width="{2.4 if col == "Prognoza modelului" else 1.4}" points="{pts}"/>')
    g.append("</svg>")
    leg = "".join(f'<span><i style="background:{COLORS.get(c)}"></i>{c}</span>' for c in cum.columns)
    return f'<div class="legend">{leg}</div>' + "".join(g)


def table(df: pd.DataFrame, cols: dict[str, str]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ["Strategie", *cols.values()])
    rows = ""
    for name, r in df.iterrows():
        cells = "".join(f"<td>{r[c]:,.0f}</td>" if not c.startswith("pct") else
                        f"<td>{'' if pd.isna(r[c]) else f'{r[c]:.1f}%'}</td>" for c in cols)
        rows += f"<tr><td>{html.escape(str(name))}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{rows}</table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--out", default="reports/value_report.html")
    a = ap.parse_args()

    bt = load_backtest(a.days, a.window).dropna(subset=["y", "pred", "naive_lag24h", "naive_mean7d"])
    bt = bt.assign(blend=0.5 * bt["pred"] + 0.5 * bt["naive_mean7d"])   # ponderi fixate dinainte, fara ajustare pe test
    full = bt.groupby("date").size()
    bt = bt[bt["date"].isin(full[full >= 92].index)]
    n_days = bt["date"].nunique()
    results, sections = {}, []

    # ---- baterii
    for hours in (2, 4):
        b = V.Battery(power_mw=10, energy_mwh=10 * hours)
        daily = V.simulate_battery(bt, STRATS, b)
        s = V.summarize(daily, REF, PERFECT)
        s["pct_of_perfect"] = s["total_eur"] / s.loc[PERFECT, "total_eur"] * 100
        results[f"baterie_{hours}h"] = s.round(1).to_dict("index")
        sections.append(f"<h2>Baterie 10 MW / {10*hours} MWh ({hours} ore): venit pe {n_days} zile</h2>"
                        + svg_cumulative(daily, f"Venit cumulat baterie {hours}h (EUR)")
                        + table(s, {"total_eur": "Total (EUR)", "per_day_eur": "EUR/zi",
                                    "vs_reference_eur": "Față de „prețul de ieri”", "pct_of_perfect": "% din plafon"}))

    # ---- parc solar si eolian (profil national real), vinde in PZU doar daca pretul prognozat > 0
    neg = int((bt["y"] < 0).sum())
    for label, psr in (("solar", "B16"), ("eolian", "B19")):
        gen = national_profile(bt, psr)
        daily = V.simulate_curtailment(bt, gen, {k: v for k, v in STRATS.items()})
        daily = daily.rename(columns={"always_sell": "Fără prognoză (vinde mereu)"})
        order = ["Fără prognoză (vinde mereu)", *STRATS]
        daily = daily[order]
        s = V.summarize(daily, "Fără prognoză (vinde mereu)", PERFECT)
        results[f"parc_{label}"] = s.round(1).to_dict("index")
        sections.append(f"<h2>Parc {label} 100 MW (profil național real): reducere la prețuri negative</h2>"
                        f"<p class=sub>Regula: vinde în PZU doar dacă prețul <em>prognozat</em> e peste 0; altfel reduce producția. "
                        f"În perioada testată au fost <b>{neg}</b> intervale cu preț real negativ din {len(bt)}.</p>"
                        + "<p class=sub>Grafic: avantajul cumulat față de „vinde mereu” (EUR). Diferențele sunt mici față de venitul total.</p>"
                        + svg_cumulative(daily.sub(daily["Fără prognoză (vinde mereu)"], axis=0).drop(columns=["Fără prognoză (vinde mereu)"]),
                                         f"Avantaj cumulat parc {label} (EUR)")
                        + table(s, {"total_eur": "Total (EUR)", "vs_reference_eur": "Față de „vinde mereu” (EUR)"}))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/value_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    page = f"""<!doctype html><html lang="ro"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raport de valoare forecast PZU</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mute:#6b6b70;--line:#e3e3e6;--card:#f7f7f8}}
@media(prefers-color-scheme:dark){{:root{{--bg:#161618;--fg:#eee;--mute:#9a9aa0;--line:#2e2e33;--card:#202024}}}}
body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}main{{max-width:960px;margin:auto;padding:16px}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:30px}}.sub{{color:var(--mute);font-size:13px}}
svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:var(--mute)}}.grid{{stroke:var(--line)}}
.legend span{{margin-right:14px;font-size:13px}}.legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}}
table{{border-collapse:collapse;width:100%;margin-top:8px}}td,th{{padding:5px 10px;text-align:right;border-bottom:1px solid var(--line)}}td:first-child,th:first-child{{text-align:left}}
.warn{{background:#fff3cd;color:#664d03;padding:10px 14px;border-radius:6px;font-size:13px}}</style><main>
<h1>Cât valorează forecast-ul? Simulare pe prețuri reale</h1>
<p class="sub">{n_days} zile de backtest ({bt.index.min().date()} – {bt.index.max().date()}). Decizii luate cu prognoza de dimineață, evaluate la prețul oficial.</p>
<p class="warn"><b>Active ipotetice, nu reale.</b> Baterie 10 MW, randament 85%, 1 ciclu/zi, 4 EUR/MWh uzură. Parcurile folosesc profilul producției naționale
(ENTSO-E nu publică producția pe centrale individuale pentru România). Se modelează doar venitul din PZU: fără dezechilibre,
taxe, scheme de sprijin sau contracte. Modelul a învățat pe vremea reală, nu pe prognoze meteo, deci în producție poate fi puțin mai slab.
Rezultatele sunt pe o perioadă scurtă (vară–toamnă) și nu garantează profit viitor.</p>
{''.join(sections)}
</main></html>"""
    Path(a.out).write_text(page, encoding="utf-8")
    print(a.out)
    for k, v in results.items():
        print(k, {s: f"{r['total_eur']:,.0f}" for s, r in v.items()})


if __name__ == "__main__":
    main()
