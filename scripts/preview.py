"""Pagina HTML de previzualizare a datelor din DB (sau sintetice, daca DB e goala).

  python scripts/preview.py [--days 7] [--out reports/preview.html]
Fara dependente externe: graficele sunt SVG inline.
"""
import argparse
import html
from pathlib import Path

import pandas as pd

from pzu import db
from pzu.baseline import evaluate_baselines
from pzu.config import TZ
from pzu.features import build_training_set

COLORS = {"RO": "#d62828", "HU": "#4c78a8", "BG": "#59a14f", "RS": "#b07aa1", "SK": "#9c9c9c"}
W, H, PAD = 980, 280, 44


def line_chart(df: pd.DataFrame, title: str, unit: str, colors: dict, width_first: str | None = None) -> str:
    x = list(range(len(df)))
    lo, hi = float(df.min().min()), float(df.max().max())
    span = (hi - lo) or 1
    sx = lambda i: PAD + i * (W - 2 * PAD) / max(len(df) - 1, 1)
    sy = lambda v: H - PAD - (v - lo) / span * (H - 2 * PAD)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for k in range(5):
        v = lo + span * k / 4
        parts.append(f'<line x1="{PAD}" x2="{W-PAD}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
                     f'<text x="{PAD-6}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    if isinstance(df.index, pd.DatetimeIndex):
        ticks = [(i, f"{ts:%d.%m}") for i, ts in enumerate(df.index.tz_convert(TZ))
                 if ts.hour == 0 and ts.minute == 0]
    else:  # axa orelor zilei
        ticks = [(i, f"{h:02d}h") for i, h in enumerate(df.index) if h % 3 == 0]
    for i, label in ticks:
        if True:
            parts.append(f'<line x1="{sx(i):.1f}" x2="{sx(i):.1f}" y1="{PAD}" y2="{H-PAD}" class="grid"/>'
                         f'<text x="{sx(i):.1f}" y="{H-PAD+16}" text-anchor="middle">{label}</text>')
    for col in df.columns:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in zip(x, df[col]) if pd.notna(v))
        w = 2 if col == width_first else 1.2
        op = 1 if col == width_first else 0.7
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[col]}" '
                     f'stroke-width="{w}" opacity="{op}"/>')
    parts.append(f'<text x="{PAD}" y="16" class="ttl">{html.escape(title)} ({unit})</text></svg>')
    legend = "".join(f'<span><i style="background:{colors[c]}"></i>{c}</span>' for c in df.columns)
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="reports/preview.html")
    a = ap.parse_args()

    eng = db.get_engine()
    db.init_db(eng)
    inputs, prices = db.read_df(eng, db.forecasts_input), db.read_prices(eng)
    note = "Date din baza de date."
    if prices.empty:
        from pzu.synthetic import generate
        inputs, prices = generate("2026-01-01", 120)
        note = "Date SINTETICE (baza de date e goala) - nu reprezinta piata reala."

    n = a.days * 96
    p, i = prices.iloc[-n:], inputs.reindex(prices.index).iloc[-n:]
    i = i.assign(residual_load=i.load_forecast_mw - i.solar_forecast_mw - i.wind_forecast_mw)
    X, y = build_training_set(inputs, prices)
    base = evaluate_baselines(X, y).round(2)
    rows = "".join(f"<tr><td>{html.escape(k)}</td>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                   for k, r in base.iterrows())
    hourly = (y.groupby(y.index.tz_convert(TZ).hour).mean()).to_frame("RO")
    ci = {"RO": COLORS["RO"], "load_forecast_mw": "#333", "solar_forecast_mw": "#f2a900",
          "wind_forecast_mw": "#2a9d8f", "residual_load": "#6a4c93"}

    page = f"""<!doctype html><html lang="ro"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preview date PZU</title><style>
body{{font:15px system-ui,sans-serif;margin:0;padding:16px;background:#fff;color:#1d1d1f}}
@media(prefers-color-scheme:dark){{body{{background:#161618;color:#eee}}.grid{{stroke:#333}}svg text{{fill:#aaa}}}}
main{{max-width:1000px;margin:auto}} svg{{width:100%;height:auto}} svg text{{font-size:11px;fill:#666}}
.ttl{{font-weight:600;fill:currentColor!important;font-size:13px!important}}
.grid{{stroke:#ddd;stroke-width:1}} .note{{background:#fff3cd;color:#664d03;padding:8px 12px;border-radius:6px}}
.legend span{{margin-right:14px;font-size:13px}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}}
table{{border-collapse:collapse}} td,th{{padding:4px 12px;border-bottom:1px solid #8884;text-align:right}} td:first-child,th:first-child{{text-align:left}}
h2{{margin-top:28px}}</style><main>
<h1>Preview date - PZU Romania</h1><p class="note">{html.escape(note)}</p>
<h2>Preturi PZU: RO vs vecini, ultimele {a.days} zile</h2>
{line_chart(p, "Pret day-ahead", "EUR/MWh", COLORS, "RO")}
<h2>Pret RO si consum/productie prognozate</h2>
{line_chart(i[["load_forecast_mw","solar_forecast_mw","wind_forecast_mw","residual_load"]], "Load, solar, eolian, residual", "MW", ci)}
<h2>Pret mediu pe ora a zilei (ora locala)</h2>
{line_chart(hourly, "Profil orar mediu RO", "EUR/MWh", COLORS, "RO")}
<h2>Baseline-uri naive (tinta pentru LightGBM)</h2>
<table><tr><th>metoda</th><th>MAE</th><th>RMSE</th><th>sMAPE %</th><th>hit rate</th></tr>{rows}</table>
</main></html>"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
