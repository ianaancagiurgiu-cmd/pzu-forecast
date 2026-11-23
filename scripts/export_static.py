"""Dashboard autonom (un singur fisier HTML, cu datele incluse) - se deschide direct in browser.
Util cand nu exista server: GitHub Actions il ataseaza la release.

  python scripts/export_static.py [--out site/dashboard.html] [--days 30]
"""
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from pzu import db
from pzu.api.main import STATIC, create_app


def build_data(client: TestClient, days: int) -> dict:
    dates = client.get(f"/api/v1/forecast/dates?limit={days}").json()["dates"]
    data = {"dates": dates, "forecast": {}, "summary": {}, "versions": {},
            "accuracy": client.get("/api/v1/accuracy?days=365").json()}
    data["region_daily"] = client.get("/api/v1/region/daily?days=120").json()
    data["insights_recent"] = client.get(f"/api/v1/insights/recent?days={max(days, 60)}").json()
    region_days = sorted(set(dates) | set(data["region_daily"]["dates"][-60:]))
    data["region"] = {d: client.get(f"/api/v1/region/prices?date={d}").json() for d in region_days}
    data["insights"] = {x["date"]: x for x in data["insights_recent"]["days"]}
    data["battery"] = {}
    for d in dates:
        data["forecast"][d] = client.get(f"/api/v1/forecast/pzu?date={d}").json()
        data["battery"][d] = client.get(f"/api/v1/battery?date={d}").json()
        data["summary"][d] = client.get(f"/api/v1/forecast/pzu/summary?date={d}").json()
        data["versions"][d] = client.get(f"/api/v1/forecast/pzu/versions?date={d}").json()
    return data


def render(data: dict) -> str:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("<script>\n", f"<script>\nwindow.PZU_DATA = {blob};\n", 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/dashboard.html")
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    client = TestClient(create_app(db.get_engine()))
    data = build_data(client, a.days)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"{out} ({len(data['dates'])} zile)")


if __name__ == "__main__":
    main()
