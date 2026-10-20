# pzu-forecast: Forecast PZU România D+1

Prognoză a prețului PZU (zona RO) pe 96 de intervale de 15 min, cu interval de încredere 90%.

## Stadiu
- [x] **A. Fundație**: config, schemă DB (SQLite/Postgres), clienți ENTSO-E și Open-Meteo
- [x] **B. Date**: `scripts/backfill.py`, generator de date sintetice, raport de calitate
- [x] **C. Features + baseline**: feature-uri anti-leakage, baseline-uri naive
- [x] **D. LightGBM + walk-forward + conformal prediction** (`scripts/run_backtest.py`, `scripts/train.py`)
- [x] **E. Automatizare**: `python -m pzu.pipeline daily|evaluate|retrain` + workflow-uri GitHub Actions
- [x] **F. API FastAPI + dashboard**: `uvicorn pzu.api.main:create_app --factory`, pagina `/`
- [ ] G. Backtest complet și raport

## Decizii de design
- **Fără prețuri D+1 ale vecinilor ca input**: se publică după gate closure (12:00 CET), deci nu
  există la ora prognozei. Vecinii intră ca lag-uri (24h, 7 zile). Verificat prin `verify_no_leakage`.
- Timestamp-urile se stochează în **UTC**; ziua de livrare se definește în `Europe/Bucharest`
  (92/96/100 intervale la schimbarea orei).
- Sursa de prețuri v1: ENTSO-E (A44). OPCOM e opțional (necesită cont plătit).
- Tabelul `prices_actual` are coloana `zone` (RO + vecini) față de spec, care avea doar RO.

- **Antrenare doar pe date reale**: `train.py` și `run_backtest.py` refuză să pornească dacă DB conține
  rânduri `SYNTHETIC`. Datele sintetice servesc doar testelor de cod (`--smoke-test`).
- **Istoric 24 de luni** (minim 12 ca să acopere toate sezoanele); fereastra de antrenare (60/180/365 zile)
  se alege din backtest, nu se presupune.
- **Intervale de încredere**: pentru 90% se folosește cuantila 90% a |erorii| (spec-ul indica 95%, ceea ce
  ar da ~95% acoperire), cu calibrare „onestă" (doar erori din zilele anterioare) și cuantile pe oră.

- **Capacități RO↔HU**: ENTSO-E nu publică NTC pentru această graniță (Core, flow-based), deci
  `capacity_hu_ro_mw`/`capacity_ro_hu_mw` nu sunt folosite. Rămân RO↔BG.
- **Rezoluție mixtă la tranziția orar→15 min** (sept. 2025): `to_15min` o tratează explicit.

## Automatizare (GitHub Actions): fără costuri, fără server
Workflow-urile din `.github/workflows/` rulează zilnic/săptămânal (ore în UTC, ca să acopere 06:00 ora României
vara și iarna). Job-urile sunt **idempotente**: reîncearcă pe parcursul dimineții dacă datele D+1 nu sunt încă
publicate, iar ultima încercare (`--final`) dă eroare vizibilă.

**Persistență gratuită:** baza de date (SQLite, ~9 MB comprimat) se păstrează ca fișier atașat release-ului
`data-latest` (`scripts/ci_db.sh`), suprascris la fiecare rulare doar dacă s-a schimbat. Modelul antrenat e în DB
(tabelul `model_store`). Fiecare rulare atașează și `dashboard.html`, o pagină autonomă (date incluse) care se
deschide direct în browser.

**Configurare (o singură dată):**
1. Unifică branch-ul în `main` (workflow-urile programate rulează doar de pe branch-ul implicit).
2. Settings → Secrets and variables → Actions → secret `ENTSOE_TOKEN`.
3. Actions → **Bootstrap** → Run workflow (~1 oră: istoric 24 luni + primul model).

Apoi totul rulează singur. Dashboard-ul: Releases → `PZU data` → `dashboard.html`.
Alternativ, pe un server propriu: Postgres (`DATABASE_URL`) + `docker compose up`.

## API
`GET /api/v1/forecast/pzu?date=YYYY-MM-DD`, `/summary`, `/export?format=csv|xlsx`, `/api/v1/forecast/dates`,
`/api/v1/accuracy?days=30`, `/health`. Dashboard la `/`. **Fără autentificare** în v1: nu expune API-ul public.

## Raport de valoare
`python scripts/value_report.py` simulează decizii (baterie 10 MW 2h/4h, parc solar/eolian 100 MW) luate cu prognoza,
evaluate la prețurile reale, și le compară cu strategii simple („prețul de ieri”, media pe 7 zile). Activele sunt
ipotetice; ENTSO-E nu publică producția pe centrale individuale pentru România, deci parcurile folosesc profilul
național real. Rezultatul actual e în `reports/value_report.html`.

## Rulare
```bash
pip install -e ".[dev]"
cp .env.example .env              # completează ENTSOE_TOKEN
python scripts/backfill.py --synthetic --days 120   # fără token, date sintetice
python scripts/backfill.py --months 24              # date reale
python scripts/run_backtest.py --windows 60,180,365 # backtest pe date reale
python scripts/train.py                             # model final -> models/
pytest
```
Datele sintetice servesc doar la dezvoltare: metricile pe ele nu spun nimic despre piața reală.
Clientul ENTSO-E nu a fost testat pe API-ul real (lipsește tokenul), doar pe răspunsuri simulate.
