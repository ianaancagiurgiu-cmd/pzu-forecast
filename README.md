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
Fiecare interval include și `mean_7d_price` (media pe 7 zile la aceeași oră) și `blend_price` (mix 50/50 cu modelul),
semnale simple care, în simulările pe baterie, au avut valoare comparabilă cu modelul.
`GET /api/v1/forecast/pzu?date=YYYY-MM-DD`, `/summary`, `/export?format=csv|xlsx`, `/api/v1/forecast/dates`,
`/api/v1/accuracy?days=30`, `/health`. Dashboard la `/`. **Fără autentificare** în v1: nu expune API-ul public.

## Model „de dimineață” (v2): ce folosește și de ce
Prognozele ENTSO-E de producție solară și eoliană pentru D+1 **nu sunt publicate nici după gate closure** (verificat
live pe 21 oct 2026, ora 13:03 a României), deci nu pot fi intrări pentru o prognoză de dimineață. Prima versiune le
folosea și rezultatele ei erau puțin optimiste. Modelul de producție folosește acum doar ce există sigur la 06:00:
prognoza de consum D+1, calendarul și prețurile din zilele trecute (RO și vecini, lag-uri). Vremea și capacitățile de
interconexiune nu au îmbunătățit rezultatele și au fost scoase (`PRODUCTION_FEATURES`), deci pipeline-ul nu mai
depinde de Open-Meteo. Pe 23 iul – 20 oct 2026: eroare medie ~29,4 EUR/MWh (față de 27,7 cu date indisponibile,
33,9 pentru media pe 7 zile și 32,9 pentru prețul de ieri). Detalii: `reports/experiments_morning.json`.
Atenție: regula „reduc producția când prețul prognozat e negativ” bazată pe model a pierdut bani la parcuri
în simulare; nu o folosi.

## Limite superioare (`scripts/production_oracle.py`, 668 de zile): ce informație ar ajuta, în principiu?
Testăm modelul de dimineață (M) cu informație pe care **nu o avem dimineața**, ca să aflăm cât ar putea câștiga:

| Intrare suplimentară | MAE | MAE iarna | Baterie (tot) | cu mix 50/50 |
|---|---|---|---|---|
| Niciuna (M, modelul de producție) | 24,9 | 26,9 | 82,4% | 86,3% |
| + prognoza operatorilor solar/eolian D+1 (A) | 24,0 | 24,7 | 84,1% | 87,3% |
| + producția solară/eoliană REALĂ a zilei (O) | 24,4 | 25,6 | 83,8% | 87,0% |
| + prețurile REALE ale vecinilor din aceeași zi (N) | **4,1** | 2,8 | 96,9% | 95,4% |

Concluzii: (1) cunoașterea producției solare/eoliene din România, chiar perfectă, aduce cel mult ~0,9 EUR/MWh (3,6%)
și ~1 punct la baterie, deci un lanț vreme → producție → preț nu poate aduce mai mult, iar o estimare proprie ar aduce
mai puțin; (2) prețul României e stabilit **regional** (cuplarea piețelor): cu prețurile vecinilor din aceeași zi eroarea
scade cu 84%, dar acestea apar abia după 12:00. Eroarea rămasă (~25) vine din condiții regionale de mâine pe care nu le
cunoaștem dimineața (gaz, regenerabile din regiune, fluxuri transfrontaliere). Datele de producție reală sunt în
`generation_actual` (`scripts/backfill_production.py`).

## Informație regională (`scripts/region_experiment.py`, 669 de zile): adoptată
Variabile adăugate modelului de dimineață (R3): prețurile de ieri și de acum 7 zile din AT, DE, CZ, HR, SI, GR, media și
dispersia regională, și prognoza de consum D+1 a vecinilor (HU, BG, RS, SK):

| | MAE | MAE iarna | Baterie (tot) | iarna | cu mix 50/50 |
|---|---|---|---|---|---|
| Modelul simplu (M) | 24,9 | 26,9 | 82,4% | 63,6% | 86,2% |
| + prețuri regionale (R1) | 24,9 | 26,9 | 83,5% | 65,4% | 86,9% |
| + consum vecini (R2) | 24,3 | 25,8 | 83,6% | 68,4% | 86,8% |
| **Ambele (R3), în producție** | **24,0** | **25,2** | **84,6%** | **69,7%** | 86,9% |

Plasă de siguranță: dacă prognoza de consum a vecinilor lipsește la momentul rulării, se folosește modelul simplu (M),
antrenat în paralel (`variant` = `main` / `fallback` în `features_used`). Variabilele regionale se folosesc la
antrenare doar dacă au istoric suficient. Date: `scripts/backfill_region.py` (workflow `Backfill regiune`).

## Reantrenare zilnică
Modelul se refit-uiește o dată pe zi, la prima rulare reușită a dimineții (secunde), pe ultimele 60 de zile **anterioare
zilei prognozate** (nu antrenează niciodată pe ziua pe care o prognozează, nici la re-rulări târzii). Calibrarea
intervalelor de încredere (backtest 90 de zile, ~2–3 min) se recalculează o dată pe săptămână. La eșec, se folosește
modelul existent. Măsurat pe ultimele 200 de zile (`scripts/refit_frequency.py`): eroare 27,1 zilnic, 28,0 săptămânal,
29,3 lunar; prognoza rămâne mai puțin în urmă pe trend (bias −4,0 față de −5,8 și −9,5). Valoarea pe baterie aproape
neschimbată (85,6% / 84,7% / 84,5%).

## Teste de îmbunătățire pe tot istoricul (`scripts/improvements.py`, 671 de zile, toate anotimpurile)
Eroarea medie și valoarea pe baterie (10 MW / 2h, % din câștigul cu prevedere perfectă), antrenare pe 60 de zile:

| Variantă | MAE | Baterie, tot | iarna | vara |
|---|---|---|---|---|
| Prețul de ieri | 28,3 | 81,2% | 60,2% | 87,5% |
| Media pe 7 zile | 28,3 | 86,8% | 71,3% | 90,5% |
| Model de producție (V0) | 24,9 | 82,7% | 63,2% | 87,6% |
| + corecție cu erorile ultimelor 7 zile (V1) | 27,0 | 80,3% | 57,4% | 85,8% |
| Modele separate pe zone orare (V2) | 25,6 | 80,9% | 60,1% | 87,2% |
| Nivel zilnic + formă a zilei (V3) | 25,9 | 83,6% | 66,7% | 88,8% |
| **Mix 50/50 model + medie 7 zile** | **24,6** | **86,6%** | 69,6% | 90,7% |

Nicio îmbunătățire (V1–V4) nu bate media pe 7 zile la valoarea pe baterie. Mixul 50/50 are cea mai mică eroare și
aceeași valoare ca media pe 7 zile, deci domină modelul simplu pe ambele măsurători. Erori pe zone orare: cele mai
mari la prânz și seara; modelul subestimează vârfurile și supraestimează prețurile mici.

## Raport de valoare
`python scripts/value_report.py` simulează decizii (baterie 10 MW 2h/4h, parc solar/eolian 100 MW) luate cu prognoza,
evaluate la prețurile reale, și le compară cu strategii simple („prețul de ieri”, media pe 7 zile). Activele sunt
ipotetice; ENTSO-E nu publică producția pe centrale individuale pentru România, deci parcurile folosesc profilul
național real. Rezultatul actual e în `reports/value_report.html`.

## Experimente de îmbunătățire (`scripts/experiments.py`)
Protocol: variantele se compară pe o perioadă de dezvoltare (24 apr – 23 iul 2026) și se confirmă o singură dată pe
holdout (23 iul – 20 oct 2026). Valoarea se măsoară pe baterie (10 MW / 2h), nu doar prin eroare.
Variante: variabile noi (forma zilei precedente, lag 48h/14z), țintă „corecție la media pe 7 zile”, obiective L1/Huber,
mix 50/50 cu media pe 7 zile. **Niciuna nu a bătut media pe 7 zile pe valoarea bateriei** (86–88% din plafon, față de
88–90% pentru media pe 7 zile), deși eroarea medie scade de la ~33 la ~27 EUR/MWh. Variabilele noi nu au ajutat
și rămân doar experimentale (`EXPERIMENTAL`). Concluzie: în piața asta forma zilei e foarte persistentă; un
forecast mai bun cere informație nouă (indisponibilități, combustibili), nu mai multă ingineria acelorași date.
Rezultate: `reports/experiments_base.json`.

A doua rundă (`--set extras`): producție reală pe tipuri (nuclear, hidro, gaz, cărbune; lag 24h) și indisponibilități
de centrale RO/HU/BG, reconstruite „as-of” 06:00 (doar documente create până atunci, cu ultima revizie cunoscută și
anulările respectate). Nici ele nu au îmbunătățit valoarea bateriei (85,9–86,8% din plafon față de 86,7–86,8% pentru
referință; media pe 7 zile 87,7–89,5%) și au redus eroarea cu cel mult ~0,5 EUR/MWh, în zona zgomotului.
Rezultate: `reports/experiments_extras.json`. Datele rămân în DB (`generation_actual`, `outages`), dar nu sunt
folosite de modelul de producție.

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
