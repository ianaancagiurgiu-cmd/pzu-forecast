-- Preturi PZU reale, pe zona (RO + vecini HU, BG, RS, SK). Timpul e stocat in UTC.
CREATE TABLE IF NOT EXISTS prices_actual (
    timestamp     TIMESTAMPTZ NOT NULL,  -- inceputul intervalului de 15 min
    zone          VARCHAR(2)  NOT NULL,
    price_eur_mwh NUMERIC,
    source        VARCHAR(20),           -- 'ENTSOE' | 'OPCOM' | 'SYNTHETIC'
    PRIMARY KEY (timestamp, zone)
);

-- Date de intrare disponibile inainte de 12:00 CET (forecast-uri si vreme)
CREATE TABLE IF NOT EXISTS forecasts_input (
    timestamp         TIMESTAMPTZ PRIMARY KEY,
    solar_forecast_mw NUMERIC,
    wind_forecast_mw  NUMERIC,
    load_forecast_mw  NUMERIC,
    capacity_hu_ro_mw NUMERIC,
    capacity_ro_hu_mw NUMERIC,
    capacity_bg_ro_mw NUMERIC,
    capacity_ro_bg_mw NUMERIC,
    temp_c            NUMERIC,
    wind_speed_ms     NUMERIC,
    radiation_wm2     NUMERIC
);

CREATE TABLE IF NOT EXISTS price_forecast (
    timestamp             TIMESTAMPTZ PRIMARY KEY,
    forecast_generated_at TIMESTAMPTZ,
    predicted_price       NUMERIC,
    ci_lower              NUMERIC,  -- limita inferioara 90%
    ci_upper              NUMERIC,  -- limita superioara 90%
    model_version         VARCHAR(20),
    features_used         JSONB
);

CREATE TABLE IF NOT EXISTS forecast_accuracy (
    forecast_date        DATE PRIMARY KEY,
    mae                  NUMERIC,
    rmse                 NUMERIC,
    smape                NUMERIC,
    directional_hit_rate NUMERIC,
    coverage_90          NUMERIC,
    model_version        VARCHAR(20)
);
