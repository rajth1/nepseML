-- ============================================================================
-- NEPSE Banking Sector Volatility Forecasting POSTGRESQL schema
-- ==============================================================================


-- ----------------------------------------------------------------------------
-- 1. companies — ticker roster, re-synced from getCompanyList() each run
-- ----------------------------------------------------------------------------
CREATE TABLE companies (
    symbol           VARCHAR(20) PRIMARY KEY,
    security_id      VARCHAR(20) NOT NULL,   -- NEPSE's internal ID; needed to call
                                              -- getCompanyPriceVolumeHistory
    company_name     TEXT NOT NULL,
    sector_name      TEXT NOT NULL DEFAULT 'Commercial Banks',
    instrument_type  TEXT NOT NULL DEFAULT 'Equity',
    status           CHAR(1) NOT NULL CHECK (status IN ('A', 'D')),  -- Active / Delisted
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_synced_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE companies IS
    'Roster of commercial bank equity tickers, re-derived from NEPSE sector data each daily run.';


-- ----------------------------------------------------------------------------
-- 2. daily_prices — core fact table (backfill + daily job both write here)
-- ----------------------------------------------------------------------------
CREATE TABLE daily_prices (
    symbol                  VARCHAR(20) NOT NULL REFERENCES companies(symbol),
    business_date           DATE NOT NULL,
    high_price              NUMERIC(12, 2),
    low_price               NUMERIC(12, 2),
    close_price             NUMERIC(12, 2),
    total_traded_quantity   BIGINT,
    total_traded_value      NUMERIC(18, 2),     -- turnover, NPR
    total_trades            INTEGER,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, business_date)
);

CREATE INDEX idx_daily_prices_date ON daily_prices (business_date);

COMMENT ON TABLE daily_prices IS
    'One row per (ticker, trading day). Populated by getCompanyPriceVolumeHistory '
    'for both historical backfill and the ongoing daily scrape. No open_price field '
    '— NEPSE does not expose it; volatility target/features use high/low (Parkinson) '
    'and close-to-close returns instead.';


-- ----------------------------------------------------------------------------
-- 3. scrape_log — one row per daily job run
-- ----------------------------------------------------------------------------
CREATE TABLE scrape_log (
    id                  BIGSERIAL PRIMARY KEY,
    run_started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_finished_at     TIMESTAMPTZ,
    status              TEXT NOT NULL CHECK (
                            status IN ('success', 'partial_failure', 'failure', 'non_trading_day')
                        ),
    tickers_attempted   INTEGER,
    tickers_succeeded   INTEGER,
    tickers_failed      INTEGER,
    failed_symbols      TEXT[],     -- quick triage without joining anywhere
    error_detail        TEXT
);

COMMENT ON TABLE scrape_log IS
    'Audit trail for the daily scrape job. status=non_trading_day is set when '
    'client.isNepseOpen()/getMarketStatus() reports the market was closed, so '
    'holidays are never mistaken for scrape failures.';


-- ----------------------------------------------------------------------------
-- 4. model_metadata — cache of the MLflow "production" model
-- ----------------------------------------------------------------------------
CREATE TABLE model_metadata (
    model_version           TEXT PRIMARY KEY,
    mlflow_run_id            TEXT NOT NULL,
    is_production            BOOLEAN NOT NULL DEFAULT false,
    trained_at               TIMESTAMPTZ NOT NULL,
    training_window_start    DATE NOT NULL,
    training_window_end      DATE NOT NULL,
    validation_qlike         DOUBLE PRECISION,
    validation_mse           DOUBLE PRECISION,
    feature_importance       JSONB,
    hyperparameters          JSONB,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enforce "at most one production model at a time" at the database level,
-- rather than trusting application code to always unset the previous one.
CREATE UNIQUE INDEX idx_model_metadata_one_production
    ON model_metadata (is_production)
    WHERE is_production = true;

COMMENT ON TABLE model_metadata IS
    'Snapshot written by the training job whenever it promotes a model to '
    'production in MLflow. The FastAPI /model/metadata endpoint reads only from '
    'here — it has no network path to the training job''s local MLflow store.';


-- ----------------------------------------------------------------------------
-- 5. forecasts — what the FastAPI service actually serves
-- ----------------------------------------------------------------------------
CREATE TABLE forecasts (
    symbol                  VARCHAR(20) NOT NULL REFERENCES companies(symbol),
    forecast_date           DATE NOT NULL,   -- the date being forecast FOR (next trading day)
    predicted_volatility    NUMERIC(12, 6) NOT NULL,
    model_version           TEXT NOT NULL REFERENCES model_metadata(model_version),
    generated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, forecast_date)
);

CREATE INDEX idx_forecasts_date ON forecasts (forecast_date);

COMMENT ON TABLE forecasts IS
    'Pre-computed forecasts written by the forecast-writing job (Phase 6). '
    'The API (Phase 7) reads from here only — no live model inference per request.';
