# NEPSE Commercial Bank Volatility Forecasting

An end-to-end data engineering and machine learning pipeline that scrapes daily price data for all actively-traded commercial bank stocks on the Nepal Stock Exchange (NEPSE), engineers volatility features, trains a pooled LightGBM model with tracked experiments, and serves next-day volatility forecasts through a deployed REST API — fully automated on a daily/weekly schedule with zero paid infrastructure.

**Live API:** `https://<your-service-name>.onrender.com` — see [API Reference](#api-reference) below.

---

## Why this exists

This is a portfolio project built to demonstrate practical data engineering and ML engineering skills end-to-end: reverse-engineering an obfuscated data source, building a resilient scheduled scraper, designing a normalized schema, engineering leak-free time-series features, running purged walk-forward cross-validation, tracking experiments with MLflow, and deploying a served model — all on free-tier infrastructure.

## Architecture

```
NEPSE (nepalstock.com)
        │  (WASM-obfuscated auth, decoded via basic-bgnr/NepseUnofficialApi)
        ▼
daily_scrape.py  ──────────────►  Postgres (Neon/Supabase)
        │  (GitHub Actions, daily, 5PM Nepal time)     │  companies
        │                                              │  daily_prices
        ▼                                              │  scrape_log
forecast_writer.py  ─────────────────────────────────► │  forecasts
        │  (same daily workflow, scores latest         │  model_metadata
        │   data with current Production model)         ▼
        │
train.py + cv.py  ──────────────►  MLflow (SQLite store, committed to repo)
        │  (GitHub Actions, weekly, Friday)              tracking + model registry
        ▼
model_metadata (Postgres)  ◄──────────────────────────────┘

app.py (FastAPI)  ◄── reads only from Postgres, no live model inference
        │
        ▼
Docker container on Render
```

**Design principle throughout:** the serving layer (FastAPI on Render) never touches MLflow or runs model inference directly — it only reads pre-computed forecasts and metadata from Postgres. This keeps the deployed service simple, fast, and fully decoupled from the training/scraping infrastructure.

## Tech stack

| Layer | Tool |
|---|---|
| Data source | NEPSE official site, via reverse-engineered auth (`basic-bgnr/NepseUnofficialApi`) |
| Database | PostgreSQL (Neon/Supabase free tier) |
| Scheduling | GitHub Actions (cron) |
| Modeling | LightGBM (pooled, ticker as categorical) |
| Experiment tracking | MLflow (SQLite backend, file-based artifacts, committed to the repo) |
| Serving | FastAPI |
| Deployment | Docker on Render (free tier) |

## Repository structure

```
daily_scrape.py           Daily scraper: checks market status, fetches OHLCV, upserts to Postgres
backfill_historical.py    One-time historical backfill (chunked, ~1 year — NEPSE's own depth limit)
features.py               Feature engineering: Parkinson variance, HAR-style lags, train/inference paths
cv.py                      Purged, expanding-window walk-forward cross-validation
train.py                   Orchestrates CV evaluation + final model fit, MLflow registration/promotion
forecast_writer.py         Daily job: scores latest data with the current Production model
app.py                     FastAPI service — reads forecasts/model_metadata from Postgres only
schema.sql                 Postgres schema: companies, daily_prices, scrape_log, model_metadata, forecasts
requirements.txt           Full pipeline dependencies (scraping, training, forecasting)
requirements-api.txt       Slim dependency set for the deployed API image only
Dockerfile                 Container definition for the FastAPI service
render.yaml                Render Blueprint (infrastructure as code)
.github/workflows/
  daily-scrape.yml          Daily: scrape → write forecast
  retrain.yml                Weekly: retrain → evaluate → promote → commit MLflow store
```

## Key design decisions

- **Ticker universe is derived programmatically**, not hardcoded: `sectorName == "Commercial Banks" AND instrumentType == "Equity" AND status == "A"`, re-evaluated from NEPSE's own data on every run. Currently resolves to 19 tickers.
- **Volatility target: Parkinson high-low estimator**, not squared close-to-close returns — more statistically efficient given only one high/low/close observation per day, and NEPSE doesn't expose an open price.
- **Model trained on log-volatility, not raw variance** — guarantees positive predictions by construction (no output clipping needed) and matches standard practice in the realized-volatility forecasting literature. QLIKE and MSE are still evaluated on the untransformed variance scale, since QLIKE is only defined there.
- **Purged, expanding-window walk-forward cross-validation**, not a single static holdout — a 1-day purge gap at each fold boundary removes label/feature overlap at the train/validation boundary. Expanding (not fixed-rolling) windows because NEPSE's own history API caps out at ~1 year regardless of requested date range (confirmed empirically), so "rolling window" collapses to "use everything available" in practice.
- **Model evaluation includes two naive baselines** (persistence, 22-day rolling average) computed on identical validation rows every fold — a pooled ML model is only worth deploying if it earns its complexity over trivial alternatives.
- **`model_metadata` exists in Postgres specifically because the deployed API has no network path to the training job's local MLflow store** — the training job snapshots the promoted model's metadata into Postgres at promotion time; the API reads only from there.
- **Non-trading days are detected via NEPSE's own market-status API**, not a manually maintained holiday calendar, with a fallback heuristic (all tickers returning empty data) in case that check is ever ambiguous.

## Known limitations

- NEPSE's history endpoint does not serve data older than ~1 year regardless of requested range — confirmed via direct testing, not assumed. This caps the usable training history until enough daily scrapes accumulate naturally over time.
- The current model performs comparably to a simple 22-day rolling-average baseline on QLIKE, and clearly beats naive persistence. It has not yet been shown to decisively outperform the strongest naive baseline — a known, explainable limitation given the current data volume (documented as a direction for future tuning, not hidden).
- `forecast_date` is computed as the next *calendar* day after the latest available data, not the next NEPSE *trading* day specifically (which would require holiday-calendar awareness). Acceptable given forecasts are consumed as "latest per ticker," not queried by exact date.
- No authentication on the API — appropriate for a portfolio demo, not for a real production deployment.

## Local development

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export DATABASE_URL="postgresql://..."

python backfill_historical.py     # one-time
python train.py                   # trains, evaluates via CV, registers + promotes to MLflow
python forecast_writer.py         # scores latest data with the current Production model

uvicorn app:app --reload          # run the API locally
```

## API reference

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness + Postgres connectivity check |
| `GET /forecasts` | Latest forecast for every tracked ticker |
| `GET /forecasts/{symbol}` | Latest forecast for one ticker (e.g. `/forecasts/NABIL`) |
| `GET /model/metadata` | Current production model's version, training window, validation metrics, feature importance |

Interactive documentation available at `/docs` on the deployed instance.

## Future work

- Residual-modeling approach (predict deviation from the rolling-22-day baseline rather than the raw level) to try to decisively beat the naive baseline.
- Nepal-trading-calendar awareness for exact next-trading-day forecast dates.
- Expanded feature set as more historical data accumulates naturally via the daily scrape.
