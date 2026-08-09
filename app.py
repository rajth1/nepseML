"""
FastAPI service for the NEPSE bank volatility forecasts.

Deliberately does ONLY reads from Postgres — no model loading, no MLflow
dependency, no live inference. That's all handled by forecast_writer.py
and train.py, running on a schedule via GitHub Actions. This service just
serves whatever's already been written to the forecasts/model_metadata
tables.

Endpoints:
    GET /health                    - liveness + DB connectivity check
    GET /forecasts                 - latest forecast for every ticker
    GET /forecasts/{symbol}        - latest forecast for one ticker
    GET /model/metadata            - current production model's metadata

Run locally:
    uvicorn app:app --reload
"""

import os
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel

DATABASE_URL = os.environ.get("DATABASE_URL")

pool: SimpleConnectionPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
    yield
    pool.closeall()


app = FastAPI(
    title="NEPSE Bank Volatility Forecasts",
    description="Serves pre-computed next-day volatility forecasts for NEPSE commercial bank stocks.",
    lifespan=lifespan,
)

# Permissive CORS — this is a portfolio demo API, not something behind auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_conn():
    return pool.getconn()


def put_conn(conn):
    pool.putconn(conn)


# --- Response models ---

class ForecastResponse(BaseModel):
    symbol: str
    forecast_date: date
    predicted_volatility: float
    model_version: str
    generated_at: datetime


class ModelMetadataResponse(BaseModel):
    model_version: str
    trained_at: datetime
    training_window_start: date
    training_window_end: date
    validation_qlike: float
    validation_mse: float
    feature_importance: dict


class HealthResponse(BaseModel):
    status: str
    database: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            db_status = "ok"
        finally:
            put_conn(conn)
    except Exception as e:
        db_status = f"error: {e}"
    return HealthResponse(status="ok", database=db_status)


@app.get("/forecasts", response_model=list[ForecastResponse])
def get_all_latest_forecasts():
    """
    Latest forecast per ticker, independently — each ticker's OWN most
    recent forecast_date, not just "whatever the single latest date in
    the table is." This stays correct even if one ticker's forecast is
    a day stale for some reason while the others are fresh.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT ON (symbol)
                symbol, forecast_date, predicted_volatility, model_version, generated_at
            FROM forecasts
            ORDER BY symbol, forecast_date DESC
            """
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        put_conn(conn)

    return [
        ForecastResponse(
            symbol=r[0], forecast_date=r[1], predicted_volatility=r[2],
            model_version=r[3], generated_at=r[4],
        )
        for r in rows
    ]


@app.get("/forecasts/{symbol}", response_model=ForecastResponse)
def get_latest_forecast(symbol: str = Path(..., description="Bank ticker symbol, e.g. NABIL")):
    symbol = symbol.upper()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol, forecast_date, predicted_volatility, model_version, generated_at
            FROM forecasts
            WHERE symbol = %s
            ORDER BY forecast_date DESC
            LIMIT 1
            """,
            (symbol,),
        )
        row = cur.fetchone()

        if row is None:
            # Distinguish "not a tracked ticker" from "tracked, but no forecast yet."
            cur.execute("SELECT 1 FROM companies WHERE symbol = %s", (symbol,))
            is_known_ticker = cur.fetchone() is not None
        cur.close()
    finally:
        put_conn(conn)

    if row is None:
        if is_known_ticker:
            raise HTTPException(status_code=404, detail=f"'{symbol}' is tracked but has no forecast yet.")
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not a tracked ticker.")

    return ForecastResponse(
        symbol=row[0], forecast_date=row[1], predicted_volatility=row[2],
        model_version=row[3], generated_at=row[4],
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def get_model_metadata():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT model_version, trained_at, training_window_start, training_window_end,
                   validation_qlike, validation_mse, feature_importance
            FROM model_metadata
            WHERE is_production = true
            LIMIT 1
            """
        )
        row = cur.fetchone()
        cur.close()
    finally:
        put_conn(conn)

    if row is None:
        raise HTTPException(status_code=404, detail="No production model found. Has train.py ever run successfully?")

    return ModelMetadataResponse(
        model_version=row[0], trained_at=row[1],
        training_window_start=row[2], training_window_end=row[3],
        validation_qlike=row[4], validation_mse=row[5],
        feature_importance=row[6],
    )
