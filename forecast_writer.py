"""
Daily forecast writing job.

Loads whichever model is currently marked "Production" in MLflow (this
might be several days old as retraining happens weekly, this job runs
daily), scores the latest available row of features for each of the 19
tickers, and writes the result to the forecasts table.

Runs AFTER daily_scrape.py in the same workflow, so "latest available
row" reflects today's just-scraped data. On a non-trading day, the latest
available row is unchanged from yesterday this job will just harmlessly
overwrite the same forecast with itself (upsert), not an error case worth
special-handling.

Note on units: the model is trained on log-VOLATILITY internally, and
QLIKE/MSE evaluation happens on the variance scale but the
forecasts.predicted_volatility column is named for volatility (standard
deviation), so that's what gets stored here, converted from the model's
internal variance-scale prediction.


Usage:
    python forecast_writer.py
"""

import os
import sys
from datetime import timedelta

import mlflow.lightgbm
import numpy as np
import psycopg2

from features import build_latest_inference_rows

REGISTERED_MODEL_NAME = "nepse_bank_volatility_model"
LOG_EPSILON = 1e-10

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is not set.")
    sys.exit(1)


def get_current_production_version(cur):
    cur.execute("SELECT model_version FROM model_metadata WHERE is_production = true LIMIT 1")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            "No production model found in model_metadata. Run train.py at least once before forecasting."
        )
    return row[0]


def upsert_forecast(cur, symbol, forecast_date, predicted_volatility, model_version):
    cur.execute(
        """
        INSERT INTO forecasts (symbol, forecast_date, predicted_volatility, model_version, generated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (symbol, forecast_date) DO UPDATE SET
            predicted_volatility = EXCLUDED.predicted_volatility,
            model_version = EXCLUDED.model_version,
            generated_at = now()
        """,
        (symbol, forecast_date, predicted_volatility, model_version),
    )


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    model_version = get_current_production_version(cur)
    print(f"Using production model version {model_version}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")

    inference_df = build_latest_inference_rows(conn)
    print(f"Scoring {len(inference_df)} tickers (latest available row each).")

    feature_cols = [c for c in inference_df.columns if c not in ("business_date", "symbol")]
    # model was trained with feature_cols including "symbol" as a categorical
    # column — reconstruct the exact same column set/order used at training time.
    X = inference_df[["symbol"] + feature_cols]

    pred_log_vol = model.predict(X)
    pred_volatility = np.exp(pred_log_vol)  # already on the volatility (std-dev) scale

    written = 0
    for i, row in inference_df.iterrows():
        symbol = row["symbol"]
        forecast_date = row["business_date"].date() + timedelta(days=1)
        upsert_forecast(cur, symbol, forecast_date, float(pred_volatility[i]), model_version)
        written += 1
        print(f"  {symbol}: forecast for {forecast_date} = {pred_volatility[i]:.6f} (volatility)")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nWrote {written} forecasts using model version {model_version}.")


if __name__ == "__main__":
    main()
