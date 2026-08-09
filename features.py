"""
Feature engineering for the NEPSE bank volatility model.

Target: next-day Parkinson realized VARIANCE (not standard deviation).
This is deliberate — QLIKE, the loss metric specified in the brief, is
defined for variance forecasts, not volatility-in-std-dev-terms. Keeping
everything on the variance scale end-to-end avoids a sqrt-related mismatch
between what the model predicts and what QLIKE/MSE actually score.

Parkinson estimator for a single day:
    parkinson_variance_t = (1 / (4 * ln 2)) * (ln(High_t / Low_t)) ** 2

All lagged/rolling features use only information available THROUGH day t
to predict day t+1's realized variance — there's no leakage of future
information into the features.
"""

import numpy as np
import pandas as pd

PARKINSON_CONST = 1 / (4 * np.log(2))

# HAR-RV-style horizons: daily / weekly / monthly.
VOL_LAG_WINDOWS = [1, 5, 22]
RETURN_LAG_WINDOWS = [1, 5]
TURNOVER_LAG_WINDOWS = [5, 10]


def load_daily_prices(conn) -> pd.DataFrame:
    """Pull all daily_prices rows into a DataFrame, sorted for rolling ops."""
    df = pd.read_sql(
        """
        SELECT symbol, business_date, high_price, low_price, close_price,
               total_traded_quantity, total_traded_value, total_trades
        FROM daily_prices
        ORDER BY symbol, business_date
        """,
        conn,
    )
    df["business_date"] = pd.to_datetime(df["business_date"])
    return df


def _add_parkinson_variance(df: pd.DataFrame) -> pd.DataFrame:
    df["parkinson_variance"] = PARKINSON_CONST * (
        np.log(df["high_price"] / df["low_price"]) ** 2
    )
    return df


def _add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close_price"] / df.groupby("symbol")["close_price"].shift(1))
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared feature computation, used by both the training path (which adds
    a target and drops the most recent day per ticker) and the inference
    path (which needs exactly that most-recent day, with no target).
    """
    df = _add_parkinson_variance(df)
    df = _add_log_return(df)

    grouped = df.groupby("symbol", group_keys=False)

    for window in VOL_LAG_WINDOWS:
        col = f"realized_var_lag_{window}d"
        df[col] = grouped["parkinson_variance"].apply(
            lambda s: s.shift(1).rolling(window).mean()
        )

    for window in RETURN_LAG_WINDOWS:
        col = f"return_lag_{window}d"
        df[col] = grouped["log_return"].apply(
            lambda s: s.shift(1).rolling(window).mean()
        )

    for window in TURNOVER_LAG_WINDOWS:
        df[f"turnover_lag_{window}d"] = grouped["total_traded_value"].apply(
            lambda s: s.shift(1).rolling(window).mean()
        )
        df[f"volume_lag_{window}d"] = grouped["total_traded_quantity"].apply(
            lambda s: s.shift(1).rolling(window).mean()
        )

    df["day_of_week"] = df["business_date"].dt.dayofweek
    df["symbol"] = df["symbol"].astype("category")
    return df


def build_feature_dataframe(conn) -> pd.DataFrame:
    """
    TRAINING path. Returns one row per (symbol, business_date) with all
    lagged/rolling feature columns plus target_next_day_variance. Rows
    with NaN features (start-of-series warmup) or NaN target (the most
    recent day per ticker, which has no known "tomorrow" yet) are dropped.
    """
    df = load_daily_prices(conn)
    df = _compute_features(df)
    grouped = df.groupby("symbol", group_keys=False)

    df["target_next_day_variance"] = grouped["parkinson_variance"].shift(-1)

    feature_cols = (
        [f"realized_var_lag_{w}d" for w in VOL_LAG_WINDOWS]
        + [f"return_lag_{w}d" for w in RETURN_LAG_WINDOWS]
        + [f"turnover_lag_{w}d" for w in TURNOVER_LAG_WINDOWS]
        + [f"volume_lag_{w}d" for w in TURNOVER_LAG_WINDOWS]
    )

    result = df[["business_date", "symbol"] + feature_cols + ["day_of_week", "target_next_day_variance"]].copy()

    before = len(result)
    result = result.dropna(subset=feature_cols + ["target_next_day_variance"])
    after = len(result)
    print(f"build_feature_dataframe: {before} rows before dropping NaN edges, {after} after.")

    return result


def build_latest_inference_rows(conn) -> pd.DataFrame:
    """
    INFERENCE path. Returns exactly ONE row per ticker — its most recent
    available business_date — with the same feature columns as training,
    but NO target column (since tomorrow hasn't happened yet; that's
    exactly what we're forecasting). This is the row the forecast-writing
    job feeds to the model.
    """
    df = load_daily_prices(conn)
    df = _compute_features(df)

    feature_cols = (
        [f"realized_var_lag_{w}d" for w in VOL_LAG_WINDOWS]
        + [f"return_lag_{w}d" for w in RETURN_LAG_WINDOWS]
        + [f"turnover_lag_{w}d" for w in TURNOVER_LAG_WINDOWS]
        + [f"volume_lag_{w}d" for w in TURNOVER_LAG_WINDOWS]
    )

    result = df[["business_date", "symbol"] + feature_cols + ["day_of_week"]].copy()
    result = result.dropna(subset=feature_cols)  # drop any ticker without enough history yet

    # Keep only the latest row per ticker.
    latest_idx = result.groupby("symbol")["business_date"].idxmax()
    return result.loc[latest_idx].reset_index(drop=True)


if __name__ == "__main__":
    import os
    import psycopg2

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    features_df = build_feature_dataframe(conn)
    conn.close()

    print(features_df.head(10))
    print(f"\nShape: {features_df.shape}")
    print(f"Tickers represented: {features_df['symbol'].nunique()}")
