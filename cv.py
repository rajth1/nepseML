"""
Purged, rolling-window walk-forward cross-validation for the NEPSE bank
volatility model.

Why "purged": our target for day t is the variance realized on day t+1.
A training row dated the day right before a validation window's start has
a label built from price data that falls on the FIRST day of that
validation window — a subtle boundary overlap between train and
validation. We drop (purge) that many days of training data immediately
before each validation window to remove it.

Why "rolling window" became "expanding window": NEPSE's history endpoint
turned out to cap at ~1 year regardless of requested start_date (confirmed
empirically — see test_history_depth.py). With only ~200 usable trading
days total, a fixed 500-day rolling window is impossible. An expanding
window is actually the more honest choice anyway: your own daily_prices
table keeps growing by one real day every day via daily_scrape.py,
independent of what NEPSE's live API can serve — so the real production
retrain policy is "use everything accumulated so far," which an expanding
window mirrors directly.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

MIN_TRAIN_WINDOW_DAYS = 120   # minimum history before the first fold
VAL_WINDOW_DAYS = 30
PURGE_DAYS = 1              # matches the 1-day-ahead target horizon
LOG_EPSILON = 1e-10



def qlike(y_true: np.ndarray, y_pred_variance: np.ndarray) -> float:
    """QLIKE loss (Patton, 2011), evaluated on the VARIANCE scale."""
    ratio = y_true / y_pred_variance
    return float(np.mean(ratio - np.log(ratio) - 1))


def make_qlike_feval(log_epsilon: float):
    """
    LightGBM custom eval function. The model trains on log-VOLATILITY
    (0.5 * log(variance)), so to score QLIKE (a variance-scale metric) we
    have to: exponentiate to get volatility back, then SQUARE it to get
    variance, then compute QLIKE. Both y_true and y_pred here are in
    log-volatility space.

    Note: newer LightGBM sklearn-API versions call custom eval functions
    as func(y_true, y_pred) directly, rather than func(y_pred, dataset)
    with a .get_label() call — this matches that newer signature.
    """
    def qlike_feval(y_true_log_vol, y_pred_log_vol):
        y_true_variance = np.exp(y_true_log_vol) ** 2
        y_pred_variance = np.clip(np.exp(y_pred_log_vol) ** 2, log_epsilon, None)
        value = qlike(y_true_variance, y_pred_variance)
        return "qlike", value, False  # False = lower is better
    return qlike_feval


def generate_folds(unique_dates, min_train_window=MIN_TRAIN_WINDOW_DAYS,
                    val_window=VAL_WINDOW_DAYS, purge=PURGE_DAYS):
    """
    Yields (train_dates, val_dates) for each fold. EXPANDING training
    window (always starts at the earliest available date), non-overlapping
    validation windows stepping forward by val_window each time, with
    `purge` days dropped from the end of each training window.
    """
    folds = []
    n = len(unique_dates)
    train_end_idx = min_train_window
    while train_end_idx + val_window <= n:
        train_dates = unique_dates[0: train_end_idx - purge]
        val_dates = unique_dates[train_end_idx: train_end_idx + val_window]
        folds.append((train_dates, val_dates))
        train_end_idx += val_window
    return folds


def _fit_one_fold(train_df, val_df, feature_cols, lgb_params):
    y_train_log_vol = 0.5 * np.log(train_df["target_next_day_variance"].clip(lower=LOG_EPSILON))
    y_val_log_vol = 0.5 * np.log(val_df["target_next_day_variance"].clip(lower=LOG_EPSILON))

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train_df[feature_cols], y_train_log_vol,
        categorical_feature=["symbol"],
        eval_set=[(val_df[feature_cols], y_val_log_vol)],
        eval_metric=make_qlike_feval(LOG_EPSILON),
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    pred_log_vol = model.predict(val_df[feature_cols])
    pred_variance = np.clip(np.exp(pred_log_vol) ** 2, LOG_EPSILON, None)
    true_variance = val_df["target_next_day_variance"].values

    return model, pred_variance, true_variance


def run_walk_forward_cv(df: pd.DataFrame, feature_cols: list, lgb_params: dict) -> pd.DataFrame:
    """
    Runs every fold, scores the model AND both naive baselines
    (persistence, 22-day rolling average) on identical validation rows
    each time, and returns a per-fold results DataFrame for aggregation
    and MLflow logging.
    """
    unique_dates = sorted(df["business_date"].unique())
    folds = generate_folds(unique_dates)
    print(f"Generated {len(folds)} walk-forward folds "
          f"(expanding window, min initial train={MIN_TRAIN_WINDOW_DAYS}d, "
          f"val window={VAL_WINDOW_DAYS}d, purge={PURGE_DAYS}d).")

    records = []
    for i, (train_dates, val_dates) in enumerate(folds):
        train_df = df[df["business_date"].isin(train_dates)]
        val_df = df[df["business_date"].isin(val_dates)]

        _, pred_variance, true_variance = _fit_one_fold(train_df, val_df, feature_cols, lgb_params)

        model_qlike = qlike(true_variance, pred_variance)
        model_mse = float(np.mean((true_variance - pred_variance) ** 2))

        persistence_pred = np.clip(val_df["realized_var_lag_1d"].values, LOG_EPSILON, None)
        rolling22_pred = np.clip(val_df["realized_var_lag_22d"].values, LOG_EPSILON, None)

        records.append({
            "fold": i,
            "train_start": train_dates[0], "train_end": train_dates[-1],
            "val_start": val_dates[0], "val_end": val_dates[-1],
            "model_qlike": model_qlike,
            "model_mse": model_mse,
            "persistence_qlike": qlike(true_variance, persistence_pred),
            "persistence_mse": float(np.mean((true_variance - persistence_pred) ** 2)),
            "rolling22_qlike": qlike(true_variance, rolling22_pred),
            "rolling22_mse": float(np.mean((true_variance - rolling22_pred) ** 2)),
        })
        print(f"  Fold {i}: val {val_dates[0].date()}..{val_dates[-1].date()} — "
              f"model QLIKE={model_qlike:.4f}, rolling22 QLIKE={records[-1]['rolling22_qlike']:.4f}")

    return pd.DataFrame(records)
