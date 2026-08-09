"""
Two distinct steps, deliberately kept separate:

  1. Purged walk-forward CV across many historical folds — this produces
     the QLIKE/MSE numbers we actually trust and log to MLflow, since a
     single 30-day holdout is too small a sample to draw a real
     conclusion from. This step does NOT produce the deployed model.

  2. A single final fit on the MOST RECENT rolling window (2 years),
     using the last 30 days of that window for early stopping only. This
     is the model that gets registered and promoted to MLflow Production,
     and whose metadata gets written to Postgres.

Usage:
    python train.py
"""

import json
import os
from datetime import datetime, timezone

import mlflow
import mlflow.lightgbm
import numpy as np
import psycopg2
from mlflow.tracking import MlflowClient

from features import build_feature_dataframe
from cv import (
    run_walk_forward_cv,
    _fit_one_fold,
    qlike,
    MIN_TRAIN_WINDOW_DAYS,
    LOG_EPSILON,
)

REGISTERED_MODEL_NAME = "nepse_bank_volatility_model"
EXPERIMENT_NAME = "nepse_bank_volatility_v2"  # new name, deliberately — see get_or_create_portable_experiment

LGB_PARAMS = {
    "objective": "regression",
    "num_leaves": 7,        # very shallow — small dataset, choke tree complexity
    "max_depth": 4,         # num_leaves alone doesn't cap depth; belt-and-suspenders
    "learning_rate": 0.02,
    "min_data_in_leaf": 50, # a leaf needs real support given ~2-3k rows per fold
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "feature_fraction": 0.8,   # subsample features per tree, extra regularization
    "bagging_fraction": 0.8,   # subsample rows per tree
    "bagging_freq": 1,
    "n_estimators": 1000,   # ceiling only — early stopping picks the real number
    "verbosity": -1,
}

DATABASE_URL = os.environ.get("DATABASE_URL")


def summarize_cv(cv_results):
    summary = {}
    for metric in ["model_qlike", "model_mse", "persistence_qlike", "persistence_mse",
                    "rolling22_qlike", "rolling22_mse"]:
        summary[f"{metric}_mean"] = float(cv_results[metric].mean())
        summary[f"{metric}_std"] = float(cv_results[metric].std())
    return summary


def get_or_create_portable_experiment():
    """
    Forces a RELATIVE artifact location ("./mlruns_artifacts") when the
    experiment is first created, instead of letting MLflow default to an
    absolute path based on the current machine/OS. Without this, an
    experiment created locally on Windows bakes in a "C:\\Users\\..."
    path that breaks the moment a different machine (e.g. a Linux GitHub
    Actions runner) tries to log an artifact against the same mlflow.db.

    Uses a NEW experiment name (nepse_bank_volatility_v2, not the original
    nepse_bank_volatility) deliberately: if an experiment by the old name
    already exists anywhere in the checked-out mlflow.db with a bad
    absolute path baked in, looking it up by name would just reuse that
    corrupted record. A new name guarantees this creates a genuinely fresh
    experiment, sidestepping any ambiguity about which mlflow.db state is
    actually present.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location="./mlruns_artifacts")
    mlflow.set_experiment(EXPERIMENT_NAME)



def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    conn = psycopg2.connect(DATABASE_URL)
    df = build_feature_dataframe(conn)
    feature_cols = [c for c in df.columns if c not in ("business_date", "target_next_day_variance")]

    # --- Step 1: purged walk-forward CV, for evaluation numbers only ---
    print("\n=== Running purged walk-forward CV ===")
    cv_results = run_walk_forward_cv(df, feature_cols, LGB_PARAMS)
    cv_summary = summarize_cv(cv_results)

    print("\n=== CV summary (mean +/- std across folds) ===")
    for name, baseline_key in [("Model", "model"), ("Persistence", "persistence"), ("Rolling 22d", "rolling22")]:
        print(f"{name:12s} QLIKE: {cv_summary[f'{baseline_key}_qlike_mean']:.4f} "
              f"+/- {cv_summary[f'{baseline_key}_qlike_std']:.4f}   "
              f"MSE: {cv_summary[f'{baseline_key}_mse_mean']:.10f} "
              f"+/- {cv_summary[f'{baseline_key}_mse_std']:.10f}")

    # --- Step 2: final production fit on ALL available history ---
    # Expanding window means "production" now just means: use everything
    # accumulated so far. The last 30 days are held out purely for early
    # stopping, not because the training window is capped at that size.
    print("\n=== Fitting final production model on all available history ===")
    unique_dates = sorted(df["business_date"].unique())
    final_train_dates = unique_dates[:-30]
    final_val_dates = unique_dates[-30:]

    final_train_df = df[df["business_date"].isin(final_train_dates)]
    final_val_df = df[df["business_date"].isin(final_val_dates)]

    final_model, final_pred_var, final_true_var = _fit_one_fold(
        final_train_df, final_val_df, feature_cols, LGB_PARAMS
    )
    final_qlike = qlike(final_true_var, final_pred_var)
    final_mse = float(np.mean((final_true_var - final_pred_var) ** 2))
    print(f"Final model's own early-stopping-window QLIKE: {final_qlike:.4f}, MSE: {final_mse:.10f}")
    print("(This number has the same 'peeked at its own validation set' caveat as before -- "
          "the trustworthy numbers are the CV summary above.)")

    feature_importance = dict(zip(feature_cols, [int(v) for v in final_model.feature_importances_]))

    # --- MLflow logging ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    get_or_create_portable_experiment()

    with mlflow.start_run() as run:
        mlflow.log_params(LGB_PARAMS)
        mlflow.log_param("min_initial_train_window_days", MIN_TRAIN_WINDOW_DAYS)
        mlflow.log_param("val_window_days", 30)
        mlflow.log_param("purge_days", 1)
        mlflow.log_param("feature_set", feature_cols)
        mlflow.log_param("n_cv_folds", len(cv_results))

        for k, v in cv_summary.items():
            mlflow.log_metric(f"cv_{k}", v)
        mlflow.log_metric("final_fit_qlike", final_qlike)
        mlflow.log_metric("final_fit_mse", final_mse)

        mlflow.log_dict(feature_importance, "feature_importance.json")
        mlflow.log_dict(cv_results.to_dict(orient="records"), "cv_fold_results.json")

        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise RuntimeError("Could not find the model version just registered -- MLflow registry issue.")
    model_version = versions[0].version

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"\nPromoted {REGISTERED_MODEL_NAME} v{model_version} (run {run_id}) to Production.")

    # --- Snapshot into Postgres model_metadata ---
    # Store the CV-derived mean QLIKE/MSE here, not the final fit's own
    # number -- the CV summary is the trustworthy estimate of expected
    # out-of-sample performance; the final fit's number has the early-
    # stopping "peeking" caveat.
    cur = conn.cursor()
    cur.execute("UPDATE model_metadata SET is_production = false WHERE is_production = true")
    cur.execute(
        """
        INSERT INTO model_metadata (
            model_version, mlflow_run_id, is_production, trained_at,
            training_window_start, training_window_end,
            validation_qlike, validation_mse, feature_importance, hyperparameters
        )
        VALUES (%s, %s, true, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_version) DO UPDATE SET
            is_production = true,
            validation_qlike = EXCLUDED.validation_qlike,
            validation_mse = EXCLUDED.validation_mse
        """,
        (
            str(model_version),
            run_id,
            datetime.now(timezone.utc),
            final_train_dates[0].date(),
            final_train_dates[-1].date(),
            cv_summary["model_qlike_mean"],
            cv_summary["model_mse_mean"],
            json.dumps(feature_importance),
            json.dumps(LGB_PARAMS),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()

    print("model_metadata updated. Training run complete.")


if __name__ == "__main__":
    main()
