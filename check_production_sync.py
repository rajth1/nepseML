"""
Checks MLflow's registry directly for whichever model version currently
holds the "Production" stage, and compares it against what Postgres's
model_metadata table claims is production. These SHOULD always agree —
if they don't, it means some past training run got far enough to promote
a version in MLflow but failed before completing the Postgres write,
leaving the two sources of truth out of sync.

Run this in the SAME environment where mlflow.db/mlruns_artifacts live —
i.e., as a GitHub Actions step, not locally (same reasoning as before:
the artifact paths are baked in for whichever machine created them).
"""

import os

import psycopg2
from mlflow.tracking import MlflowClient
import mlflow

REGISTERED_MODEL_NAME = "nepse_bank_volatility_model"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

print("=== All registered model versions and their MLflow stage ===")
all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
for v in sorted(all_versions, key=lambda x: int(x.version)):
    print(f"  version={v.version}  stage={v.current_stage}  run_id={v.run_id}")

production_versions = [v for v in all_versions if v.current_stage == "Production"]
if not production_versions:
    print("\nNo version is currently in the Production stage in MLflow!")
else:
    print(f"\nMLflow says PRODUCTION version is: {production_versions[0].version} "
          f"(run_id={production_versions[0].run_id})")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("SELECT model_version, mlflow_run_id FROM model_metadata WHERE is_production = true")
row = cur.fetchone()
conn.close()

if row is None:
    print("\nPostgres model_metadata has NO row marked is_production=true!")
else:
    print(f"Postgres model_metadata says PRODUCTION version is: {row[0]} (run_id={row[1]})")

if production_versions and row:
    if str(production_versions[0].version) == str(row[0]):
        print("\nMATCH — MLflow and Postgres agree on the production version.")
    else:
        print("\nMISMATCH — MLflow and Postgres disagree. Forecasts have been using "
              "whatever MLflow says (since forecast_writer.py loads via the MLflow "
              "Production alias directly), but model_metadata's displayed info is wrong.")
