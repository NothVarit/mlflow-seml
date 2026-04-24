"""
Register the already-trained local checkpoint into MLflow when the registry is empty.
Run this once after `make full-stack` if the API fails to load the model.

Usage:
    python bootstrap_model.py
    MLFLOW_TRACKING_URI=http://localhost:5001 python bootstrap_model.py
"""
import os
import sys

import mlflow
import mlflow.transformers
import transformers
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "DistilBERT_Tagger"
MODEL_DIR = os.getenv("MODEL_DIR", "./distilbert_tagging_model")


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"ERROR: local model dir not found: {MODEL_DIR}")
        sys.exit(1)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Check whether the production alias already exists — nothing to do if so
    try:
        existing = client.get_model_version_by_alias(MODEL_NAME, "production")
        print(f"Model already registered: {MODEL_NAME} version={existing.version} alias=production — nothing to do.")
        return
    except Exception:
        pass

    print(f"Registering {MODEL_DIR} into MLflow at {TRACKING_URI} ...")
    mlflow.set_experiment("Article_Tagging")

    pipeline = transformers.pipeline(
        task="text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        top_k=None,
    )

    with mlflow.start_run(run_name="bootstrap"):
        mlflow.log_param("source", "bootstrap_from_local_checkpoint")
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="distilbert_tagger",
            registered_model_name=MODEL_NAME,
        )

    # Wait for the version to appear then set alias
    import time
    for _ in range(15):
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            break
        time.sleep(2)
    else:
        print("ERROR: model version never appeared in registry")
        sys.exit(1)

    latest_version = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias(MODEL_NAME, "production", latest_version)
    print(f"Done. {MODEL_NAME} version={latest_version} alias=production")


if __name__ == "__main__":
    main()
