import os
import time

import mlflow
import mlflow.transformers
from fastapi import FastAPI

app = FastAPI()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/DistilBERT_Tagger@production")
MODEL_FALLBACK_URI = os.getenv("MODEL_FALLBACK_URI", "models:/DistilBERT_Tagger/latest")

mlflow.set_tracking_uri(TRACKING_URI)

tagging_pipeline = None
for attempt in range(10):
    try:
        tagging_pipeline = mlflow.transformers.load_model(MODEL_URI)
        print(f"Model loaded from {MODEL_URI}")
        break
    except Exception as primary_error:
        try:
            tagging_pipeline = mlflow.transformers.load_model(MODEL_FALLBACK_URI)
            print(f"Model loaded from fallback {MODEL_FALLBACK_URI}")
            break
        except Exception:
            if attempt == 9:
                raise RuntimeError(f"Could not load model after 10 attempts: {primary_error}")
        print(f"MLflow not ready, retrying in 6s... ({primary_error})")
        time.sleep(6)

THRESHOLD = 0.3


@app.post("/predict")
def predict(data: dict):
    inputs = data["inputs"]
    if isinstance(inputs, str):
        inputs = [inputs]
    raw_results = tagging_pipeline(inputs)
    if raw_results and isinstance(raw_results[0], dict):
        raw_results = [raw_results]
    predictions = []
    for result in raw_results:
        tags = [item["label"] for item in result if item["score"] >= THRESHOLD]
        predictions.append(tags)
    return {"predictions": predictions}
