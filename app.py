import os
import time
import mlflow.transformers
from fastapi import FastAPI

app = FastAPI()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/DistilBERT_Tagger/latest")

mlflow.set_tracking_uri(TRACKING_URI)

tagging_pipeline = None
for attempt in range(10):
    try:
        tagging_pipeline = mlflow.transformers.load_model(MODEL_URI)
        print(f"Model loaded from {MODEL_URI}")
        break
    except Exception as e:
        if attempt == 9:
            raise RuntimeError(f"Could not load model after 10 attempts: {e}")
        print(f"MLflow not ready, retrying in 6s... ({e})")
        time.sleep(6)

THRESHOLD = 0.3

@app.post("/predict")
def predict(data: dict):
    inputs = data["inputs"]
    if isinstance(inputs, str):
        inputs = [inputs]
    raw_results = tagging_pipeline(inputs)
    # Single input returns flat list; multiple inputs returns list of lists
    if raw_results and isinstance(raw_results[0], dict):
        raw_results = [raw_results]
    predictions = []
    for result in raw_results:
        tags = [r["label"] for r in result if r["score"] >= THRESHOLD]
        predictions.append(tags)
    return {"predictions": predictions}
