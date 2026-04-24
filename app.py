import logging
import os
import time

import mlflow
import mlflow.transformers
import transformers
from mlflow import MlflowClient
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/DistilBERT_Tagger@production")
MODEL_FALLBACK_URI = os.getenv("MODEL_FALLBACK_URI", "models:/DistilBERT_Tagger/latest")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./distilbert_tagging_model")
MODEL_RELOAD_TOKEN = os.getenv("MODEL_RELOAD_TOKEN")
MODEL_NAME = MODEL_URI.removeprefix("models:/").split("@", 1)[0].split("/", 1)[0]

mlflow.set_tracking_uri(TRACKING_URI)

tagging_pipeline = None
loaded_model_uri = None


def load_tagging_pipeline():
    global tagging_pipeline, loaded_model_uri
    for attempt in range(10):
        try:
            tagging_pipeline = mlflow.transformers.load_model(MODEL_URI)
            loaded_model_uri = MODEL_URI
            logger.info("model_loaded model_uri=%s", MODEL_URI)
            return loaded_model_uri
        except Exception as primary_error:
            try:
                tagging_pipeline = mlflow.transformers.load_model(MODEL_FALLBACK_URI)
                loaded_model_uri = MODEL_FALLBACK_URI
                logger.info("model_loaded_fallback model_uri=%s", MODEL_FALLBACK_URI)
                return loaded_model_uri
            except Exception:
                if attempt == 9:
                    if os.path.isdir(LOCAL_MODEL_PATH):
                        logger.warning(
                            "mlflow_registry_empty falling_back_to_local path=%s error=%r",
                            LOCAL_MODEL_PATH,
                            str(primary_error),
                        )
                        tagging_pipeline = transformers.pipeline(
                            task="text-classification",
                            model=LOCAL_MODEL_PATH,
                            tokenizer=LOCAL_MODEL_PATH,
                            top_k=None,
                        )
                        loaded_model_uri = f"local:{LOCAL_MODEL_PATH}"
                        logger.info("model_loaded_local path=%s", LOCAL_MODEL_PATH)
                        return loaded_model_uri
                    raise RuntimeError(f"Could not load model after 10 attempts: {primary_error}")
            logger.warning("mlflow_not_ready attempt=%s retry_in_seconds=6 error=%r", attempt + 1, str(primary_error))
            time.sleep(6)


load_tagging_pipeline()

THRESHOLD = 0.3


@app.post("/predict")
def predict(data: dict):
    inputs = data["inputs"]
    if isinstance(inputs, str):
        inputs = [inputs]
    started_at = time.perf_counter()
    raw_results = tagging_pipeline(inputs, truncation=True, max_length=128)
    if raw_results and isinstance(raw_results[0], dict):
        raw_results = [raw_results]
    predictions = []
    for result in raw_results:
        tags = [item["label"] for item in result if item["score"] >= THRESHOLD]
        predictions.append(tags)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "ml_predict_succeeded input_count=%s latency_ms=%.2f predictions=%s",
        len(inputs),
        elapsed_ms,
        predictions,
    )
    return {"predictions": predictions}


@app.post("/admin/reload-model")
def reload_model(payload: dict | None = None, x_model_reload_token: str | None = Header(default=None)):
    if not MODEL_RELOAD_TOKEN or x_model_reload_token != MODEL_RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    expected_model_version = None if payload is None else payload.get("expected_model_version")
    model_uri = load_tagging_pipeline()
    if expected_model_version is not None:
        if model_uri != MODEL_URI:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Reloaded fallback model instead of production alias",
                    "loaded_model_uri": model_uri,
                    "expected_model_version": str(expected_model_version),
                },
            )
        production_version = MlflowClient(tracking_uri=TRACKING_URI).get_model_version_by_alias(MODEL_NAME, "production")
        if str(production_version.version) != str(expected_model_version):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Production alias does not match expected model version after reload",
                    "loaded_model_uri": model_uri,
                    "expected_model_version": str(expected_model_version),
                    "actual_production_version": str(production_version.version),
                },
            )
    return {"status": "reloaded", "model_uri": model_uri}
