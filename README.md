# Article Tagger Demo

This repository contains two connected layers:

- the existing ML core for training, model registration, drift detection, and raw inference
- a demo web application layer built around that ML service

The ML core uses a fine-tuned DistilBERT multi-label classifier for Medium article tagging. Training is tracked in MLflow with PostgreSQL, `data/medium_articles.csv` is versioned with DVC, and Airflow can retrain, evaluate, promote, and reload newer models when drift is detected.

The demo stack keeps responsibilities separate:

- `app.py` is the ML inference service
- `backend/` is the FastAPI application backend for authentication, status, and ML-service integration
- `frontend/` is the React + Vite + TypeScript single-page app

The retraining flow is:

```text
drift detection -> DAG trigger -> retrain -> evaluate -> compare -> promote -> reload serving model
```

## Supported Tags

30 labels including: `Blockchain`, `Data Science`, `Technology`, `Programming`, `Poetry`, `Machine Learning`, `Python`, `JavaScript`, and `Artificial Intelligence`.

## Requirements

- Python 3.12+
- `uv`
- Docker + Docker Compose
- DVC if you need to restore the dataset locally

## 1. Get the Dataset with DVC

The dataset pointer is tracked in Git as `data/medium_articles.csv.dvc` while the raw `data/medium_articles.csv` stays out of source control.

Install DVC:

```bash
uv tool install dvc
```

If the CSV is already available in your local DVC cache or configured remote, restore it with:

```bash
dvc pull data/medium_articles.csv.dvc
```

If you download a fresh copy from Kaggle, track it with:

```bash
dvc add data/medium_articles.csv
git add data/medium_articles.csv.dvc data/.gitignore
```

If you want other clones to be able to retrieve the dataset, configure a remote and push:

```bash
dvc remote add -d storage <remote-path>
dvc push
```

## 2. Start Services

### Simplest path

```bash
cd /Users/pt/project/SEML_PROJ && make full-stack
```

Then open [http://localhost:8001](http://localhost:8001).

The ML API retries the MLflow registry for about a minute. If the registry is empty because the Docker Postgres volume was reset, it automatically falls back to the local checkpoint at `distilbert_tagging_model/` and the demo still works.

This warning is normal in that situation:

```text
WARNING: mlflow_registry_empty falling_back_to_local path=/app/distilbert_tagging_model
```

To repopulate the MLflow registry without retraining, keep the stack running and execute:

```bash
cd /Users/pt/project/SEML_PROJ && make bootstrap-model
```

### Other start modes

```bash
cd /Users/pt/project/SEML_PROJ && make app-layer
cd /Users/pt/project/SEML_PROJ && make postgres
cd /Users/pt/project/SEML_PROJ && make backend
cd /Users/pt/project/SEML_PROJ && make frontend-install
cd /Users/pt/project/SEML_PROJ && make frontend
```

For local backend development with the ML service running too:

```bash
docker compose up --build postgres mlflow api
cd /Users/pt/project/SEML_PROJ && make backend
```

`docker compose up --build` also starts the full stack directly.

### Services and ports

| Service | URL | Description |
|---|---|---|
| `postgres` | `localhost:5435` | PostgreSQL for MLflow, Airflow, and the demo app database |
| `mlflow` | `http://localhost:5001` | Experiment tracking and model registry |
| `airflow-webserver` | `http://localhost:8080` | Airflow UI |
| `api` | `http://localhost:8002` | ML inference service |
| `web` | `http://localhost:8001` | App backend and built frontend |
| `frontend` dev server | `http://127.0.0.1:5173` | Vite development server |

Airflow login defaults:

- Username: `airflow`
- Password: `airflow`

## 3. Train the Model

If the dataset is not already in DVC storage, download the [Medium Articles dataset](https://www.kaggle.com/datasets/fabiochiusano/medium-articles) from Kaggle, place it at `data/medium_articles.csv`, and run `dvc add data/medium_articles.csv`.

Install the ML dependencies:

```bash
uv pip install -r requirements.txt
```

Run training after the dataset is present locally and Docker Compose is running:

```bash
uv run python train.py --data data/medium_articles.csv
```

Training will:

1. preprocess and split the dataset
2. fine-tune `distilbert-base-uncased` for multi-label classification
3. log params and metrics to MLflow
4. register the model as `DistilBERT_Tagger`
5. mark the registered version as the `challenger` alias

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to `data/medium_articles.csv` |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--model-dir` | `./distilbert_tagging_model` | Where to save model weights |

## 4. Airflow Retraining DAG

The DAG is defined in `dags/retraining_pipeline.py`.

It executes:

1. `detect_drift`
2. `retrain_model`
3. `evaluate_candidate`
4. `compare_and_promote`
5. `reload_serving_model`

Schedule:

- runs automatically every day at `00:00` Airflow scheduler time
- `catchup=False`, so it only runs forward and does not backfill old dates

Drift detection checks:

- label distribution drift in recent predictions
- out-of-vocabulary spike versus the rolling baseline
- tag acceptance rate drop versus the rolling baseline

The DAG short-circuits unless at least one signal is triggered.

Default paths in Docker:

- `REFERENCE_DATA_PATH=/opt/airflow/project/data/medium_articles.csv`
- `CURRENT_DATA_PATH=/opt/airflow/project/data/medium_articles.csv`
- `REFERENCE_PREDICTIONS_PATH=/opt/airflow/project/data/reference_predictions.csv`
- `CURRENT_PREDICTIONS_PATH=/opt/airflow/project/data/current_predictions.csv`
- `OOV_HISTORY_PATH=/opt/airflow/project/data/oov_history.csv`
- `CTR_HISTORY_PATH=/opt/airflow/project/data/ctr_history.csv`
- `CURRENT_FEEDBACK_PATH=/opt/airflow/project/data/current_feedback.csv`
- `DRIFT_REPORT_DIR=/opt/airflow/project/reports`
- `MODEL_RELOAD_URL=http://api:8000/admin/reload-model`

Trigger manually from the Airflow container:

```bash
airflow dags trigger article_tagger_retraining
```

Example with per-run overrides:

```bash
airflow dags trigger article_tagger_retraining --conf '{"current_data_path":"/opt/airflow/project/data/medium_articles.csv","current_prediction_path":"/opt/airflow/project/data/current_predictions.csv","oov_history_path":"/opt/airflow/project/data/oov_history.csv","ctr_history_path":"/opt/airflow/project/data/ctr_history.csv","current_feedback_path":"/opt/airflow/project/data/current_feedback.csv","label_frequency_threshold":0.15,"oov_sigma_threshold":2.0,"ctr_drop_threshold":0.20,"epochs":1,"metric_name":"micro_f1"}'
```

## 5. Promotion and Serving

Candidate models are compared against the current production model using `micro_f1`.

- If no production model exists yet, the candidate is promoted.
- If the candidate beats production, it is assigned the `production` alias.
- The previous production model is moved to `previous-production`.
- Every newly trained model is marked as `challenger`.

The ML service tries to load `models:/DistilBERT_Tagger@production` first, then `models:/DistilBERT_Tagger/latest`, and finally the local checkpoint directory if the registry is empty.

When Airflow promotes a candidate, it calls the ML service reload endpoint so serving picks up the new model without waiting for a process restart.

## 6. Demo Web Application

The web application layer is demo-focused and intentionally minimal.

### Routes

- `/signup`
- `/login`
- `/demo`
- `/status`

### Backend responsibilities

The FastAPI app under `backend/` handles:

- sign-up, login, logout, and signed session-cookie auth
- protected access control
- prediction proxying to the ML service
- request logging and per-user request history
- health and status endpoints
- Airflow DAG triggering when the OOV detector sees a spike

### Frontend responsibilities

The React app under `frontend/` provides:

- authentication screens
- a protected single-article demo flow
- tag-only prediction output
- a protected system status page

The built frontend is served by the backend so the browser talks to one same-origin app on port `8001`.

### Useful commands

```bash
cd /Users/pt/project/SEML_PROJ && make backend-test
cd /Users/pt/project/SEML_PROJ && make frontend-build
cd /Users/pt/project/SEML_PROJ && make health
cd /Users/pt/project/SEML_PROJ && make app-layer-down
docker compose down
```

## 7. Call the ML Service API

The raw prediction service remains separate from the app backend.

### `POST /predict`

Single input:

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Python is a popular language for machine learning."}'
```

Batch input:

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Python is great for ML.", "Bitcoin is leading the crypto rally."]}'
```

Response:

```json
{
  "predictions": [
    ["Python", "Machine Learning"],
    ["Blockchain", "Cryptocurrency"]
  ]
}
```

Tags with confidence score above `0.3` are returned.

### `POST /admin/reload-model`

This internal endpoint reloads the serving model after Airflow promotes a candidate.

It requires the `X-Model-Reload-Token` header when `MODEL_RELOAD_TOKEN` is configured.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow tracking server URI |
| `MODEL_URI` | `models:/DistilBERT_Tagger@production` | Primary MLflow model URI |
| `MODEL_FALLBACK_URI` | `models:/DistilBERT_Tagger/latest` | Fallback model URI |
| `LOCAL_MODEL_PATH` | `./distilbert_tagging_model` | Local checkpoint used when the registry is empty |
| `MODEL_RELOAD_TOKEN` | unset | Shared token for model reload requests |

## MLflow UI

View experiments and registered models at [http://localhost:5001](http://localhost:5001) after starting the stack.
