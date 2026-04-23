# Article Tagger - DistilBERT Multi-Label Classification

A FastAPI service that predicts topic tags for Medium articles using a fine-tuned DistilBERT model. Training experiments are tracked with MLflow backed by PostgreSQL, and `data/medium_articles.csv` is versioned with DVC.

The project now also includes an Airflow retraining DAG for this flow:

```text
drift detection -> DAG trigger -> retrain -> evaluate -> compare -> promote
```

## Supported Tags

30 labels including: `Blockchain`, `Data Science`, `Technology`, `Programming`, `Poetry`, `Machine Learning`, `Python`, `JavaScript`, `Artificial Intelligence`, and more.

---

## Requirements

- Python 3.11+
- Docker + Docker Compose

---

## Workflow

```text
1. Pull dataset with DVC -> 2. Start Docker Compose -> 3. Train model or trigger Airflow DAG -> 4. Call the API
```

---

## 1. Get the Dataset with DVC

The dataset pointer is tracked in Git as `data/medium_articles.csv.dvc` while the raw `data/medium_articles.csv` stays out of source control.

Install DVC:

```bash
pip install dvc
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

---

## 2. Start Services

```bash
docker compose up --build
```

This starts these containers:

| Container | URL | Description |
|-----------|-----|-------------|
| `postgres` | `localhost:5435` | Metadata backend for MLflow and Airflow |
| `mlflow` | `http://localhost:5001` | Experiment tracking and model registry |
| `airflow-webserver` | `http://localhost:8080` | Airflow UI |
| `api` | `http://localhost:8000` | FastAPI prediction endpoint |

Airflow login defaults:

- Username: `airflow`
- Password: `airflow`

---

## 3. Train the Model

If you do not already have the dataset in DVC storage, download the [Medium Articles dataset](https://www.kaggle.com/datasets/fabiochiusano/medium-articles) from Kaggle, place it at `data/medium_articles.csv`, and run `dvc add data/medium_articles.csv`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training after the dataset is present locally and Docker Compose is running:

```bash
python train.py --data data/medium_articles.csv
```

Training will:

1. Preprocess and split the dataset (80/20 train/test)
2. Fine-tune `distilbert-base-uncased` for multi-label classification
3. Log params and metrics (micro F1, macro F1, Precision@5) to MLflow
4. Register the model as `DistilBERT_Tagger`
5. Mark the registered version as the `challenger` alias

Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to `data/medium_articles.csv` |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--model-dir` | `./distilbert_tagging_model` | Where to save model weights |

> Training on CPU takes about 3 hours. For faster results, train on Kaggle or Colab and set `MLFLOW_TRACKING_URI=http://localhost:5001` before running.

---

## 4. Airflow DAG

The DAG is defined in `dags/retraining_pipeline.py`.

It executes:

1. `detect_drift`
2. `retrain_model`
3. `evaluate_candidate`
4. `compare_and_promote`

Schedule:

- runs automatically every day at `00:00` Airflow scheduler time
- `catchup=False`, so it only runs forward and does not backfill old dates

Drift detection is code-based and now checks:

- `Label Distribution Drift`: shift in the Top-50 predicted tags frequency exceeds `15` percentage points versus the baseline prediction distribution.
- `Out-of-Vocabulary (OOV) Spike`: unseen token rate in current input text exceeds the 14-day rolling baseline mean by more than `2 sigma`.
- `Tag Acceptance Drop`: current CTR or user tag acceptance rate drops by more than `20` percentage points from the 7-day rolling average.

The DAG short-circuits unless at least one of those signals is triggered.

Default paths in Docker:

- `REFERENCE_DATA_PATH=/opt/airflow/project/data/medium_articles.csv`
- `CURRENT_DATA_PATH=/opt/airflow/project/data/medium_articles.csv`
- `REFERENCE_PREDICTIONS_PATH=/opt/airflow/project/data/reference_predictions.csv`
- `CURRENT_PREDICTIONS_PATH=/opt/airflow/project/data/current_predictions.csv`
- `OOV_HISTORY_PATH=/opt/airflow/project/data/oov_history.csv`
- `CTR_HISTORY_PATH=/opt/airflow/project/data/ctr_history.csv`
- `CURRENT_FEEDBACK_PATH=/opt/airflow/project/data/current_feedback.csv`
- `DRIFT_REPORT_DIR=/opt/airflow/project/reports`

Expected drift input shapes:

- prediction files: CSV/JSON/JSONL with `predicted_tags`, `prediction`, `predictions`, or `tags`
- OOV history: CSV/JSON/JSONL with date column plus `oov_rate`, `unseen_token_rate`, or `unseen_word_rate`
- CTR history/current feedback: CSV/JSON/JSONL with date column plus `ctr`, `acceptance_rate`, or `user_acceptance_rate`

For a real retraining loop, point `CURRENT_DATA_PATH` at fresh production text, `CURRENT_PREDICTIONS_PATH` at recent model outputs, and `CURRENT_FEEDBACK_PATH` at the latest user feedback aggregate before triggering the DAG.

Trigger manually from the Airflow container:

```bash
airflow dags trigger article_tagger_retraining
```

Example with per-run overrides:

```bash
airflow dags trigger article_tagger_retraining --conf '{"current_data_path":"/opt/airflow/project/data/medium_articles.csv","current_prediction_path":"/opt/airflow/project/data/current_predictions.csv","oov_history_path":"/opt/airflow/project/data/oov_history.csv","ctr_history_path":"/opt/airflow/project/data/ctr_history.csv","current_feedback_path":"/opt/airflow/project/data/current_feedback.csv","label_frequency_threshold":0.15,"oov_sigma_threshold":2.0,"ctr_drop_threshold":0.20,"epochs":1,"metric_name":"micro_f1"}'
```

---

## 5. Promotion Logic

Candidate models are compared against the current production model using `micro_f1`.

- If no production model exists yet, the candidate is promoted.
- If the candidate beats production, it is assigned the `production` alias.
- The previous production model is moved to `previous-production`.
- Every newly trained model is marked as `challenger`.

The API now tries to load `models:/DistilBERT_Tagger@production` first and falls back to `models:/DistilBERT_Tagger/latest` if no production alias exists yet.

---

## 6. Call the API

### `POST /predict`

Single input:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Python is a popular language for machine learning."}'
```

Batch input:

```bash
curl -X POST http://localhost:8000/predict \
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

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow tracking server URI |
| `MODEL_URI` | `models:/DistilBERT_Tagger@production` | Primary MLflow model URI |
| `MODEL_FALLBACK_URI` | `models:/DistilBERT_Tagger/latest` | Fallback model URI |

---

## MLflow UI

View experiments and registered models at [http://localhost:5001](http://localhost:5001) after running `docker compose up`.
