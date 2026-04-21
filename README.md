# Article Tagger — DistilBERT Multi-Label Classification

A FastAPI service that predicts topic tags for Medium articles using a fine-tuned DistilBERT model. Training experiments are tracked with MLflow backed by PostgreSQL.

## Supported Tags

30 labels including: `Blockchain`, `Data Science`, `Technology`, `Programming`, `Poetry`, `Machine Learning`, `Python`, `JavaScript`, `Artificial Intelligence`, and more.

---

## Requirements

- Python 3.11+
- Docker + Docker Compose

---

## Workflow

```
1. Start Docker Compose  →  2. Train model  →  3. Call the API
```

---

## 1. Start Services

```bash
docker compose up --build
```

This starts three containers:

| Container | URL | Description |
|-----------|-----|-------------|
| `postgres` | `localhost:5435` | Metadata backend for MLflow |
| `mlflow` | `http://localhost:5001` | Experiment tracking & model registry |
| `api` | `http://localhost:8000` | FastAPI prediction endpoint |

---

## 2. Train the Model

Download the [Medium Articles dataset](https://www.kaggle.com/datasets/fabiochiusano/medium-articles) from Kaggle and place `medium_articles.csv` in the project folder.

Install dependencies:
```bash
pip install -r requirements.txt
```

Run training (make sure Docker Compose is running first):
```bash
python train.py --data medium_articles.csv
```

Training will:
1. Preprocess and split the dataset (80/20 train/test)
2. Fine-tune `distilbert-base-uncased` for multi-label classification
3. Log params and metrics (micro F1, macro F1, Precision@5) to MLflow
4. Register the model as `DistilBERT_Tagger` in the MLflow Model Registry

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to `medium_articles.csv` |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--model-dir` | `./distilbert_tagging_model` | Where to save model weights |

> **Tip:** Training on CPU takes ~3 hours. For faster results, train on [Kaggle](https://www.kaggle.com) or Google Colab (free GPU) and set `MLFLOW_TRACKING_URI=http://localhost:5001` before running.

---

## 3. Call the API

### `POST /predict`

**Single input:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Python is a popular language for machine learning."}'
```

**Batch input:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Python is great for ML.", "Bitcoin is leading the crypto rally."]}'
```

**Response:**
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
| `MODEL_URI` | `models:/DistilBERT_Tagger/latest` | MLflow model URI to load |

---

## MLflow UI

View experiments and registered models at [http://localhost:5001](http://localhost:5001) after running `docker compose up`.
