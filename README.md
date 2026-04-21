# Article Tagger - DistilBERT Multi-Label Classification

A FastAPI service that predicts topic tags for Medium articles using a fine-tuned DistilBERT model. Training experiments are tracked with MLflow backed by PostgreSQL, and `data/medium_articles.csv` is versioned with DVC.

## Supported Tags

30 labels including: `Blockchain`, `Data Science`, `Technology`, `Programming`, `Poetry`, `Machine Learning`, `Python`, `JavaScript`, `Artificial Intelligence`, and more.

---

## Requirements

- Python 3.11+
- Docker + Docker Compose

---

## Workflow

```text
1. Pull dataset with DVC -> 2. Start Docker Compose -> 3. Train model -> 4. Call the API
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

This starts three containers:

| Container | URL | Description |
|-----------|-----|-------------|
| `postgres` | `localhost:5435` | Metadata backend for MLflow |
| `mlflow` | `http://localhost:5001` | Experiment tracking & model registry |
| `api` | `http://localhost:8000` | FastAPI prediction endpoint |

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
4. Register the model as `DistilBERT_Tagger` in the MLflow Model Registry

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to `data/medium_articles.csv` |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--model-dir` | `./distilbert_tagging_model` | Where to save model weights |

> **Tip:** Training on CPU takes about 3 hours. For faster results, train on [Kaggle](https://www.kaggle.com) or Google Colab and set `MLFLOW_TRACKING_URI=http://localhost:5001` before running.

---

## 4. Call the API

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
