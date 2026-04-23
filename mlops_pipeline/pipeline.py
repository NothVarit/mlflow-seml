import gc
import json
import os
import time
from tempfile import TemporaryDirectory

import mlflow
import mlflow.transformers
import numpy as np
import torch
import transformers
from datasets import Dataset
from mlflow import MlflowClient
from scipy.special import expit
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .constants import ID2LABEL, LABEL2ID, MODEL_NAME, SELECTED_LABELS, TARGET_TAGS
from .data import load_and_preprocess


def precision_at_k(y_true, y_score, k=5):
    top_k_idx = np.argsort(y_score, axis=1)[:, -k:]
    relevant = sum(np.sum(y_true[i, top_k_idx[i]]) for i in range(y_true.shape[0]))
    return relevant / (k * y_true.shape[0])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_at_5": float(precision_at_k(labels, probs, k=5)),
    }


def _tokenize_dataset(tokenizer, texts, labels):
    dataset = Dataset.from_dict({"text": texts.tolist(), "labels": labels.astype(float).tolist()})
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
        batched=True,
        remove_columns=["text"],
    )


def _set_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Article_Tagging")
    return tracking_uri


def _find_model_version(client, run_id):
    for _ in range(10):
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        matching_versions = [version for version in versions if version.run_id == run_id]
        if matching_versions:
            return max(matching_versions, key=lambda item: int(item.version))
        time.sleep(2)
    raise RuntimeError(f"Could not find registered model version for run_id={run_id}")


def _set_model_alias(client, version, alias):
    try:
        client.set_registered_model_alias(MODEL_NAME, alias, version)
    except Exception:
        if alias == "production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
        elif alias == "challenger":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
        elif alias == "previous-production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Archived",
                archive_existing_versions=False,
            )


def _get_model_version_for_alias(client, alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        stage_name = {
            "production": "Production",
            "challenger": "Staging",
            "previous-production": "Archived",
        }.get(alias)
        if not stage_name:
            return None
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage_name])
        return versions[0] if versions else None


def _get_model_metric(client, version, metric_name):
    run = client.get_run(version.run_id)
    metric_value = run.data.metrics.get(metric_name)
    return None if metric_value is None else float(metric_value)


def train_and_register_model(
    csv_path,
    epochs=2,
    batch_size=16,
    lr=5e-5,
    model_dir="./distilbert_tagging_model",
    run_name="DistilBERT",
):
    print("Loading and preprocessing data...")
    x_values, y_values = load_and_preprocess(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = _tokenize_dataset(tokenizer, x_train, y_train)
    test_dataset = _tokenize_dataset(tokenizer, x_test, y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(SELECTED_LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    tracking_uri = _set_tracking()

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "model": "distilbert-base-uncased",
                "num_labels": len(SELECTED_LABELS),
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_length": 128,
                "weight_decay": 0.01,
                "training_data_path": csv_path,
            }
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            report_to="mlflow",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        normalized_metrics = {
            "micro_f1": float(eval_metrics["eval_micro_f1"]),
            "macro_f1": float(eval_metrics["eval_macro_f1"]),
            "precision_at_5": float(eval_metrics["eval_precision_at_5"]),
        }
        mlflow.log_metrics(normalized_metrics)

        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        pipeline = transformers.pipeline(
            task="text-classification",
            model=model_dir,
            tokenizer=model_dir,
            top_k=None,
        )
        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="distilbert_tagger",
            registered_model_name=MODEL_NAME,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        saved_model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
        saved_model.eval()
        saved_tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        data_collator = DataCollatorWithPadding(tokenizer=saved_tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                logits = saved_model(**inputs).logits
                all_probs.append(torch.sigmoid(logits).cpu().numpy())

        y_score = np.vstack(all_probs)
        y_pred = (y_score >= 0.5).astype(int)
        target_indices = [SELECTED_LABELS.index(tag) for tag in TARGET_TAGS]

        target_tag_precision = {}
        for tag, idx in zip(TARGET_TAGS, target_indices):
            target_tag_precision[tag] = float(precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0))

        combined_precision = float(
            precision_score(
                y_test[:, target_indices],
                y_pred[:, target_indices],
                average="micro",
                zero_division=0,
            )
        )

        evaluation_summary = {
            "metrics": normalized_metrics,
            "target_tag_precision": target_tag_precision,
            "combined_target_micro_precision": combined_precision,
            "tracking_uri": tracking_uri,
            "run_id": active_run.info.run_id,
        }

        with TemporaryDirectory() as tmp_dir:
            evaluation_report_path = os.path.join(tmp_dir, "evaluation_summary.json")
            with open(evaluation_report_path, "w", encoding="utf-8") as report_file:
                json.dump(evaluation_summary, report_file, indent=2)
            mlflow.log_artifact(evaluation_report_path, artifact_path="reports")

    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = _find_model_version(client, evaluation_summary["run_id"])
    _set_model_alias(client, model_version.version, "challenger")

    evaluation_summary["model_name"] = MODEL_NAME
    evaluation_summary["model_version"] = model_version.version
    evaluation_summary["model_uri"] = f"models:/{MODEL_NAME}/{model_version.version}"
    return evaluation_summary


def compare_and_promote_model(candidate_version, metric_name="micro_f1"):
    tracking_uri = _set_tracking()
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate = client.get_model_version(MODEL_NAME, str(candidate_version))
    candidate_metric = _get_model_metric(client, candidate, metric_name)
    if candidate_metric is None:
        raise RuntimeError(f"Candidate version {candidate_version} is missing metric '{metric_name}'")

    current_production = _get_model_version_for_alias(client, "production")
    current_metric = None if current_production is None else _get_model_metric(client, current_production, metric_name)

    promoted = current_production is None or current_metric is None or candidate_metric > current_metric
    if promoted:
        if current_production is not None and str(current_production.version) != str(candidate.version):
            _set_model_alias(client, current_production.version, "previous-production")
        _set_model_alias(client, candidate.version, "production")

    return {
        "tracking_uri": tracking_uri,
        "model_name": MODEL_NAME,
        "candidate_version": str(candidate.version),
        "candidate_run_id": candidate.run_id,
        "metric_name": metric_name,
        "candidate_metric": candidate_metric,
        "previous_production_version": None if current_production is None else str(current_production.version),
        "previous_production_metric": current_metric,
        "promoted": promoted,
    }
