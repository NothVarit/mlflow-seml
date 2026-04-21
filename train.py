import argparse
import ast
import gc
import os
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.transformers
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
from scipy.special import expit
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

SELECTED_LABELS = [
    "Blockchain", "Data Science", "Technology", "Programming", "Poetry",
    "Cryptocurrency", "Machine Learning", "Life", "Writing", "Politics",
    "Startup", "Life Lessons", "Self Improvement", "Covid 19", "Software Development",
    "Love", "Python", "Business", "Health", "Mental Health",
    "JavaScript", "Relationships", "Education", "Artificial Intelligence",
    "Culture", "Design", "Self", "Marketing", "Entrepreneurship", "Personal Development"
]

TAG_TO_IDX = {tag: idx for idx, tag in enumerate(SELECTED_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(SELECTED_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(SELECTED_LABELS)}

TARGET_TAGS = ["Blockchain", "Data Science", "Technology", "Programming", "Poetry"]


def precision_at_k(y_true, y_score, k=5):
    top_k_idx = np.argsort(y_score, axis=1)[:, -k:]
    relevant = sum(np.sum(y_true[i, top_k_idx[i]]) for i in range(y_true.shape[0]))
    return relevant / (k * y_true.shape[0])


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    def parse_tags(tag_string):
        try:
            return ast.literal_eval(tag_string)
        except Exception:
            return []

    def encode(tags):
        vec = np.zeros(len(SELECTED_LABELS), dtype=int)
        for tag in tags:
            if tag in TAG_TO_IDX:
                vec[TAG_TO_IDX[tag]] = 1
        return vec

    df["tag_list"] = df["tags"].apply(parse_tags)
    df["filtered_tag_list"] = df["tag_list"].apply(lambda tags: [t for t in tags if t in TAG_TO_IDX])
    df = df[df["filtered_tag_list"].apply(len) > 0].copy()
    df["label_vector"] = df["filtered_tag_list"].apply(encode)
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X = df["combined_text"].values
    Y = np.stack(df["label_vector"].values)
    return X, Y


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_at_5": float(precision_at_k(labels, probs, k=5)),
    }


def train_distilbert(X_train, X_test, y_train, y_test, epochs, batch_size, lr, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Training Model B: DistilBERT with Micro and Macro Metrics ---")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(texts, labels):
        ds = Dataset.from_dict({"text": texts.tolist(), "labels": labels.astype(float).tolist()})
        return ds.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128),
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = tokenize(X_train, y_train)
    test_dataset = tokenize(X_test, y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(SELECTED_LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Article_Tagging")

    with mlflow.start_run(run_name="DistilBERT"):
        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "num_labels": len(SELECTED_LABELS),
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": 128,
            "weight_decay": 0.01,
        })

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
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        pipeline = transformers.pipeline(
            task="text-classification",
            model=model_dir,
            tokenizer=model_dir,
            top_k=None,
        )
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="distilbert_tagger",
            registered_model_name="DistilBERT_Tagger",
        )
        print(f"Model registered in MLflow: {model_info.model_uri}")

    # Batch inference for specific tag precision
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Running Batch Inference (Memory Efficient) ---")
    saved_model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    saved_tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    saved_model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=saved_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = saved_model(**inputs).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            del inputs, logits

    y_score = np.vstack(all_probs)
    y_pred = (y_score >= 0.5).astype(int)

    target_indices = [SELECTED_LABELS.index(tag) for tag in TARGET_TAGS]
    print("\n--- DistilBERT: Top 5 Tags Precision ---")
    for tag, idx in zip(TARGET_TAGS, target_indices):
        score = precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        print(f"{tag}: {score:.4f}")

    combined_p = precision_score(y_test[:, target_indices], y_pred[:, target_indices], average="micro", zero_division=0)
    print("-" * 35)
    print(f"Combined Micro-Precision: {combined_p:.4f}")


def train(csv_path, epochs=2, batch_size=16, lr=5e-5, model_dir="./distilbert_tagging_model"):
    print("Loading and preprocessing data...")
    X, Y = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    train_distilbert(X_train, X_test, y_train, y_test, epochs, batch_size, lr, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train article tagger (Logistic, SVM, DistilBERT)")
    parser.add_argument("--data", required=True, help="Path to medium_articles.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-dir", default="./distilbert_tagging_model")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.model_dir)
