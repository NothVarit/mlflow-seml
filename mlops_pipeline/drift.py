import ast
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import load_dataset_frame

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
TAG_COLUMNS = ("predicted_tags", "prediction", "predictions", "tags")
TEXT_COLUMNS = ("text", "combined_text", "title", "input_text", "content")
CTR_COLUMNS = ("ctr", "acceptance_rate", "user_acceptance_rate")
DATE_COLUMNS = ("date", "ds", "timestamp", "event_date")
OOV_COLUMNS = ("oov_rate", "unseen_token_rate", "unseen_word_rate")


def _load_table(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if "rows" in payload and isinstance(payload["rows"], list):
                return pd.DataFrame(payload["rows"])
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported file format for drift input: {path}")


def _find_first_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Expected one of {candidates}, found columns: {list(df.columns)}")


def _parse_tag_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return []


def _normalize_rate_series(history_path, rate_columns, window_days):
    history_df = _load_table(history_path)
    date_column = _find_first_column(history_df, DATE_COLUMNS)
    rate_column = _find_first_column(history_df, rate_columns)

    history_df = history_df.copy()
    history_df[date_column] = pd.to_datetime(history_df[date_column], errors="coerce")
    history_df = history_df.dropna(subset=[date_column, rate_column]).sort_values(date_column)
    history_df = history_df.tail(window_days)

    if history_df.empty:
        raise ValueError(f"No valid rate history rows found in {history_path}")

    return history_df[rate_column].astype(float).to_numpy()


def _baseline_vocabulary(reference_path):
    reference_df = load_dataset_frame(reference_path)
    vocabulary = set()
    for text in reference_df["combined_text"].astype(str):
        vocabulary.update(token.lower() for token in TOKEN_PATTERN.findall(text))
    if not vocabulary:
        raise ValueError(f"Reference vocabulary is empty for {reference_path}")
    return vocabulary


def _current_texts(current_path):
    current_df = load_dataset_frame(current_path)
    return current_df["combined_text"].astype(str).tolist(), int(len(current_df))


def _top_tag_distribution(path, top_k):
    df = _load_table(path)
    tag_column = _find_first_column(df, TAG_COLUMNS)

    counts = {}
    for tags in df[tag_column].apply(_parse_tag_list):
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1

    if not counts:
        raise ValueError(f"No tags found in {path}")

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_counts = dict(sorted_counts[:top_k])
    total = sum(top_counts.values())
    return {tag: count / total for tag, count in top_counts.items()}


def _label_distribution_drift(reference_prediction_path, current_prediction_path, top_k, threshold):
    baseline_dist = _top_tag_distribution(reference_prediction_path, top_k=top_k)
    current_dist = _top_tag_distribution(current_prediction_path, top_k=top_k)
    candidate_tags = sorted(set(baseline_dist) | set(current_dist))

    deltas = [
        {
            "tag": tag,
            "baseline_frequency": float(baseline_dist.get(tag, 0.0)),
            "current_frequency": float(current_dist.get(tag, 0.0)),
            "absolute_delta": float(abs(current_dist.get(tag, 0.0) - baseline_dist.get(tag, 0.0))),
        }
        for tag in candidate_tags
    ]
    max_delta = max(item["absolute_delta"] for item in deltas)
    triggered = max_delta > threshold

    return {
        "triggered": triggered,
        "threshold": threshold,
        "max_frequency_delta": float(max_delta),
        "top_tag_changes": sorted(deltas, key=lambda item: item["absolute_delta"], reverse=True)[:10],
    }


def _oov_spike(reference_data_path, current_data_path, oov_history_path, sigma_threshold):
    vocabulary = _baseline_vocabulary(reference_data_path)
    current_texts, current_rows = _current_texts(current_data_path)

    total_tokens = 0
    unseen_tokens = 0
    for text in current_texts:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        total_tokens += len(tokens)
        unseen_tokens += sum(1 for token in tokens if token not in vocabulary)

    current_oov_rate = float(unseen_tokens / max(total_tokens, 1))
    history = _normalize_rate_series(oov_history_path, OOV_COLUMNS, window_days=14)
    baseline_mean = float(history.mean())
    baseline_std = float(history.std(ddof=0))
    threshold = baseline_mean + (sigma_threshold * baseline_std)
    triggered = current_oov_rate > threshold

    return {
        "triggered": triggered,
        "current_oov_rate": current_oov_rate,
        "baseline_mean_oov_rate": baseline_mean,
        "baseline_std_oov_rate": baseline_std,
        "threshold": float(threshold),
        "reference_vocabulary_size": len(vocabulary),
        "current_rows": current_rows,
    }


def _extract_ctr_from_current_feedback(current_feedback_path):
    feedback_df = _load_table(current_feedback_path)
    ctr_column = _find_first_column(feedback_df, CTR_COLUMNS)
    ctr_values = feedback_df[ctr_column].dropna().astype(float)
    if ctr_values.empty:
        raise ValueError(f"No valid CTR values found in {current_feedback_path}")
    return float(ctr_values.iloc[-1])


def _tag_acceptance_drop(ctr_history_path, current_feedback_path, drop_threshold):
    history = _normalize_rate_series(ctr_history_path, CTR_COLUMNS, window_days=7)
    rolling_average = float(history.mean())
    current_ctr = _extract_ctr_from_current_feedback(current_feedback_path)
    drop = float(rolling_average - current_ctr)
    triggered = drop > drop_threshold

    return {
        "triggered": triggered,
        "current_ctr": current_ctr,
        "rolling_average_ctr": rolling_average,
        "absolute_drop": drop,
        "threshold": drop_threshold,
    }


def detect_drift(
    reference_path,
    current_path,
    report_path=None,
    reference_prediction_path=None,
    current_prediction_path=None,
    oov_history_path=None,
    ctr_history_path=None,
    current_feedback_path=None,
    label_frequency_threshold=0.15,
    oov_sigma_threshold=2.0,
    ctr_drop_threshold=0.20,
    top_k_tags=50,
):
    signals = {}

    if reference_prediction_path and current_prediction_path:
        signals["label_distribution_drift"] = _label_distribution_drift(
            reference_prediction_path=reference_prediction_path,
            current_prediction_path=current_prediction_path,
            top_k=top_k_tags,
            threshold=label_frequency_threshold,
        )

    if oov_history_path:
        signals["oov_spike"] = _oov_spike(
            reference_data_path=reference_path,
            current_data_path=current_path,
            oov_history_path=oov_history_path,
            sigma_threshold=oov_sigma_threshold,
        )

    if ctr_history_path and current_feedback_path:
        signals["tag_acceptance_drop"] = _tag_acceptance_drop(
            ctr_history_path=ctr_history_path,
            current_feedback_path=current_feedback_path,
            drop_threshold=ctr_drop_threshold,
        )

    if not signals:
        raise ValueError(
            "No drift signals configured. Provide prediction paths, OOV history, or CTR history/current feedback paths."
        )

    drift_detected = any(signal["triggered"] for signal in signals.values())
    report = {
        "reference_path": reference_path,
        "current_path": current_path,
        "reference_prediction_path": reference_prediction_path,
        "current_prediction_path": current_prediction_path,
        "oov_history_path": oov_history_path,
        "ctr_history_path": ctr_history_path,
        "current_feedback_path": current_feedback_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drift_detected": drift_detected,
        "signals": signals,
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

    return report
