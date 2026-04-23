import ast

import numpy as np
import pandas as pd

from .constants import TAG_TO_IDX


def _parse_tags(tag_string):
    try:
        return ast.literal_eval(tag_string)
    except Exception:
        return []


def _encode(tags):
    vec = np.zeros(len(TAG_TO_IDX), dtype=int)
    for tag in tags:
        if tag in TAG_TO_IDX:
            vec[TAG_TO_IDX[tag]] = 1
    return vec


def load_dataset_frame(csv_path):
    df = pd.read_csv(csv_path)
    df["tag_list"] = df["tags"].apply(_parse_tags)
    df["filtered_tag_list"] = df["tag_list"].apply(lambda tags: [tag for tag in tags if tag in TAG_TO_IDX])
    df = df[df["filtered_tag_list"].apply(len) > 0].copy()
    df["label_vector"] = df["filtered_tag_list"].apply(_encode)
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df


def load_and_preprocess(csv_path):
    df = load_dataset_frame(csv_path)
    x_values = df["combined_text"].values
    y_values = np.stack(df["label_vector"].values)
    return x_values, y_values
