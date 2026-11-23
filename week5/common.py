"""Shared data preparation, evaluation, and output handling."""
import html
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

WORD_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"</?[a-zA-Z][\w:-]*(?=[\s/>])[^<>]*>", " ", text)
    text = re.sub(r"\b(?:https?://|www\.)[^\s<>]+", " ", text.lower())
    import unicodedata
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def tokenize(text):
    return WORD_PATTERN.findall(text.translate(str.maketrans("\u2019\u2010\u2011", "'--")))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_data(name, seed=42, quick=False):
    dataset_id = {"imdb": "stanfordnlp/imdb", "banking77": "PolyAI/banking77"}[name]
    raw = load_dataset(dataset_id)
    frame = raw["train"].to_pandas()
    train, validation = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=frame["label"]
    )
    frames = {"train": train, "validation": validation, "test": raw["test"].to_pandas()}
    # Every class remains represented in the smoke run.
    if quick:
        per_class = {"train": 8 if name == "banking77" else 128,
                     "validation": 3 if name == "banking77" else 32,
                     "test": 3 if name == "banking77" else 32}
        frames = {
            split: data.groupby("label", group_keys=False).sample(
                n=per_class[split], random_state=seed
            )
            for split, data in frames.items()
        }
    for split, data in frames.items():
        data = data.copy().reset_index(drop=True)
        data["clean_text"] = data["text"].map(clean_text)
        frames[split] = data
    labels = raw["train"].features["label"].names
    metadata = {
        "dataset": name, "dataset_id": dataset_id, "classes": len(labels),
        "label_names": labels, "seed": seed, "quick": quick,
        "split_sizes": {key: len(value) for key, value in frames.items()},
        "fingerprints": {key: raw[key]._fingerprint for key in ("train", "test")},
    }
    return frames, metadata


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    values = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
    if probabilities.shape[1] == 2:
        values["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    return values


def save_evaluation(folder, frames, probabilities, label_names):
    scores = {}
    for split in ("validation", "test"):
        labels = frames[split]["label"].to_numpy()
        probs = probabilities[split]
        scores[split] = metrics(labels, probs)
        predictions = probs.argmax(axis=1)
        pd.DataFrame({
            "text": frames[split]["text"], "true_label": labels,
            "predicted_label": predictions, "confidence": probs.max(axis=1),
        }).to_csv(folder / f"{split}_predictions.csv", index=False)
        save_json(folder / f"{split}_classification_report.json", classification_report(
            labels, predictions, labels=list(range(len(label_names))),
            target_names=label_names, output_dict=True, zero_division=0,
        ))
    return scores
