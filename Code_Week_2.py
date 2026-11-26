"""Week 5: IMDb, BANKING77, and BERT fine-tuning. Saves only PNG figures."""
import argparse
import html
import math
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
# Reuse existing downloads when available. Otherwise use Hugging Face's default cache.
if (ROOT / ".cache" / "huggingface").exists():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
BATCH_SIZE = 64
MAX_LENGTH = 128
LEARNING_RATE = 5e-4
CHECKPOINT = "prajjwal1/bert-tiny"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def clean_text(text):
    text = html.unescape(text).lower()
    text = re.sub(r"</?[a-zA-Z][\w:-]*(?=[\s/>])[^<>]*>", " ", text)
    text = re.sub(r"\b(?:https?://|www\.)[^\s<>]+", " ", text)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def tokenize(text):
    text = text.translate(str.maketrans("\u2019\u2010\u2011", "'--"))
    return re.findall(r"[^\W_]+(?:['-][^\W_]+)*", text)


def save_plot(figure, name, quick=False):
    FIGURES.mkdir(exist_ok=True)
    prefix = "quick_" if quick else ""
    figure.savefig(FIGURES / f"{prefix}{name}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def prepare_data(name, quick):
    dataset_id = {"imdb": "stanfordnlp/imdb", "banking77": "PolyAI/banking77"}[name]
    raw = load_dataset(dataset_id)
    frame = raw["train"].to_pandas()
    train, validation = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame["label"],
    )
    frames = {"train": train, "validation": validation, "test": raw["test"].to_pandas()}
    for split, frame in frames.items():
        if quick:
            samples = {"train": 8, "validation": 3, "test": 3}
            if name == "imdb":
                samples = {"train": 128, "validation": 32, "test": 32}
            frame = frame.groupby("label", group_keys=False).sample(
                n=samples[split], random_state=SEED,
            )
        frame = frame.copy().reset_index(drop=True)
        frame["clean_text"] = frame["text"].map(clean_text)
        frames[split] = frame
    print(name, {split: len(frame) for split, frame in frames.items()}, flush=True)
    return frames, raw["train"].features["label"].names


def plot_dataset(frames, labels, name, quick):
    train = frames["train"]
    counts = train["label"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.16)), layout="constrained")
    axis.barh(range(len(labels)), counts.reindex(range(len(labels)), fill_value=0))
    axis.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels], fontsize=9)
    axis.invert_yaxis()
    axis.set(xlabel="Training examples", title=f"{name} training class counts")
    save_plot(figure, f"{name}_class_counts", quick)

    lengths = pd.DataFrame({
        "Words": train["clean_text"].map(lambda text: len(tokenize(text))),
        "Characters (with spaces)": train["clean_text"].str.len(),
        "Characters (without spaces)": train["clean_text"].str.replace(" ", "", regex=False).str.len(),
    })
    print(lengths.describe().round(2).to_string(), flush=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for axis, column in zip(axes, ("Words", "Characters (with spaces)")):
        axis.hist(lengths[column], bins=40)
        axis.set(xlabel=column, ylabel="Training examples")
    figure.suptitle(f"{name} cleaned training text lengths")
    save_plot(figure, f"{name}_lengths", quick)


def scores(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def baseline(frames, classes):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False, max_features=20000,
        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
    )
    features = vectorizer.fit_transform(frames["train"]["clean_text"])
    model = LogisticRegression(
        C=1, max_iter=1000, solver="liblinear" if classes == 2 else "lbfgs", random_state=SEED,
    )
    model.fit(features, frames["train"]["label"])
    result = {"model": "baseline", "weights": model.coef_.size + model.intercept_.size}
    for split in ("validation", "test"):
        probabilities = model.predict_proba(vectorizer.transform(frames[split]["clean_text"]))
        result[split] = scores(frames[split]["label"], probabilities)
    return result


class TransformerClassifier(nn.Module):
    def __init__(self, vocabulary_size, classes):
        super().__init__()
        width = 64
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        position = torch.arange(MAX_LENGTH).unsqueeze(1)
        rates = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(1, MAX_LENGTH, width)
        encoding[0, :, 0::2] = torch.sin(position * rates)
        encoding[0, :, 1::2] = torch.cos(position * rates)
        self.register_buffer("position", encoding)
        self.dropout = nn.Dropout(0.2)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=128, dropout=0.2,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, classes),
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.dropout(self.embedding(input_ids) * 8 + self.position[:, :input_ids.shape[1]])
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        weights = attention_mask.unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.classifier(pooled)


def encode_data(frames, tokenizer=None):
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    if tokenizer is None:
        counts = Counter(token for text in frames["train"]["clean_text"] for token in tokenize(text))
        for token, count in counts.most_common(19998):
            if count >= 2:
                vocabulary[token] = len(vocabulary)
    datasets = {}
    for split, frame in frames.items():
        if tokenizer is not None:
            encoded = tokenizer(
                frame["text"].tolist(), padding="max_length", truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            ids, masks = encoded["input_ids"], encoded["attention_mask"]
        else:
            rows, attention = [], []
            for text in frame["clean_text"]:
                tokens = tokenize(text)[:MAX_LENGTH]
                row = [vocabulary.get(token, 1) for token in tokens] or [1]
                attention.append([1] * len(row) + [0] * (MAX_LENGTH - len(row)))
                rows.append(row + [0] * (MAX_LENGTH - len(row)))
            ids, masks = torch.tensor(rows), torch.tensor(attention)
        datasets[split] = TensorDataset(ids, masks, torch.tensor(frame["label"].to_numpy()))
    return datasets, len(vocabulary)


def logits_of(model, ids, masks):
    length = max(1, int(masks.sum(dim=1).max().item()))
    output = model(input_ids=ids[:, :length], attention_mask=masks[:, :length])
    return output.logits if hasattr(output, "logits") else output


def evaluate(model, loader, device):
    model.eval()
    total_loss, labels, probabilities = 0.0, [], []
    with torch.no_grad():
        for ids, masks, target in loader:
            logits = logits_of(model, ids.to(device), masks.to(device))
            total_loss += nn.functional.cross_entropy(logits, target.to(device), reduction="sum").item()
            probabilities.append(logits.softmax(-1).cpu().numpy())
            labels.extend(target.tolist())
    return total_loss / len(labels), scores(labels, np.concatenate(probabilities))


def train_model(model, datasets, name, model_name, epochs, device, quick):
    model.to(device)
    loaders = {
        split: DataLoader(
            data, batch_size=BATCH_SIZE, shuffle=split == "train",
            generator=torch.Generator().manual_seed(SEED),
        )
        for split, data in datasets.items()
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)
    history, best_state, best_accuracy, stale = [], None, -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        if model_name.startswith("transfer_") and model_name != "transfer_full":
            model.bert.eval()
            for block in model.bert.encoder.layer:
                if any(parameter.requires_grad for parameter in block.parameters()):
                    block.train()
        total_loss = 0.0
        for ids, masks, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = logits_of(model, ids.to(device), masks.to(device))
            loss = nn.functional.cross_entropy(logits, target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            total_loss += loss.item() * len(target)
        validation_loss, validation = evaluate(model, loaders["validation"], device)
        history.append((epoch, total_loss / len(datasets["train"]), validation_loss, validation["accuracy"]))
        print(f"{name} {model_name} epoch {epoch}: validation accuracy={validation['accuracy']:.4f}", flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, stale = validation["accuracy"], 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 2:
            break
    model.load_state_dict(best_state)
    result = {"model": model_name, "weights": sum(parameter.numel() for parameter in parameters)}
    for split in ("validation", "test"):
        result[split] = evaluate(model, loaders[split], device)[1]
    values = np.asarray(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(values[:, 0], values[:, 1], "o-", label="Training")
    axes[0].plot(values[:, 0], values[:, 2], "o-", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(values[:, 0], 100 * values[:, 3], "o-")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0, 100))
    for axis in axes:
        axis.set_xticks(values[:, 0].astype(int))
    save_plot(figure, f"{name}_{model_name}_learning", quick)
    return result


def run_transfer(frames, labels, epochs, device, quick):
    from transformers import BertForSequenceClassification, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)
    datasets, _ = encode_data(frames, tokenizer)
    results = []
    for mode in ("head", "upper", "lower", "full"):
        set_seed()
        model = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(labels))
        for parameter in model.parameters():
            parameter.requires_grad = mode == "full"
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if mode in ("upper", "lower"):
            block = model.bert.encoder.layer[-1 if mode == "upper" else 0]
            for parameter in block.parameters():
                parameter.requires_grad = True
        results.append(train_model(
            model, datasets, "banking77", f"transfer_{mode}", epochs, device, quick,
        ))
    return results


def plot_results(results, name, quick):
    print(f"\n{name} results", flush=True)
    table = pd.DataFrame([{
        "Model": result["model"], "Trainable weights": result["weights"],
        "Validation accuracy (%)": 100 * result["validation"]["accuracy"],
        "Test accuracy (%)": 100 * result["test"]["accuracy"],
        "Test macro-F1 (%)": 100 * result["test"]["macro_f1"],
    } for result in results])
    print(table.round(2).to_string(index=False), flush=True)
    figure, axis = plt.subplots(figsize=(11, 1.2 + 0.45 * len(results)), layout="constrained")
    axis.axis("off")
    cells = [[row["model"], f"{row['weights']:,}",
              f"{100 * row['validation']['accuracy']:.2f}",
              f"{100 * row['test']['accuracy']:.2f}",
              f"{100 * row['test']['macro_f1']:.2f}"] for row in results]
    summary = axis.table(cellText=cells, colLabels=["Model", "Trainable weights",
                         "Validation accuracy (%)", "Test accuracy (%)", "Test macro-F1 (%)"], loc="center")
    summary.auto_set_font_size(False)
    summary.set_fontsize(9)
    summary.scale(1, 1.5)
    save_plot(figure, f"{name}_results", quick)
    figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    locations = np.arange(len(results))
    axis.bar(locations - 0.2, table["Test accuracy (%)"], width=0.4, label="Accuracy")
    axis.bar(locations + 0.2, table["Test macro-F1 (%)"], width=0.4, label="Macro-F1")
    axis.set_xticks(locations, table["Model"], rotation=20, ha="right")
    axis.set(ylabel="Test score (%)", ylim=(0, 100), title=f"{name} model comparison")
    axis.legend()
    save_plot(figure, f"{name}_comparison", quick)

    transfer = [result for result in results if result["model"].startswith("transfer_")]
    if not transfer:
        return
    weights = np.array([result["weights"] for result in transfer])
    for split in ("validation", "test"):
        accuracy = np.array([result[split]["accuracy"] for result in transfer])
        frontier = np.array([
            not np.any((weights <= count) & (accuracy >= score)
                       & ((weights < count) | (accuracy > score)))
            for count, score in zip(weights, accuracy)
        ])
        figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
        axis.scatter(weights, accuracy * 100, label="Configurations")
        order = np.flatnonzero(frontier)[np.argsort(weights[frontier])]
        axis.plot(weights[order], accuracy[order] * 100, "o--", color="#c05640", label="Pareto frontier")
        for count, score, result in zip(weights, accuracy, transfer):
            offset = {"transfer_upper": (6, 10), "transfer_lower": (6, -16),
                      "transfer_full": (-6, 10)}.get(result["model"], (6, 8))
            axis.annotate(
                result["model"].replace("transfer_", ""), (count, 100 * score),
                xytext=offset, textcoords="offset points",
                ha="right" if result["model"] == "transfer_full" else "left",
            )
        axis.set(xscale="log", xlabel="Trainable parameters (log scale)",
                 ylabel=f"{split.title()} accuracy (%)", ylim=(0, 103), title="BANKING77 fine-tuning")
        axis.legend()
        save_plot(figure, f"{name}_pareto_{split}", quick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", "imdb", "banking77", "transfer"), default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Small real-data test with one epoch")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    epochs = 1 if args.quick else args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    print(f"Using {device}. Figures are saved in {FIGURES}", flush=True)
    names = ("imdb", "banking77") if args.part == "all" else (
        ("banking77",) if args.part == "transfer" else (args.part,)
    )
    with threadpool_limits(limits=4):
        for name in names:
            frames, labels = prepare_data(name, args.quick)
            plot_dataset(frames, labels, name, args.quick)
            results = []
            if args.part != "transfer":
                results.append(baseline(frames, len(labels)))
                set_seed()
                datasets, vocabulary_size = encode_data(frames)
                model = TransformerClassifier(vocabulary_size, len(labels))
                results.append(train_model(
                    model, datasets, name, "scratch", epochs, device, args.quick,
                ))
            if name == "banking77" and args.part in ("all", "transfer"):
                results.extend(run_transfer(frames, labels, epochs, device, args.quick))
            plot_results(results, name, args.quick)
    print(f"\nFinished. All figures are in {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
