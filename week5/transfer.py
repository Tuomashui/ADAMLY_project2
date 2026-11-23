"""Controlled BERT head, upper block, lower block, and full fine-tuning."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import TensorDataset

MODES = ("head", "upper", "lower", "full")


def encode_pretrained(frames, checkpoint, max_length):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    datasets = {}
    truncation = {}
    for split, frame in frames.items():
        # Keep casing and punctuation for the checkpoint tokenizer.
        texts = frame["text"].tolist()
        lengths = tokenizer(texts, truncation=False, verbose=False)["input_ids"]
        truncation[split] = sum(len(row) > max_length for row in lengths) / len(lengths)
        encoded = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=max_length, return_tensors="pt")
        datasets[split] = TensorDataset(
            encoded["input_ids"], encoded["attention_mask"],
            torch.tensor(frame["label"].to_numpy()),
        )
    return tokenizer, datasets, truncation


def create_model(checkpoint, labels, mode):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )
    if model.config.model_type != "bert":
        raise ValueError("Layer selection supports BERT checkpoints, such as prajjwal1/bert-tiny.")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if mode in ("upper", "lower"):
        block = model.bert.encoder.layer[-1 if mode == "upper" else 0]
        for parameter in block.parameters():
            parameter.requires_grad = True
    return model
