"""Controlled BERT head, upper block, lower block, and full fine-tuning."""
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import TensorDataset

MODES = ("head", "upper", "lower", "full")


def checkpoint_revision(checkpoint):
    """Read the resolved Hub revision without relying on private model attributes."""
    if Path(checkpoint).is_dir():
        return None
    config_path = try_to_load_from_cache(checkpoint, "config.json")
    return Path(config_path).parent.name if isinstance(config_path, str) else None


def encode_pretrained(frames, checkpoint, max_length):
    # Older BERT checkpoints may provide vocab.txt without tokenizer metadata.
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
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
    # The original BERT-tiny config predates the model_type metadata field.
    model = BertForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if mode in ("upper", "lower"):
        block = model.bert.encoder.layer[-1 if mode == "upper" else 0]
        for parameter in block.parameters():
            parameter.requires_grad = True
    return model
