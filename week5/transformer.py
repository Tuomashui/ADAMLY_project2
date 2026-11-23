"""Randomly initialized Transformer with training-only word vocabulary."""
import math
from collections import Counter
import torch
from torch import nn
from torch.utils.data import TensorDataset
from .common import tokenize


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, classes, max_length=128):
        super().__init__()
        width = 64
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        position = torch.arange(max_length).unsqueeze(1)
        rates = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(1, max_length, width)
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


def encode_frames(frames, max_length):
    counts = Counter(token for text in frames["train"]["clean_text"] for token in tokenize(text))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.most_common(19998):
        if count >= 2:
            vocabulary[token] = len(vocabulary)
    datasets = {}
    truncated = {}
    for split, frame in frames.items():
        rows = []
        masks = []
        lengths = []
        for text in frame["clean_text"]:
            tokens = tokenize(text)
            lengths.append(len(tokens))
            ids = [vocabulary.get(token, 1) for token in tokens[:max_length]] or [1]
            masks.append([1] * len(ids) + [0] * (max_length - len(ids)))
            rows.append(ids + [0] * (max_length - len(ids)))
        datasets[split] = TensorDataset(
            torch.tensor(rows), torch.tensor(masks), torch.tensor(frame["label"].to_numpy()),
        )
        truncated[split] = sum(length > max_length for length in lengths) / len(lengths)
    return vocabulary, datasets, truncated
