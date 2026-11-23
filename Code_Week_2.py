
# ============================================================
# IMDb SENTIMENT CLASSIFICATION
# Week 3: Text preprocessing, Logistic Regression and Transformer
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import re
import html
import math
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Transformer configuration
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LENGTH = 256

D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
PATIENCE = 2


# ============================================================
# 3. LOAD IMDb DATASET
# ============================================================

print("\nLoading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Split original training data into training and validation
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"]
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nDataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    # Decode HTML entities
    text = html.unescape(text)

    # Replace HTML line breaks
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning reviews...")

train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

print("Text cleaning completed!")

print("\nExample original review:")
print(train_df["text"].iloc[0][:500])

print("\nExample cleaned review:")
print(train_df["clean_text"].iloc[0][:500])


# ============================================================
# 5. DATA EXPLORATION
# ============================================================

# 5.1 Class distribution

class_counts = train_df["label"].value_counts().sort_index()

print("\nClass distribution:")
print(class_counts)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train_df,
    x="label"
)

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.title("Distribution of sentiment classes")

plt.tight_layout()
plt.show()


# 5.2 Review length statistics

train_df["char_length"] = train_df["clean_text"].str.len()

train_df["word_length"] = train_df["clean_text"].apply(
    lambda x: len(x.split())
)

print("\nReview length statistics:")
print(
    train_df[
        ["char_length", "word_length"]
    ].describe()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    data=train_df,
    x="word_length",
    hue="label",
    bins=50,
    ax=axes[0],
    element="step"
)

axes[0].set_title("Distribution of review word counts")
axes[0].set_xlabel("Number of words")

sns.boxplot(
    data=train_df,
    x="label",
    y="word_length",
    ax=axes[1]
)

axes[1].set_xticks(
    [0, 1],
    ["Negative", "Positive"]
)

axes[1].set_title("Review length by sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Number of words")

plt.tight_layout()
plt.show()


# ============================================================
# 6. WORD CLOUDS AND FREQUENT WORDS
# ============================================================

positive_text = " ".join(
    train_df.loc[
        train_df["label"] == 1,
        "clean_text"
    ]
)

negative_text = " ".join(
    train_df.loc[
        train_df["label"] == 0,
        "clean_text"
    ]
)

# Preserve negation words
stopwords = set(ENGLISH_STOP_WORDS) - {
    "no", "not", "nor", "never"
}

# Word clouds
positive_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=100,
    stopwords=stopwords,
    collocations=False
).generate(positive_text)

negative_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=100,
    stopwords=stopwords,
    collocations=False
).generate(negative_text)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(
    positive_wc,
    interpolation="bilinear"
)

axes[0].set_title("Positive Reviews")
axes[0].axis("off")

axes[1].imshow(
    negative_wc,
    interpolation="bilinear"
)

axes[1].set_title("Negative Reviews")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# Most frequent words

def plot_common_words(text, title, n=20):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    words = [
        word for word in words
        if word not in stopwords
    ]

    word_counts = Counter(words).most_common(n)

    words, counts = zip(*word_counts)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=list(counts),
        y=list(words)
    )

    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")

    plt.tight_layout()
    plt.show()


plot_common_words(
    positive_text,
    "Most common words in positive reviews"
)

plot_common_words(
    negative_text,
    "Most common words in negative reviews"
)


# ============================================================
# 7. TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Fit only on training data
X_train = tfidf.fit_transform(train_df["clean_text"])

# Transform validation and test data
X_val = tfidf.transform(val_df["clean_text"])
X_test = tfidf.transform(test_df["clean_text"])

y_train = train_df["label"].to_numpy()
y_val = val_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

print("\nTF-IDF dimensions:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("Vocabulary size:", len(tfidf.get_feature_names_out()))


# ============================================================
# 8. TF-IDF DIMENSIONALITY REDUCTION
# ============================================================

print("\nPerforming TruncatedSVD...")

svd = TruncatedSVD(
    n_components=2,
    random_state=SEED
)

# Use a sample for faster visualization
svd_sample_size = min(5000, X_train.shape[0])

X_svd_sample = X_train[:svd_sample_size]
y_svd_sample = y_train[:svd_sample_size]

X_2d = svd.fit_transform(X_svd_sample)

print(
    "Explained variance by two components:",
    f"{svd.explained_variance_ratio_.sum():.4f}"
)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y_svd_sample,
    cmap="coolwarm",
    alpha=0.4,
    s=10
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("IMDb Reviews - TF-IDF dimensionality reduction")

plt.colorbar(
    scatter,
    label="Sentiment"
)

plt.tight_layout()
plt.show()


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(y_true, predictions, probabilities):

    return {
        "Accuracy": accuracy_score(y_true, predictions),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1-score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


def print_metrics(metrics, title):

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def plot_confusion_matrix(y_true, predictions, title):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. BASELINE: LOGISTIC REGRESSION
# ============================================================

print("\nTraining logistic regression...")

logistic_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
    random_state=SEED
)

logistic_model.fit(
    X_train,
    y_train
)

# Validation predictions
lr_val_pred = logistic_model.predict(X_val)

lr_val_prob = logistic_model.predict_proba(X_val)[:, 1]

lr_val_metrics = calculate_metrics(
    y_val,
    lr_val_pred,
    lr_val_prob
)

print_metrics(
    lr_val_metrics,
    "LOGISTIC REGRESSION - VALIDATION RESULTS"
)

print("\nLogistic regression classification report:")

print(
    classification_report(
        y_val,
        lr_val_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0
    )
)

plot_confusion_matrix(
    y_val,
    lr_val_pred,
    "Logistic Regression - Validation Confusion Matrix"
)


# Most influential TF-IDF features

feature_names = np.array(
    tfidf.get_feature_names_out()
)

coefficients = logistic_model.coef_[0]

positive_indices = np.argsort(coefficients)[-20:]
negative_indices = np.argsort(coefficients)[:20]

positive_words = feature_names[positive_indices]
positive_values = coefficients[positive_indices]

negative_words = feature_names[negative_indices]
negative_values = coefficients[negative_indices]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    x=positive_values,
    y=positive_words,
    ax=axes[0]
)

axes[0].set_title("Features associated with positive sentiment")
axes[0].set_xlabel("Coefficient")
axes[0].set_ylabel("Feature")

sns.barplot(
    x=negative_values,
    y=negative_words,
    ax=axes[1]
)

axes[1].set_title("Features associated with negative sentiment")
axes[1].set_xlabel("Coefficient")
axes[1].set_ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 11. TRANSFORMER TOKENIZATION
# ============================================================

print("\nBuilding Transformer vocabulary...")

def tokenize(text):

    return re.findall(
        r"\b\w+(?:'\w+)?\b|[^\w\s]",
        text.lower()
    )


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1

counter = Counter()

# Build vocabulary from training reviews only
for text in train_df["clean_text"]:
    counter.update(tokenize(text))

vocabulary = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID
}

for word, count in counter.most_common():

    if count < 2:
        break

    if len(vocabulary) >= MAX_VOCAB_SIZE:
        break

    vocabulary[word] = len(vocabulary)

print("Vocabulary size:", len(vocabulary))


# ============================================================
# 12. CONVERT REVIEWS TO INTEGER SEQUENCES
# ============================================================

def encode_text(text):

    tokens = tokenize(text)

    token_ids = [
        vocabulary.get(token, UNK_ID)
        for token in tokens
    ]

    # Truncate long reviews while retaining both
    # the beginning and the end of each review.
    if len(token_ids) > MAX_SEQ_LENGTH:

        half_length = MAX_SEQ_LENGTH // 2

        token_ids = (
            token_ids[:half_length]
            + token_ids[-half_length:]
        )

    # Pad shorter reviews
    padding_length = MAX_SEQ_LENGTH - len(token_ids)

    token_ids += [PAD_ID] * padding_length

    return token_ids


def prepare_dataset(dataframe):

    sequences = [
        encode_text(text)
        for text in dataframe["clean_text"]
    ]

    X = torch.tensor(
        sequences,
        dtype=torch.long
    )

    y = torch.tensor(
        dataframe["label"].to_numpy(),
        dtype=torch.float32
    )

    return TensorDataset(X, y)


print("\nEncoding reviews...")

train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# ============================================================
# 13. TRANSFORMER COMPONENTS
# ============================================================

class InputEmbeddings(nn.Module):

    def __init__(
        self,
        d_model,
        vocab_size,
        padding_idx
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx
        )

    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        seq_len,
        dropout
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(
            0,
            seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)


# ============================================================
# 14. TRANSFORMER ENCODER CLASSIFIER
# ============================================================

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        seq_len,
        dropout,
        padding_idx
    ):
        super().__init__()

        self.padding_idx = padding_idx

        self.embedding = InputEmbeddings(
            d_model,
            vocab_size,
            padding_idx
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            seq_len,
            dropout
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):

        # Identify padding tokens
        padding_mask = input_ids == self.padding_idx

        # Embedding
        x = self.embedding(input_ids)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Masked mean pooling
        valid_mask = (
            ~padding_mask
        ).unsqueeze(-1).float()

        x = x * valid_mask

        pooled = (
            x.sum(dim=1)
            / valid_mask.sum(dim=1).clamp(min=1)
        )

        # Classification output
        logits = self.classifier(pooled)

        return logits.squeeze(-1)


# ============================================================
# 15. INITIALIZE TRANSFORMER
# ============================================================

print("\nInitializing Transformer...")

transformer_model = TransformerClassifier(
    vocab_size=len(vocabulary),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    seq_len=MAX_SEQ_LENGTH,
    dropout=DROPOUT,
    padding_idx=PAD_ID
).to(device)

total_parameters = sum(
    parameter.numel()
    for parameter in transformer_model.parameters()
    if parameter.requires_grad
)

print(transformer_model)

print("\nTrainable parameters:", total_parameters)


# ============================================================
# 16. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    transformer_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 17. TRANSFORMER EVALUATION FUNCTION
# ============================================================

def evaluate_transformer(model, dataloader):

    model.eval()

    all_labels = []
    all_probabilities = []

    total_loss = 0

    with torch.no_grad():

        for input_ids, labels in dataloader:

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.sigmoid(logits)

            total_loss += (
                loss.item() * labels.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    predictions = (
        all_probabilities >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        all_labels,
        predictions,
        all_probabilities
    )

    metrics["Loss"] = (
        total_loss / len(dataloader.dataset)
    )

    return (
        metrics,
        all_labels,
        predictions,
        all_probabilities
    )


# ============================================================
# 18. TRAIN TRANSFORMER
# ============================================================

best_f1 = -1
best_model_state = None
epochs_without_improvement = 0

history = []

print("\nStarting Transformer training...")

for epoch in range(EPOCHS):

    transformer_model.train()

    total_train_loss = 0

    for input_ids, labels in train_loader:

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = transformer_model(input_ids)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            transformer_model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += (
            loss.item() * labels.size(0)
        )

    train_loss = (
        total_train_loss / len(train_loader.dataset)
    )

    # Validation
    val_metrics, _, _, _ = evaluate_transformer(
        transformer_model,
        val_loader
    )

    history.append({
        "Epoch": epoch + 1,
        "Training Loss": train_loss,
        "Validation Loss": val_metrics["Loss"],
        "Validation Accuracy": val_metrics["Accuracy"],
        "Validation F1": val_metrics["F1-score"]
    })

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"Training loss:   {train_loss:.4f}")
    print(f"Validation loss: {val_metrics['Loss']:.4f}")
    print(f"Accuracy:        {val_metrics['Accuracy']:.4f}")
    print(f"Precision:       {val_metrics['Precision']:.4f}")
    print(f"Recall:          {val_metrics['Recall']:.4f}")
    print(f"F1-score:        {val_metrics['F1-score']:.4f}")
    print(f"ROC-AUC:         {val_metrics['ROC-AUC']:.4f}")

    # Save best model based on validation F1
    if val_metrics["F1-score"] > best_f1:

        best_f1 = val_metrics["F1-score"]

        best_model_state = copy.deepcopy(
            transformer_model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# Load best model
if best_model_state is not None:

    transformer_model.load_state_dict(
        best_model_state
    )

print("\nTransformer training completed!")


# ============================================================
# 19. TRANSFORMER VALIDATION RESULTS
# ============================================================

(
    transformer_val_metrics,
    transformer_y_val,
    transformer_val_pred,
    transformer_val_prob
) = evaluate_transformer(
    transformer_model,
    val_loader
)

print_metrics(
    transformer_val_metrics,
    "TRANSFORMER - VALIDATION RESULTS"
)

print("\nTransformer classification report:")

print(
    classification_report(
        transformer_y_val,
        transformer_val_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0
    )
)

plot_confusion_matrix(
    transformer_y_val,
    transformer_val_pred,
    "Transformer - Validation Confusion Matrix"
)


# ============================================================
# 20. TRANSFORMER TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    history_df["Epoch"],
    history_df["Training Loss"],
    marker="o",
    label="Training loss"
)

axes[0].plot(
    history_df["Epoch"],
    history_df["Validation Loss"],
    marker="o",
    label="Validation loss"
)

axes[0].set_title("Transformer training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    history_df["Epoch"],
    history_df["Validation Accuracy"],
    marker="o",
    label="Validation accuracy"
)

axes[1].plot(
    history_df["Epoch"],
    history_df["Validation F1"],
    marker="o",
    label="Validation F1"
)

axes[1].set_title("Transformer validation performance")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 21. VALIDATION PERFORMANCE COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],

    "Logistic Regression": [
        lr_val_metrics["Accuracy"],
        lr_val_metrics["Precision"],
        lr_val_metrics["Recall"],
        lr_val_metrics["F1-score"],
        lr_val_metrics["ROC-AUC"]
    ],

    "Transformer": [
        transformer_val_metrics["Accuracy"],
        transformer_val_metrics["Precision"],
        transformer_val_metrics["Recall"],
        transformer_val_metrics["F1-score"],
        transformer_val_metrics["ROC-AUC"]
    ]
})

comparison["Difference"] = (
    comparison["Transformer"]
    - comparison["Logistic Regression"]
)

print("\n========================================")
print("VALIDATION MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(
        index=False,
        formatters={
            "Logistic Regression": "{:.4f}".format,
            "Transformer": "{:.4f}".format,
            "Difference": "{:+.4f}".format
        }
    )
)


# Comparison plot
comparison_plot = comparison.set_index("Metric")[
    [
        "Logistic Regression",
        "Transformer"
    ]
]

comparison_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Logistic Regression vs Transformer")
plt.ylabel("Validation score")
plt.xlabel("Metric")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 22. FINAL TEST SET EVALUATION
# ============================================================

# The test set has not been used for training or model selection.
# Evaluate both models only after completing the comparison.

print("\nEvaluating final models on the test set...")


# Logistic regression test results

lr_test_pred = logistic_model.predict(X_test)

lr_test_prob = logistic_model.predict_proba(X_test)[:, 1]

lr_test_metrics = calculate_metrics(
    y_test,
    lr_test_pred,
    lr_test_prob
)

print_metrics(
    lr_test_metrics,
    "LOGISTIC REGRESSION - TEST RESULTS"
)


# Transformer test results

(
    transformer_test_metrics,
    transformer_y_test,
    transformer_test_pred,
    transformer_test_prob
) = evaluate_transformer(
    transformer_model,
    test_loader
)

print_metrics(
    transformer_test_metrics,
    "TRANSFORMER - TEST RESULTS"
)


# Test set comparison
test_comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],

    "Logistic Regression": [
        lr_test_metrics["Accuracy"],
        lr_test_metrics["Precision"],
        lr_test_metrics["Recall"],
        lr_test_metrics["F1-score"],
        lr_test_metrics["ROC-AUC"]
    ],

    "Transformer": [
        transformer_test_metrics["Accuracy"],
        transformer_test_metrics["Precision"],
        transformer_test_metrics["Recall"],
        transformer_test_metrics["F1-score"],
        transformer_test_metrics["ROC-AUC"]
    ]
})

test_comparison["Difference"] = (
    test_comparison["Transformer"]
    - test_comparison["Logistic Regression"]
)

print("\n========================================")
print("FINAL TEST SET COMPARISON")
print("========================================")

print(
    test_comparison.to_string(
        index=False,
        formatters={
            "Logistic Regression": "{:.4f}".format,
            "Transformer": "{:.4f}".format,
            "Difference": "{:+.4f}".format
        }
    )
)


# ============================================================
# 23. FINAL TEST CONFUSION MATRICES
# ============================================================

plot_confusion_matrix(
    y_test,
    lr_test_pred,
    "Logistic Regression - Test Confusion Matrix"
)

plot_confusion_matrix(
    transformer_y_test,
    transformer_test_pred,
    "Transformer - Test Confusion Matrix"
)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

comparison.to_csv(
    "validation_model_comparison.csv",
    index=False
)

test_comparison.to_csv(
    "test_model_comparison.csv",
    index=False
)

history_df.to_csv(
    "transformer_training_history.csv",
    index=False
)

torch.save(
    transformer_model.state_dict(),
    "imdb_transformer_model.pth"
)


# ============================================================
# 25. COMPLETION
# ============================================================

print("\n========================================")
print("IMDb CLASSIFICATION COMPLETED")
print("========================================")

print("Logistic regression and Transformer trained.")
print("Validation and test comparisons completed.")
print("Results and Transformer model saved.")
