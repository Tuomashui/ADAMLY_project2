
# ============================================================
# IMDb SENTIMENT CLASSIFICATION
# Week 3: Text preprocessing, visualization and baseline model
# ============================================================


# 1. IMPORT LIBRARIES
# ============================================================

import re
import html
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


FIGURE_DIR = Path(__file__).resolve().parent / "figs"
FIGURE_DIR.mkdir(exist_ok=True)
SENTIMENTS = {0: "Negative", 1: "Positive"}
SENTIMENT_COLORS = {"Negative": "#c44e52", "Positive": "#4c72b0"}
WORD_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")
WORD_CONNECTORS = str.maketrans("\u2019\u2010\u2011", "'--")


def tokenize(text):
    """Return words and numbers, keeping internal apostrophes and hyphens."""
    return WORD_PATTERN.findall(text.translate(WORD_CONNECTORS))


# 2. LOAD DATASET
# ============================================================

dataset = load_dataset("stanfordnlp/imdb")

# Convert labelled datasets into Pandas DataFrames
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

print("Dataset loaded successfully!")

# Split original training data into training and validation sets
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"]
)

print("\nDataset sizes:")
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))


# 3. DATA EXPLORATION
# ============================================================

# 3.1 Class balance

class_counts = train_df["label"].value_counts().sort_index()

print("\nClass distribution:")
print(class_counts)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train_df,
    x="label"
)

plt.xticks([0, 1], ["Negative", "Positive"])
plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.title("Sentiment classes in the training split")

plt.tight_layout()
plt.show()


# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    # Decode HTML entities
    text = html.unescape(text)

    # Replace HTML tags with spaces without removing comparisons such as 2 < 3.
    text = re.sub(r"</?[a-zA-Z][\w:-]*(?=[\s/>])[^<>]*>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"\b(?:https?://|www\.)[^\s<>]+", " ", text)

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return unicodedata.normalize("NFC", text)


# Apply cleaning to all datasets
train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# Compare original and cleaned reviews
print("\nOriginal review:")
print(train_df["text"].iloc[0])

print("\nCleaned review:")
print(train_df["clean_text"].iloc[0])


# 5. REVIEW LENGTH ANALYSIS
# ============================================================

# Character counts include punctuation, with whitespace normalized to single spaces.
train_df["char_length"] = train_df["clean_text"].str.len()
train_df["char_length_no_spaces"] = train_df["clean_text"].str.replace(
    " ", "", regex=False
).str.len()
train_df["tokens"] = train_df["clean_text"].map(tokenize)
train_df["word_length"] = train_df["tokens"].map(len)
train_df["sentiment"] = train_df["label"].map(SENTIMENTS)

length_labels = {
    "word_length": "Words",
    "char_length": "Characters (with spaces)",
    "char_length_no_spaces": "Characters (without spaces)",
}
print("\nCleaned training review lengths (per review):")
print(train_df[list(length_labels)].rename(columns=length_labels).describe().round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
for column, metric in enumerate(("word_length", "char_length")):
    label = length_labels[metric]
    sns.histplot(
        data=train_df, x=metric, hue="sentiment", hue_order=list(SENTIMENT_COLORS),
        palette=SENTIMENT_COLORS, bins=50, element="step", fill=False,
        ax=axes[0, column],
    )
    axes[0, column].set(xlabel=label, ylabel="Number of reviews")
    sns.boxplot(
        data=train_df, x="sentiment", y=metric, order=list(SENTIMENT_COLORS),
        ax=axes[1, column],
    )
    axes[1, column].set(xlabel="Sentiment", ylabel=label)
fig.suptitle("Cleaned IMDb training review lengths")
fig.savefig(FIGURE_DIR / "word_distribution.png", dpi=160)
plt.show()


# 6. TOKENIZATION
# ============================================================

example = train_df.iloc[0]
print(
    f"\nExample review length: {example['word_length']} words, "
    f"{example['char_length']} characters with spaces, "
    f"{example['char_length_no_spaces']} without spaces."
)
print("First 30 example tokens:", example["tokens"][:30])


# 7. TF-IDF VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(
    tokenizer=tokenize,
    token_pattern=None,
    lowercase=False,
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

# Extract sentiment labels
y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

print("\nTF-IDF results:")
print("Training matrix:", X_train.shape)
print("Validation matrix:", X_val.shape)
print("Test matrix:", X_test.shape)

print("\nNumber of features:", len(tfidf.get_feature_names_out()))


# 8. WORD CLOUD VISUALIZATION
# ============================================================

# Use the same token counts for word clouds and frequency bars.
stopwords = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}
sentiment_counts = {}
for sentiment, reviews in train_df.groupby("sentiment"):
    sentiment_counts[sentiment] = Counter(
        word for tokens in reviews["tokens"] for word in tokens if word not in stopwords
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
for ax, sentiment in zip(axes, SENTIMENT_COLORS):
    cloud = WordCloud(
        width=1000, height=500, background_color="white",
        max_words=100, random_state=42,
    ).generate_from_frequencies(sentiment_counts[sentiment])
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title(sentiment)
    ax.axis("off")
fig.suptitle("Words in IMDb training reviews (excluding common stopwords)")
plt.show()


# 9. MOST FREQUENT WORDS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, layout="constrained")
for ax, (sentiment, color) in zip(axes, SENTIMENT_COLORS.items()):
    words, counts = zip(*sentiment_counts[sentiment].most_common(20))
    ax.barh(words, counts, color=color)
    ax.invert_yaxis()
    ax.set(title=sentiment, xlabel="Word occurrences")
fig.suptitle("Most frequent words in IMDb training reviews (excluding common stopwords)")
plt.show()


# 10. DIMENSIONALITY REDUCTION VISUALIZATION
# ============================================================

# Reduce TF-IDF features to two dimensions using SVD
svd = TruncatedSVD(
    n_components=2,
    random_state=42
)

X_2d = svd.fit_transform(X_train)

# Print explained variance
explained_variance = svd.explained_variance_ratio_.sum()

print("\nSVD explained variance:")
print(f"Two components: {explained_variance:.4f}")

# Plot discrete sentiment classes on the training projection.
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
for label, sentiment in SENTIMENTS.items():
    selected = y_train.to_numpy() == label
    ax.scatter(
        X_2d[selected, 0], X_2d[selected, 1],
        color=SENTIMENT_COLORS[sentiment], label=sentiment,
        alpha=0.3, s=10, linewidths=0,
    )
ax.set(
    xlabel="Component 1",
    ylabel="Component 2",
    title=f"Training review TF-IDF projection ({explained_variance:.1%} variance explained)",
)
ax.legend(title="Sentiment")
fig.savefig(FIGURE_DIR / "TF_IDF.png", dpi=160)
plt.show()


# 11. FINAL DATASET INFORMATION
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print("\nDataset dimensions:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nLabel dimensions:")
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

print("\nClass distributions:")
print("Training:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())

print("\nTest:")
print(y_test.value_counts())

print("\nAll preprocessing steps completed successfully!")


# ============================================================
# WEEK 3: BASELINE LOGISTIC REGRESSION
# ============================================================


# 12. INITIALIZE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
    random_state=42
)


# 13. TRAIN THE MODEL
# ============================================================

print("\nTraining logistic regression model...")

model.fit(X_train, y_train)

print("Training completed!")


# 14. MAKE PREDICTIONS ON VALIDATION DATA
# ============================================================

# Predict sentiment classes
y_pred = model.predict(X_val)

# Predict positive class probabilities
y_prob = model.predict_proba(X_val)[:, 1]


# 15. CALCULATE CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_prob)

print("\n========================================")
print("LOGISTIC REGRESSION - VALIDATION RESULTS")
print("========================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0
    )
)


# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
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

plt.title("Validation confusion matrix")
plt.tight_layout()
plt.show()


# 18. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_val,
    y_prob,
    name="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier"
)

plt.title("Validation ROC curve")
plt.legend()
plt.tight_layout()
plt.show()


# 19. MOST INFLUENTIAL TF-IDF FEATURES
# ============================================================

# Extract TF-IDF feature names and model coefficients
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = model.coef_[0]

# Identify 20 most positive and negative coefficients
positive_indices = np.argsort(coefficients)[-20:][::-1]
negative_indices = np.argsort(coefficients)[:20]

positive_words = feature_names[positive_indices]
positive_values = coefficients[positive_indices]

negative_words = feature_names[negative_indices]
negative_values = coefficients[negative_indices]

# Visualize influential features
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    x=positive_values,
    y=positive_words,
    ax=axes[0]
)

axes[0].set_title("Features associated with positive sentiment")
axes[0].set_xlabel("Logistic regression coefficient")
axes[0].set_ylabel("Feature")

sns.barplot(
    x=negative_values,
    y=negative_words,
    ax=axes[1]
)

axes[1].set_title("Features associated with negative sentiment")
axes[1].set_xlabel("Logistic regression coefficient")
axes[1].set_ylabel("Feature")

plt.tight_layout()
plt.show()


# 20. FINAL RESULTS SUMMARY
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Validation Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

print("\n========================================")
print("FINAL VALIDATION RESULTS")
print("========================================")

print(results.to_string(index=False))

print("\nBaseline logistic regression completed!")
