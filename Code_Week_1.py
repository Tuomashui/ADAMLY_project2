
# ============================================================
# IMDb SENTIMENT CLASSIFICATION
# Week 3: Text preprocessing, visualization and baseline model
# ============================================================


# 1. IMPORT LIBRARIES
# ============================================================

import re
import html
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
plt.title("Distribution of sentiment classes")

plt.tight_layout()
plt.show()


# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    # Decode HTML entities
    text = html.unescape(text)

    # Replace HTML line breaks with spaces
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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

# Calculate review lengths using cleaned text
train_df["char_length"] = train_df["clean_text"].str.len()

train_df["word_length"] = train_df["clean_text"].apply(
    lambda x: len(x.split())
)

print("\nReview length statistics:")
print(train_df[["char_length", "word_length"]].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word count distribution
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

# Review length by sentiment
sns.boxplot(
    data=train_df,
    x="label",
    y="word_length",
    ax=axes[1]
)

axes[1].set_xticks([0, 1], ["Negative", "Positive"])
axes[1].set_title("Review length by sentiment")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Number of words")

plt.tight_layout()
plt.show()


# 6. TOKENIZATION
# ============================================================

# Example whitespace tokenization
example = train_df["clean_text"].iloc[0]

tokens = example.split()

print("\nExample tokens:")
print(tokens[:30])


# 7. TF-IDF VECTORIZATION
# ============================================================

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

# Combine positive and negative reviews separately
positive_text = " ".join(
    train_df[train_df["label"] == 1]["clean_text"]
)

negative_text = " ".join(
    train_df[train_df["label"] == 0]["clean_text"]
)

# English stopwords, preserving negation words
stopwords = set(ENGLISH_STOP_WORDS) - {
    "no", "not", "nor", "never"
}

# Generate word clouds
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

# Display word clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(positive_wc, interpolation="bilinear")
axes[0].set_title("Positive Reviews")
axes[0].axis("off")

axes[1].imshow(negative_wc, interpolation="bilinear")
axes[1].set_title("Negative Reviews")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# 9. MOST FREQUENT WORDS
# ============================================================

def plot_common_words(text, title, n=20):

    # Extract words and remove punctuation
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    # Remove common English stopwords
    words = [
        word for word in words
        if word not in stopwords
    ]

    word_counts = Counter(words).most_common(n)

    if not word_counts:
        print("No words available for:", title)
        return

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

# Visualize reviews in two dimensions
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.4,
    s=10
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("IMDb Reviews - TF-IDF dimensionality reduction")

plt.colorbar(scatter, label="Sentiment")

plt.tight_layout()
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

plt.title("Logistic Regression - Confusion Matrix")
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

plt.title("Logistic Regression - ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# 19. MOST INFLUENTIAL TF-IDF FEATURES
# ============================================================

# Extract TF-IDF feature names and model coefficients
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = model.coef_[0]

# Identify 20 most positive and negative coefficients
positive_indices = np.argsort(coefficients)[-20:]
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
