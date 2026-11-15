
# ============================================================
# IMDb SENTIMENT CLASSIFICATION
# Week 3: Text preprocessing and visualization
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# 2. LOAD DATASET
# ============================================================

dataset = load_dataset("stanfordnlp/imdb")

# Convert labelled datasets into Pandas DataFrames
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

print("Dataset loaded successfully!")
print(dataset)

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


# 3.2 Review length distribution

train_df["char_length"] = train_df["text"].apply(len)

train_df["word_length"] = train_df["text"].apply(
    lambda x: len(x.split())
)

print("\nReview length statistics:")
print(train_df[["char_length", "word_length"]].describe())

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

axes[1].set_xticks([0, 1], ["Negative", "Positive"])
axes[1].set_title("Review length by sentiment")

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


# 5. TOKENIZATION
# ============================================================

# Example whitespace tokenization
example = train_df["clean_text"].iloc[0]

tokens = example.split()

print("\nExample tokens:")
print(tokens[:30])


# 6. TF-IDF VECTORIZATION
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

print("\nFirst 30 vocabulary features:")
print(tfidf.get_feature_names_out()[:30])


# 7. WORD CLOUD VISUALIZATION
# ============================================================

# Combine positive and negative reviews separately
positive_text = " ".join(
    train_df[train_df["label"] == 1]["clean_text"]
)

negative_text = " ".join(
    train_df[train_df["label"] == 0]["clean_text"]
)

# Generate word clouds
positive_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=100
).generate(positive_text)

negative_wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=100
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


# 8. MOST FREQUENT WORDS
# ============================================================

# Remove common English stopwords for visualization only
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = set(ENGLISH_STOP_WORDS)

def plot_common_words(text, title, n=20):

    words = [
        word for word in text.split()
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


# 9. DIMENSIONALITY REDUCTION VISUALIZATION
# ============================================================

# Reduce TF-IDF features to two dimensions using SVD
svd = TruncatedSVD(
    n_components=2,
    random_state=42
)

X_2d = svd.fit_transform(X_train)

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


# 10. FINAL DATASET INFORMATION
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
