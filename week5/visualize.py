"""Training-set exploration, learning curves, and measured Pareto plots."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import tokenize


def dataset_plots(frames, metadata, folder):
    folder.mkdir(parents=True, exist_ok=True)
    train = frames["train"]
    labels = metadata["label_names"]
    counts = train["label"].value_counts().sort_index()
    # 77 intent names remain readable in a separate horizontal plot.
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.15)), layout="constrained")
    ax.barh(range(len(labels)), counts.reindex(range(len(labels)), fill_value=0))
    ax.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Training examples", title=f"{metadata['dataset']} training class counts")
    fig.savefig(folder / "class_counts.pdf")
    plt.close(fig)
    if len(labels) > 20:
        ranked = counts.sort_values(kind="stable")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        for ax, subset, title in zip(
            axes, (ranked.tail(10), ranked.head(10)),
            ("Ten largest classes", "Ten smallest classes"),
        ):
            ax.barh(range(len(subset)), subset.to_numpy(), color="#3c6e8f")
            ax.set_yticks(range(len(subset)),
                          [labels[index].replace("_", " ") for index in subset.index], fontsize=9)
            ax.set(xlabel="Training examples", title=title)
        fig.savefig(folder / "class_overview.pdf")
        plt.close(fig)

    lengths = pd.DataFrame({
        "Words": train["clean_text"].map(lambda text: len(tokenize(text))),
        "Characters (with spaces)": train["clean_text"].str.len(),
        "Characters (without spaces)": train["clean_text"].str.replace(" ", "", regex=False).str.len(),
    })
    lengths.describe().to_csv(folder / "length_statistics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, column in zip(axes, ("Words", "Characters (with spaces)")):
        ax.hist(lengths[column], bins=40, color="#3c6e8f")
        ax.set(xlabel=column, ylabel="Training examples")
    fig.suptitle(f"{metadata['dataset']} cleaned training text lengths")
    fig.savefig(folder / "lengths.pdf")
    plt.close(fig)
    examples = train.groupby("label", sort=True).head(2).copy()
    examples["intent"] = examples["label"].map(dict(enumerate(labels)))
    examples[["text", "intent"]].to_csv(folder / "examples.csv", index=False)


def nondominated(weights, accuracies):
    weights, accuracies = np.asarray(weights), np.asarray(accuracies)
    return np.array([
        not np.any((weights <= count) & (accuracies >= accuracy)
                   & ((weights < count) | (accuracies > accuracy)))
        for count, accuracy in zip(weights, accuracies)
    ])


def result_plots(output):
    for history_path in output.glob("*/*/history.csv"):
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Training")
        axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
        axes[0].legend()
        axes[1].plot(history["epoch"], 100 * history["accuracy"], marker="o")
        axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0, 100))
        from matplotlib.ticker import MaxNLocator
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(history_path.parent / "learning_curves.pdf")
        plt.close(fig)
    for name in ("imdb", "banking77"):
        paths = sorted((output / name).glob("transfer_*/result.json"))
        if not paths:
            continue
        results = [json.loads(path.read_text()) for path in paths]
        weights = np.array([row["trainable_weights"] for row in results])
        for split in ("validation", "test"):
            accuracy = np.array([row["scores"][split]["accuracy"] for row in results])
            frontier = nondominated(weights, accuracy)
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            ax.scatter(weights, accuracy * 100, color="#3c6e8f", label="Configurations")
            order = np.flatnonzero(frontier)[np.argsort(weights[frontier])]
            ax.plot(weights[order], accuracy[order] * 100, "o--", color="#c05640",
                    label="Pareto frontier")
            for x, y, row in zip(weights, accuracy, results):
                offset = {"transfer_upper": (6, 10), "transfer_lower": (6, -16),
                          "transfer_full": (-6, 10)}.get(row["model"], (6, 8))
                ax.annotate(row["model"].replace("transfer_", ""), (x, y * 100),
                            xytext=offset, textcoords="offset points",
                            ha="right" if row["model"] == "transfer_full" else "left")
            ax.set(xscale="log", xlabel="Trainable scalar parameters (log scale)",
                   ylabel=f"{split.title()} accuracy (%)", ylim=(0, 103),
                   title=f"{name}: pretrained layer comparison")
            ax.legend()
            fig.savefig(output / name / f"pareto_{split}.pdf")
            plt.close(fig)
