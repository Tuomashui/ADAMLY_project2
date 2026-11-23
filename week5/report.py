"""Generate report tables and evidence-based comparisons from saved runs."""
import importlib.metadata
import json
import platform
import unicodedata
from pathlib import Path
import pandas as pd
from .common import ROOT, save_json
from .visualize import result_plots, nondominated


def escape(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "replace").decode("ascii")
    return text.translate(str.maketrans({
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
        "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$",
    }))


def figure(path, caption, height="75mm"):
    relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    if not path.exists():
        return ""
    return (r"\begin{figure}[htbp]\centering" + "\n"
            + rf"\includegraphics[width=.96\linewidth,height={height},keepaspectratio]{{{relative}}}"
            + "\n" + rf"\caption{{{escape(caption)}}}\end{{figure}}" + "\n")


def generate_report(output, config):
    result_plots(output)
    rows = []
    sections = [
        r"\section{Recorded run}",
        "This section is generated from saved experiment outputs.",
        ("This is a reduced-data smoke run for checking the pipeline. "
         "Its scores describe only the sampled subsets."
         if config["quick"] else "This run uses the complete official training and test splits, "
         "with a stratified validation holdout from the training data."),
        rf"The random seed is {config['seed']}. The device is \texttt{{{escape(config['device'])}}}. "
        rf"The maximum training budget is {config['epochs']} epochs per neural configuration, "
        rf"with patience {config['patience']}. Batch size is {config['batch_size']}.",
    ]
    for name in ("imdb", "banking77"):
        metadata_path = output / name / "dataset.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        sections.append(rf"\subsection{{{escape(name)} results}}")
        sizes = metadata["split_sizes"]
        sections.append(
            f"Training, validation, and test sizes are {sizes['train']}, "
            f"{sizes['validation']}, and {sizes['test']}, respectively."
        )
        results = [json.loads(path.read_text()) for path in sorted((output / name).glob("*/result.json"))]
        if not results:
            sections.append("Training results have not yet been generated for this dataset.")
        else:
            sections.extend([
                r"\begin{table}[htbp]\centering\small",
                r"\caption{Measured performance. Accuracies and macro-F1 are percentages. Time includes neural validation and checkpoint selection.}",
                r"\begin{tabular}{lrrrrr}\toprule",
                r"Model & Weights & Val.\ acc. & Test acc. & Test F1 & Seconds\\\midrule",
            ])
            for row in results:
                val, test = row["scores"]["validation"], row["scores"]["test"]
                sections.append(
                    f"{escape(row['model'])} & {row['trainable_weights']:,} & "
                    f"{100*val['accuracy']:.2f} & {100*test['accuracy']:.2f} & "
                    f"{100*test['macro_f1']:.2f} & {row['training_seconds']:.1f}" + r"\\"
                )
                rows.append({
                    "dataset": name, "model": row["model"],
                    "trainable_weights": row["trainable_weights"],
                    "validation_accuracy": val["accuracy"], "test_accuracy": test["accuracy"],
                    "test_macro_f1": test["macro_f1"], "training_seconds": row["training_seconds"],
                })
            sections.append(r"\bottomrule\end{tabular}\end{table}")
            by_model = {row["model"]: row for row in results}
            if {"baseline", "scratch"} <= by_model.keys():
                delta = 100 * (by_model["scratch"]["scores"]["test"]["accuracy"]
                               - by_model["baseline"]["scores"]["test"]["accuracy"])
                sections.append(
                    f"The randomly initialized Transformer changes test accuracy by {delta:+.2f} "
                    "percentage points relative to the TF--IDF baseline. "
                    "The baseline directly models word and bigram evidence, while the small "
                    "Transformer learns its embeddings and contextual representation from the "
                    "task training data. Input truncation and the fixed training budget may "
                    "also affect the comparison. These possible explanations are not isolated "
                    "causally by this experiment."
                )
                if name == "imdb":
                    sections.append(
                        "For IMDb, the baseline represents the complete cleaned review, while "
                        "the Transformer sees only the initial token window. Sentiment evidence "
                        "later in a review can therefore be unavailable to the Transformer. "
                        "Its embeddings are learned from task data without language-model pretraining."
                    )
                else:
                    sections.append(
                        f"BANKING77 has {metadata['classes']} classes and only "
                        f"{sizes['train']/metadata['classes']:.1f} training queries per class on average. "
                        "Short domain-specific queries can provide useful word and bigram cues "
                        "to the baseline, while the random Transformer must learn representations "
                        "and a 77-way decision rule within the fixed epoch budget. The recorded "
                        "truncation fractions help distinguish input-length effects from these factors."
                    )
            transfers = [row for row in results if row["model"].startswith("transfer_")]
            if transfers:
                best = max(transfers, key=lambda row: row["scores"]["validation"]["accuracy"])
                sections.append(
                    f"The validation-selected fine-tuning configuration is "
                    rf"\texttt{{{escape(best['model'])}}}, with test accuracy "
                    f"{100*best['scores']['test']['accuracy']:.2f}\\%. "
                    "Selection uses validation accuracy; test results are reported afterward."
                )
                frontier = nondominated(
                    [row["trainable_weights"] for row in transfers],
                    [row["scores"]["validation"]["accuracy"] for row in transfers],
                )
                names = ", ".join(escape(row["model"]) for row, keep in zip(transfers, frontier) if keep)
                sections.append(f"The measured validation Pareto frontier contains {names}.")
                head = by_model.get("transfer_head")
                full = by_model.get("transfer_full")
                if head and full:
                    gain = 100 * (full["scores"]["test"]["accuracy"] - head["scores"]["test"]["accuracy"])
                    sections.append(
                        f"Full fine-tuning changes test accuracy by {gain:+.2f} percentage points "
                        "relative to head-only training. The upper- and lower-block runs each "
                        "unfreeze one encoder block, enabling a comparison at the same block budget."
                    )
                upper, lower = by_model.get("transfer_upper"), by_model.get("transfer_lower")
                if upper and lower:
                    delta = 100 * (upper["scores"]["validation"]["accuracy"]
                                   - lower["scores"]["validation"]["accuracy"])
                    test_delta = 100 * (upper["scores"]["test"]["accuracy"]
                                        - lower["scores"]["test"]["accuracy"])
                    sections.append(
                        f"Unfreezing the upper block changes validation accuracy by {delta:+.2f} "
                        "percentage points relative to unfreezing the lower block. "
                        f"The corresponding test difference is {test_delta:+.2f} percentage points. "
                        "This quantifies layer-position sensitivity for this checkpoint and training budget."
                    )
                if full and "scratch" in by_model:
                    delta = 100 * (full["scores"]["test"]["accuracy"]
                                   - by_model["scratch"]["scores"]["test"]["accuracy"])
                    sections.append(
                        f"The fully fine-tuned pretrained model changes test accuracy by {delta:+.2f} "
                        "percentage points relative to the random Transformer. Pretraining and "
                        "architecture both differ in this comparison, so their effects are not isolated."
                    )
            for row in results:
                extra = row.get("truncated_fraction")
                if extra:
                    sections.append(
                        rf"For \texttt{{{escape(row['model'])}}}, "
                        f"{100*extra['train']:.1f}\\% of training texts exceed the input limit. "
                        + (f"The selected checkpoint is from epoch {row['best_epoch']}."
                           if "best_epoch" in row else "")
                    )
            for row in results:
                prediction_path = output / name / row["model"] / "test_predictions.csv"
                predictions = pd.read_csv(prediction_path)
                errors = predictions[predictions.true_label != predictions.predicted_label].head(2)
                if not errors.empty:
                    sections.append(r"\par\medskip\noindent\begin{minipage}{\linewidth}")
                    sections.append(rf"\paragraph{{Example errors: {escape(row['model'])}}}")
                    sections.append(r"\begin{itemize}")
                    for _, error in errors.iterrows():
                        true = metadata["label_names"][int(error.true_label)]
                        pred = metadata["label_names"][int(error.predicted_label)]
                        text = str(error.text).replace("\n", " ")[:180]
                        sections.append(r"\item " + escape(f'True: {true}; predicted: {pred}. "{text}..."'))
                    sections.append(r"\end{itemize}")
                    sections.append(r"\end{minipage}\par")
        counts_plot = "class_overview.pdf" if name == "banking77" else "class_counts.pdf"
        caption = ("BANKING77 ten largest and ten smallest training classes. The complete "
                   "77-class plot is saved separately as class_counts.pdf."
                   if name == "banking77" else "IMDb training class frequencies.")
        sections.append(figure(output / name / counts_plot, caption, "90mm"))
        sections.append(figure(output / name / "lengths.pdf", f"{name} cleaned training text lengths."))
        sections.append(figure(output / name / "scratch" / "learning_curves.pdf",
                               f"{name} randomly initialized Transformer learning curves."))
        sections.append(figure(output / name / "pareto_validation.pdf",
                               f"{name} validation Pareto frontier used for configuration comparison."))
        sections.append(figure(output / name / "pareto_test.pdf",
                               f"{name} test accuracy versus trained weights; descriptive evaluation."))
        sections.append(r"\clearpage")
    sections.extend([
        r"\section{Conclusions and limitations}",
        "Results use one random seed, one fixed hyperparameter budget, and a compact "
        "pretrained BERT checkpoint. Accuracy differences do not establish that one "
        "architecture is universally better. Broader learning-rate searches, repeated "
        "seeds, and larger checkpoints could change the ranking. Fine-tuning importance "
        "is assessed through the tested frozen-layer configurations.",
    ])
    for name in ("imdb", "banking77"):
        measured = [row for row in rows if row["dataset"] == name]
        if measured:
            winner = max(measured, key=lambda row: row["validation_accuracy"])
            sections.append(
                f"For {escape(name)}, the highest validation accuracy among completed models "
                rf"is obtained by \texttt{{{escape(winner['model'])}}}. "
                f"Its held-out test accuracy is {100*winner['test_accuracy']:.2f}\\%."
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.tex").write_text("\n\n".join(sections), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "scikit-learn", "pandas",
                             "numpy", "matplotlib", "tokenizers", "huggingface-hub")}
    save_json(output / "environment.json", {"python": platform.python_version(), **versions})
    print(f"Report tables: {output / 'results.tex'}", flush=True)
