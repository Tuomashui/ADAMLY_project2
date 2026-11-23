"""Week 1 TF-IDF logistic regression, extended to 77 classes."""
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .common import tokenize, save_evaluation, save_json


def run_baseline(frames, metadata, folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,
        max_features=20000, ngram_range=(1, 2), min_df=2,
        max_df=0.95, sublinear_tf=True,
    )
    features = {"train": vectorizer.fit_transform(frames["train"]["clean_text"])}
    for split in ("validation", "test"):
        features[split] = vectorizer.transform(frames[split]["clean_text"])
    # liblinear preserves the Week 1 binary model. lbfgs supports multinomial loss.
    classifier = LogisticRegression(
        C=1.0, max_iter=1000, solver="liblinear" if metadata["classes"] == 2 else "lbfgs",
        random_state=config["seed"],
    )
    classifier.fit(features["train"], frames["train"]["label"])
    elapsed = time.perf_counter() - start
    probabilities = {split: classifier.predict_proba(features[split])
                     for split in ("validation", "test")}
    scores = save_evaluation(folder, frames, probabilities, metadata["label_names"])
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, folder / "model.joblib")
    result = {
        "dataset": metadata["dataset"], "model": "baseline", "scores": scores,
        "trainable_weights": int(classifier.coef_.size + classifier.intercept_.size),
        "training_seconds": elapsed, "features": len(vectorizer.vocabulary_),
        "solver": classifier.solver,
    }
    save_json(folder / "result.json", result)
    return result
