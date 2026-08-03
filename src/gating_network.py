"""Week 8: gating network -- predicts drift risk from an adversarial note.

Trained on the 960 labeled trials from Week 6 (src/eval_drift.py output):
given an adversarial note's text + category, predict whether it caused
Claude to drift. The predicted probability is meant to be used as a
per-case gate: scale ECD's alpha by predicted risk (e.g. alpha_used =
predicted_risk * alpha_max) instead of applying the same fixed alpha to
every case regardless of how dangerous the specific note actually looks.

Kept deliberately simple -- TF-IDF + logistic regression, not a deep model.
960 labeled examples is a small dataset; a heavier model would likely
overfit, and the point here is a usable per-case gating signal, not a
novel classifier architecture.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_training_data(drift_results_path: str) -> tuple[list[str], list[int]]:
    with open(drift_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    # fold category into the text itself as a cheap way to give the
    # classifier that signal without a separate one-hot feature matrix
    texts = [f"[{r['category']}] {r['adversarial_note']}" for r in results]
    labels = [int(r["drifted"]) for r in results]
    return texts, labels


def train_gating_network(texts: list[str], labels: list[int], seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train_vec, y_train)

    preds = clf.predict(x_test_vec)
    probs = clf.predict_proba(x_test_vec)[:, 1]

    metrics = {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_accuracy": round(accuracy_score(y_test, preds), 3),
        "test_auc": round(roc_auc_score(y_test, probs), 3),
        "baseline_majority_class_accuracy": round(max(np.mean(y_test), 1 - np.mean(y_test)), 3),
    }
    return clf, vectorizer, metrics


def predict_drift_risk(clf, vectorizer, adversarial_note: str, category: str) -> float:
    """Returns P(this note causes drift) -- use to scale alpha per case,
    e.g. alpha_used = predict_drift_risk(...) * alpha_max."""
    text = f"[{category}] {adversarial_note}"
    return float(clf.predict_proba(vectorizer.transform([text]))[0, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-results", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument("--out-model", default="../data/processed/gating_network.pkl")
    parser.add_argument("--out-metrics", default="../data/processed/gating_network_metrics.json")
    args = parser.parse_args()

    texts, labels = load_training_data(args.drift_results)
    print(f"loaded {len(texts)} labeled trials, {sum(labels)} drifted ({100*sum(labels)/len(labels):.1f}%)")

    clf, vectorizer, metrics = train_gating_network(texts, labels)
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"clf": clf, "vectorizer": vectorizer}, f)
    print(f"saved model -> {out_path}")

    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics -> {args.out_metrics}")

    # sanity check: is the model actually beating the trivial baseline of
    # always predicting the majority class? if not, it hasn't learned
    # anything useful and shouldn't be trusted as a gating signal
    if metrics["test_accuracy"] <= metrics["baseline_majority_class_accuracy"] + 0.02:
        print(
            "WARNING: barely beats (or loses to) the majority-class baseline -- "
            "this gating signal may not be worth using yet"
        )
