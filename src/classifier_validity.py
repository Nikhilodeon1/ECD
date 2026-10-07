"""T9: is the drift-risk classifier (note text -> drifted?) valid?

GroupKFold by case_id (a case's five trials never straddle train/test), pooled
out-of-fold AUC with a cluster bootstrap CI (cluster = case), a within-category
permutation test, and baselines: category only, note length only, majority.
Run again with diagnosis tokens masked to check for leakage.

Rule (set before running): keep the classifier in the main text only if the
lower CI bound of its AUC is >= 0.55 AND it beats the category-only baseline;
otherwise it goes to the appendix and the abstract mention is cut.
"""
import argparse
import json
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from paths import DATA, RESULTS, read_jsonl


def mask_diagnosis(note: str, diagnosis: str) -> str:
    toks = {t for t in re.findall(r"[a-z]+", diagnosis.lower()) if len(t) >= 4}
    return re.sub(r"[A-Za-z]+", lambda m: "DX" if m.group(0).lower() in toks else m.group(0), note)


def make_text_model():
    return make_pipeline(
        TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def oof_scores(X, y, groups, make_model, k=5):
    scores = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_model()
        m.fit([X[i] for i in tr] if isinstance(X, list) else X[tr], y[tr])
        Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
        scores[te] = m.predict_proba(Xte)[:, 1]
    return scores


def cluster_bootstrap_auc(y, s, groups, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        ix = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[ix])) == 2:
            out.append(roc_auc_score(y[ix], s[ix]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def within_category_permutation(y, cats, groups, texts, n_perm, seed=42):
    """Shuffle labels among trials of the same category (keeps base rates, breaks
    the note-text/label link); returns the null AUC list."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for c in np.unique(cats):
            ix = np.where(cats == c)[0]
            yp[ix] = rng.permutation(y[ix])
        null.append(roc_auc_score(yp, oof_scores(texts, yp, groups, make_text_model)))
    return null


def evaluate(texts, y, cats, groups, lengths, n_perm, B):
    res = {}
    s_text = oof_scores(texts, y, groups, make_text_model)
    auc = roc_auc_score(y, s_text)
    lo, hi = cluster_bootstrap_auc(y, s_text, groups, B)
    null = within_category_permutation(y, cats, groups, texts, n_perm) if n_perm else []
    res["text_model"] = {"auc": round(auc, 3), "ci_95": [round(lo, 3), round(hi, 3)],
                         "perm_p": round((1 + sum(a >= auc for a in null)) / (1 + len(null)), 4) if null else None,
                         "n_perm": len(null)}
    cat_oh = np.array([[c == k for k in sorted(set(cats))] for c in cats], dtype=float)
    s_cat = oof_scores(cat_oh, y, groups, lambda: LogisticRegression(max_iter=1000, class_weight="balanced"))
    s_len = oof_scores(lengths.reshape(-1, 1).astype(float), y, groups,
                       lambda: LogisticRegression(max_iter=1000, class_weight="balanced"))
    for name, s in (("category_only", s_cat), ("note_length_only", s_len)):
        a = roc_auc_score(y, s)
        l, h = cluster_bootstrap_auc(y, s, groups, B)
        res[name] = {"auc": round(a, 3), "ci_95": [round(l, 3), round(h, 3)]}
    res["majority"] = {"auc": 0.5}
    # paired cluster bootstrap of text minus category-only
    rng = np.random.default_rng(7)
    ids = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ids}
    diffs = []
    for _ in range(B):
        ix = np.concatenate([idx_by[g] for g in rng.choice(ids, size=len(ids), replace=True)])
        if len(np.unique(y[ix])) == 2:
            diffs.append(roc_auc_score(y[ix], s_text[ix]) - roc_auc_score(y[ix], s_cat[ix]))
    res["text_minus_category_only"] = {"diff": round(auc - roc_auc_score(y, s_cat), 3),
                                       "ci_95": [round(float(np.percentile(diffs, 2.5)), 3),
                                                 round(float(np.percentile(diffs, 97.5)), 3)]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-results", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "classifier_validity.json"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()

    trials = read_jsonl(args.drift_results)
    y = np.array([int(t["drifted"]) for t in trials])
    cats = np.array([t["category"] for t in trials])
    groups = np.array([t["case_id"] for t in trials])
    lengths = np.array([len(t["adversarial_note"].split()) for t in trials])
    raw = [f"[{t['category']}] {t['adversarial_note']}" for t in trials]
    masked = [f"[{t['category']}] {mask_diagnosis(t['adversarial_note'], t['diagnosis_before'])}" for t in trials]

    out = {"script": "src/classifier_validity.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_trials": len(trials), "n_cases": len(set(groups.tolist())), "drift_rate": round(float(y.mean()), 3),
           "sources": sorted({t["source"] for t in trials}),
           "unmasked": evaluate(raw, y, cats, groups, lengths, args.n_perm, args.B),
           "diagnosis_masked": evaluate(masked, y, cats, groups, lengths, 0, args.B)}
    for key in ("unmasked", "diagnosis_masked"):
        r = out[key]
        keep = r["text_model"]["ci_95"][0] >= 0.55 and r["text_minus_category_only"]["ci_95"][0] > 0
        r["keep_in_main_text"] = bool(keep)
    out["recommendation"] = (
        "keep in main text" if out["unmasked"]["keep_in_main_text"] and out["diagnosis_masked"]["keep_in_main_text"]
        else "move to appendix and cut from the abstract"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
