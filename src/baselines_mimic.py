"""T4c: what accuracy do trivial predictors get on the MIMIC half of the eval sample?

* constant: always answer the most frequent ground-truth diagnosis in the sample
* nearest neighbour: TF-IDF over the evidence text, trained on MIMIC cases NOT in
  the sample (a random --train-n), answer = the neighbour's ground-truth title
* random class: expected accuracy of drawing a diagnosis from the training label
  distribution

Scored two ways: strict normalized string match, and (optional, --judge claude)
the same locked Claude judge as the headline numbers. Reads the full cases file
(MIMIC-derived text: run only where ECD_ALLOW_MIMIC=1). Aggregates only ->
results/mimic_baselines.json.
"""
import argparse
import json
import random
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from judge import Judge, normalize
from paths import ALLOW_MIMIC, DATA, RESULTS, read_jsonl

WORDS = 300  # evidence words used for the TF-IDF features


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="full cases.jsonl (MIMIC + MedQA constructed pool)")
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--train-n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge", choices=["none", "claude"], default="none")
    ap.add_argument("--out", default=str(RESULTS / "mimic_baselines.json"))
    a = ap.parse_args()
    if not ALLOW_MIMIC:
        raise SystemExit("MIMIC baselines need ECD_ALLOW_MIMIC=1 (cleared host only)")

    sample = [c for c in read_jsonl(a.sample) if c["source"] == "mimic-iv-note"]
    sample_ids = {c["id"] for c in sample}
    rng = random.Random(a.seed)
    train, seen = [], 0
    with open(a.cases, encoding="utf-8") as f:  # reservoir sample, streaming (file is ~2 GB)
        for line in f:
            if '"mimic-iv-note"' not in line[:400] and "MIMIC-" not in line[:80]:
                continue
            r = json.loads(line)
            if r["source"] != "mimic-iv-note" or r["id"] in sample_ids:
                continue
            seen += 1
            item = (" ".join(r["original_note"].split()[:WORDS]), r["diagnosis_ground_truth"])
            if len(train) < a.train_n:
                train.append(item)
            else:
                j = rng.randrange(seen)
                if j < a.train_n:
                    train[j] = item
    print(f"{len(sample)} MIMIC sample cases; training pool {len(train)} of {seen} other MIMIC cases")

    labels = [t[1] for t in train]
    dist = Counter(normalize(x) for x in labels)
    top_label_raw = Counter(labels).most_common(1)[0][0]
    tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform([t[0] for t in train])
    Xq = tf.transform([" ".join(c["original_note"].split()[:WORDS]) for c in sample])
    nn_idx = np.asarray((Xq @ Xtr.T).argmax(axis=1)).ravel()
    nn_pred = [labels[i] for i in nn_idx]

    gt = [c["diagnosis_ground_truth"] for c in sample]
    sample_dist = Counter(normalize(g) for g in gt)
    const_in_sample = Counter(gt).most_common(1)[0][0]  # most frequent in the sample itself (the favorable choice)
    n = len(sample)

    def strict(preds):
        return sum(normalize(p) == normalize(g) for p, g in zip(preds, gt))

    out = {"script": "src/baselines_mimic.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_mimic_sample": n,
           "train_pool": len(train), "train_n_unique_labels": len(dist),
           "strict_string_match": {}, "note": "strict match is a lower bound on a judged score"}
    for name, preds in (("constant_most_frequent_in_sample", [const_in_sample] * n),
                        ("constant_most_frequent_in_training_pool", [top_label_raw] * n),
                        ("tfidf_nearest_neighbour", nn_pred)):
        k = strict(preds)
        out["strict_string_match"][name] = {"accuracy": round(k / n, 4), "k": k, "n": n, "wilson_95": wilson(k, n)}
    # expected accuracy of sampling labels from the training distribution
    tot = sum(dist.values())
    exp = sum((dist[l] / tot) * (sample_dist.get(l, 0) / n) for l in dist)
    out["strict_string_match"]["random_label_from_training_distribution_expected"] = {"accuracy": round(exp, 4)}

    if a.judge == "claude":
        from llm_clients import AnthropicClient

        client = AnthropicClient()
        judge = Judge(client, "claude-binary", str(DATA / "judge_cache_claude.jsonl"))
        out["claude_judged"] = {}
        for name, preds in (("constant_most_frequent_in_sample", [const_in_sample] * n), ("tfidf_nearest_neighbour", nn_pred)):
            k = sum(judge.compare(f"mimic-baseline-{name}", c["id"], c["diagnosis_ground_truth"], p, kind="grade")["match"]
                    for c, p in zip(sample, preds))
            out["claude_judged"][name] = {"accuracy": round(k / n, 4), "k": k, "n": n, "wilson_95": wilson(k, n)}
        out["claude_judge_calls"] = judge.calls
        client.dump_meta(str(RESULTS / "api_manifest.jsonl"))
    RESULTS.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
