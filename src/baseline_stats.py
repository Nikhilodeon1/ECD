"""Baseline accuracy by source for the Claude benchmark: Wilson CIs, the Fisher gap,
and a length-adjusted logistic fit (accuracy ~ source + log evidence words).
One row per case, so a plain logit is the right model (no clustering needed).
Aggregates -> results/baseline_stats.json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from paths import DATA, RESULTS, read_jsonl
from significance import two_proportion_fisher

MAX_WORDS = 800


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DATA / "eval_results_claude.jsonl"))
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "baseline_stats.json"))
    a = ap.parse_args()
    base = pd.DataFrame(read_jsonl(a.baseline))
    samp = pd.DataFrame(read_jsonl(a.sample))[["id", "original_note"]]
    samp["words"] = samp["original_note"].str.split().str.len().clip(upper=MAX_WORDS)
    df = base.merge(samp[["id", "words"]], left_on="case_id", right_on="id")
    df["correct"] = df["correct"].astype(int)
    df["log_words"] = np.log(df["words"])
    out = {"script": "src/baseline_stats.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(df), "by_source": {}}
    for s, g in df.groupby("source"):
        k, n = int(g["correct"].sum()), len(g)
        out["by_source"][s] = {"accuracy": round(k / n, 4), "k": k, "n": n, "wilson_95": wilson(k, n),
                               "evidence_words_median": float(g["words"].median())}
    srcs = sorted(out["by_source"])
    if len(srcs) == 2:
        a_, b_ = srcs
        A, B = out["by_source"][a_], out["by_source"][b_]
        out["fisher_gap"] = {"comparison": f"{a_} vs {b_}", **two_proportion_fisher(A["k"], A["n"], B["k"], B["n"])}
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        m = smf.glm("correct ~ C(source, Treatment('medqa')) + log_words", data=df, family=sm.families.Binomial()).fit()
        ci = m.conf_int()
        key = [k for k in m.params.index if k.startswith("C(source")][0]
        out["length_adjusted_logit"] = {
            "odds_ratio_mimic_vs_medqa": round(float(np.exp(m.params[key])), 3),
            "ci_95": [round(float(np.exp(ci.loc[key, 0])), 3), round(float(np.exp(ci.loc[key, 1])), 3)],
            "p": float(m.pvalues[key]),
            "log_words_odds_ratio": round(float(np.exp(m.params["log_words"])), 3),
            "survives_length_adjustment": bool(not (ci.loc[key, 0] <= 0 <= ci.loc[key, 1])),
        }
    RESULTS.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
