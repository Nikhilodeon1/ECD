"""T8: statistics upgrade for the Claude benchmark.

* GEE logistic (exchangeable, cluster = case_id):
      drifted ~ source + category + log(evidence_words)
  odds ratios, 95% CIs, and a joint Wald test for category.
* Cluster-bootstrap CIs (B=10000, cluster = case) for every rate and for the
  MIMIC - MedQA risk difference (resampled within source).
* Drift by evidence-length quartile, per source.
* A plain statement of whether the source gap survives length adjustment.

Existing tests (Fisher, Mann-Whitney, Cochran's Q, McNemar+Holm) stay in
week9_significance.py. Aggregates only -> results/stats_v2.json.
Without ECD_ALLOW_MIMIC=1 only MedQA rows load, so the source term is skipped.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from paths import DATA, RESULTS, read_jsonl

MAX_WORDS = 800  # prompts truncate evidence at 800 words (src/prompts.py)


def case_table(trials: pd.DataFrame, by: list[str]) -> dict:
    """per-case (drifted, total) arrays for each group in `by`."""
    out = {}
    for key, g in trials.groupby(by, dropna=False):
        per_case = g.groupby("case_id")["drifted"].agg(["sum", "count"])
        out[key if isinstance(key, tuple) else (key,)] = (per_case["sum"].to_numpy(float), per_case["count"].to_numpy(float))
    return out


def boot_rate(d: np.ndarray, t: np.ndarray, B: int, rng) -> dict:
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    rates = d[idx].sum(1) / t[idx].sum(1)
    return {"rate": round(float(d.sum() / t.sum()), 4), "ci_95": [round(float(np.percentile(rates, 2.5)), 4),
            round(float(np.percentile(rates, 97.5)), 4)], "n_trials": int(t.sum()), "n_cases": n}


def boot_diff(a, b, B, rng) -> dict:
    (da, ta), (db, tb) = a, b
    ia = rng.integers(0, len(da), size=(B, len(da)))
    ib = rng.integers(0, len(db), size=(B, len(db)))
    diff = da[ia].sum(1) / ta[ia].sum(1) - db[ib].sum(1) / tb[ib].sum(1)
    est = da.sum() / ta.sum() - db.sum() / tb.sum()
    return {"diff": round(float(est), 4), "ci_95": [round(float(np.percentile(diff, 2.5)), 4),
            round(float(np.percentile(diff, 97.5)), 4)]}


def gee(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    terms = ["C(category)", "log_words"]
    if df["source"].nunique() > 1:
        terms.insert(0, "C(source, Treatment('medqa'))")
    model = smf.gee("drifted ~ " + " + ".join(terms), groups="case_id", data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    ci = res.conf_int()
    table = {name: {"odds_ratio": round(float(np.exp(res.params[name])), 3),
                    "ci_95": [round(float(np.exp(ci.loc[name, 0])), 3), round(float(np.exp(ci.loc[name, 1])), 3)],
                    "p": float(res.pvalues[name])} for name in res.params.index if name != "Intercept"}
    cat = [i for i, n in enumerate(res.params.index) if n.startswith("C(category)")]
    b = res.params.values[cat]
    V = res.cov_params().values[np.ix_(cat, cat)]
    wald = float(b @ np.linalg.solve(V, b))
    return {"formula": "drifted ~ " + " + ".join(terms), "cov_struct": "exchangeable",
            "n_trials": int(res.nobs), "n_cases": int(df["case_id"].nunique()), "coefficients": table,
            "category_joint_wald": {"chi2": round(wald, 3), "df": len(cat), "p": float(chi2.sf(wald, len(cat)))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-results", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "stats_v2.json"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    trials = pd.DataFrame(read_jsonl(args.drift_results))
    sample = pd.DataFrame(read_jsonl(args.sample))[["id", "original_note"]]
    sample["evidence_words"] = sample["original_note"].str.split().str.len().clip(upper=MAX_WORDS)
    df = trials.merge(sample[["id", "evidence_words"]], left_on="case_id", right_on="id", how="left")
    assert df["evidence_words"].notna().all(), "drift trials with no matching eval_sample case"
    df["drifted"] = df["drifted"].astype(int)
    df["log_words"] = np.log(df["evidence_words"])
    sources = sorted(df["source"].unique())

    out = {"script": "src/stats_v2.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "B": args.B,
           "sources_present": sources, "n_trials": len(df), "n_cases": int(df["case_id"].nunique())}

    overall = case_table(df.assign(all="all"), ["all"])
    out["drift_rate_overall"] = boot_rate(*overall[("all",)], args.B, rng)
    by_src = case_table(df, ["source"])
    out["drift_rate_by_source"] = {k[0]: boot_rate(*v, args.B, rng) for k, v in by_src.items()}
    out["drift_rate_by_category"] = {k[0]: boot_rate(*v, args.B, rng) for k, v in case_table(df, ["category"]).items()}
    out["drift_rate_by_source_and_category"] = {f"{k[0]}|{k[1]}": boot_rate(*v, args.B, rng)
                                                for k, v in case_table(df, ["source", "category"]).items()}
    if {"mimic-iv-note", "medqa"} <= set(sources):
        out["risk_difference_mimic_minus_medqa"] = boot_diff(by_src[("mimic-iv-note",)], by_src[("medqa",)], args.B, rng)

    # length: quartiles within each source (per case, since evidence is per case)
    quart = {}
    for s, g in df.groupby("source"):
        case_len = g.groupby("case_id")["evidence_words"].first()
        q = pd.qcut(case_len, 4, labels=False, duplicates="drop")
        g = g.assign(quartile=g["case_id"].map(q))
        quart[s] = {}
        for qi, gq in g.groupby("quartile"):
            lo, hi = case_len[q == qi].min(), case_len[q == qi].max()
            d, t = case_table(gq.assign(all="all"), ["all"])[("all",)]
            quart[s][f"Q{int(qi) + 1} ({int(lo)}-{int(hi)} words)"] = boot_rate(d, t, min(args.B, 2000), rng)
    out["drift_by_evidence_length_quartile_within_source"] = quart
    out["evidence_words_by_source"] = {s: {"median": float(g.groupby("case_id")["evidence_words"].first().median()),
                                           "iqr": [float(g.groupby("case_id")["evidence_words"].first().quantile(.25)),
                                                   float(g.groupby("case_id")["evidence_words"].first().quantile(.75))]}
                                       for s, g in df.groupby("source")}

    out["gee"] = gee(df)
    src_key = next((k for k in out["gee"]["coefficients"] if k.startswith("C(source")), None)
    if src_key:
        c = out["gee"]["coefficients"][src_key]
        survives = not (c["ci_95"][0] <= 1.0 <= c["ci_95"][1])
        out["source_gap_survives_length_adjustment"] = {
            "survives": survives, "adjusted_odds_ratio_mimic_vs_medqa": c["odds_ratio"], "ci_95": c["ci_95"],
            "statement": ("The MIMIC > MedQA drift gap persists after adjusting for category and log evidence length."
                          if survives else
                          "STOP: the source gap does NOT survive length adjustment (CI includes 1). Contradicts a headline claim."),
        }
    else:
        out["source_gap_survives_length_adjustment"] = {"survives": None, "statement": "only one source loaded; run on the cleared host with ECD_ALLOW_MIMIC=1"}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("drift_rate_overall", "drift_rate_by_source", "gee",
                                          "source_gap_survives_length_adjustment") if k in out}, indent=2))


if __name__ == "__main__":
    main()
