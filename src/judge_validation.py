"""T4a/T4b: judge validation on the Claude benchmark.

Re-judge every baseline judgment (n=300 on the cleared host) and every drift
judgment (n=994) with a second judge, then report agreement with the recorded
Claude verdicts (Cohen's kappa, cluster-bootstrap CI by case, by source and task)
and recompute the headline statistics under the second judge.

  --judge open    open-weight instruct model, same binary_v1 rubric text, blind to
                  Claude's verdict (T4a)
  --judge tiered  Claude with the tiered rubric equivalent/narrower/broader/different;
                  headline numbers under strict (equivalent) and lenient
                  (equivalent+narrower+broader) readings (T4b, paid)

Caches under data/ (text); aggregates -> results/judge_validation_<name>.json.
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score

from judge import Judge
from paths import DATA, RESULTS, read_jsonl
from significance import cochrans_q, mann_whitney_u, two_proportion_fisher


def kappa_ci(a, b, clusters, B=2000, seed=42):
    a, b, clusters = np.asarray(a, int), np.asarray(b, int), np.asarray(clusters)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return {"kappa": None, "ci_95": [None, None], "note": "one rater is constant"}
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in ids}
    ks = []
    for _ in range(B):
        ix = np.concatenate([idx_by[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        if len(set(a[ix])) == 2 and len(set(b[ix])) == 2:
            ks.append(cohen_kappa_score(a[ix], b[ix]))
    return {"kappa": round(float(cohen_kappa_score(a, b)), 3),
            "ci_95": [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)] if ks else [None, None]}


def agreement(rows, flag_claude, flag_alt):
    out = {}
    for scope in ["all"] + sorted({r["source"] for r in rows}):
        sel = [r for r in rows if scope == "all" or r["source"] == scope]
        a, b = [r[flag_claude] for r in sel], [r[flag_alt] for r in sel]
        out[scope] = {"n": len(sel), "raw_agreement": round(float(np.mean(np.array(a) == np.array(b))), 4) if sel else None,
                      **kappa_ci(a, b, [r["case_id"] for r in sel])}
    return out


def headline(baseline, drift, acc_key, drift_key):
    """Accuracy gap (Fisher, one row per case), drift gap (case-level Mann-Whitney),
    category omnibus (Cochran's Q over cases with all five categories)."""
    res = {}
    srcs = sorted({r["source"] for r in baseline})
    res["baseline_accuracy"] = {s: round(float(np.mean([r[acc_key] for r in baseline if r["source"] == s])), 4) for s in srcs}
    if len(srcs) == 2:
        a, b = srcs
        ka, na = sum(r[acc_key] for r in baseline if r["source"] == a), sum(1 for r in baseline if r["source"] == a)
        kb, nb = sum(r[acc_key] for r in baseline if r["source"] == b), sum(1 for r in baseline if r["source"] == b)
        res["accuracy_gap_fisher"] = {"comparison": f"{a} vs {b}", **two_proportion_fisher(ka, na, kb, nb)}
    by_case = defaultdict(lambda: [0, 0, None])
    for r in drift:
        c = by_case[r["case_id"]]
        c[0] += int(r[drift_key]); c[1] += 1; c[2] = r["source"]
    prop = defaultdict(list)
    for d, n, s in by_case.values():
        prop[s].append(d / n)
    res["drift_rate_by_source_mean_case_proportion"] = {s: round(float(np.mean(v)), 4) for s, v in prop.items()}
    if len(prop) == 2:
        a, b = sorted(prop)
        res["drift_gap_mann_whitney"] = {"comparison": f"{a} vs {b}", **mann_whitney_u(prop[a], prop[b])}
    cats = sorted({r["category"] for r in drift})
    mat = defaultdict(dict)
    for r in drift:
        mat[r["case_id"]][r["category"]] = bool(r[drift_key])
    full = [[row[c] for c in cats] for row in mat.values() if all(c in row for c in cats)]
    res["category_cochrans_q"] = {**cochrans_q(full), "n_cases": len(full), "categories": cats} if full else None
    res["drift_rate_by_category"] = {c: round(float(np.mean([r[drift_key] for r in drift if r["category"] == c])), 4) for c in cats}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["open", "tiered"], required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct", help="open judge")
    ap.add_argument("--baseline", default=str(DATA / "eval_results_claude.jsonl"))
    ap.add_argument("--drift", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base_rows = read_jsonl(a.baseline)
    drift_rows = read_jsonl(a.drift)
    if a.limit:
        base_rows, drift_rows = base_rows[: a.limit], drift_rows[: a.limit]
    print(f"{len(base_rows)} baseline + {len(drift_rows)} drift judgments to re-judge")
    if a.dry_run:
        print("dry run: no calls"); return

    if a.judge == "open":
        from llm_clients import OpenJudgeClient
        backend, jid, rubric = OpenJudgeClient(a.model_name), a.model_name.split("/")[-1], "binary_v1"
    else:
        from llm_clients import AnthropicClient
        backend, jid, rubric = AnthropicClient(), "claude-tiered", "tiered_v1"
    judge = Judge(backend, jid, str(DATA / f"judge_cache_{jid}.jsonl"), rubric=rubric)

    base = []
    for r in base_rows:
        v = judge.compare("baseline", r["case_id"], r["ground_truth"], r["predicted"], kind="grade")
        base.append({"case_id": r["case_id"], "source": r["source"], "claude": bool(r["correct"]), "label": v["label"],
                     "strict": v["label"] == "equivalent", "lenient": v["label"] in ("equivalent", "narrower", "broader")})
    drift = []
    for r in drift_rows:
        v = judge.compare("drift", f"{r['case_id']}|{r['category']}", r["diagnosis_before"], r["diagnosis_after"], kind="same")
        drift.append({"case_id": r["case_id"], "source": r["source"], "category": r["category"], "claude": not r["drifted"],  # claude: same answer
                      "label": v["label"], "strict": v["label"] == "equivalent",
                      "lenient": v["label"] in ("equivalent", "narrower", "broader")})
    for r in drift:
        r["drift_strict"] = not r["strict"]
        r["drift_lenient"] = not r["lenient"]
        r["drift_claude"] = not r["claude"]

    out = {"script": "src/judge_validation.py", "judge": jid, "rubric": rubric, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_baseline": len(base), "n_drift": len(drift), "judge_calls": judge.calls,
           "label_counts": {"baseline": {k: sum(r["label"] == k for r in base) for k in sorted({r["label"] for r in base})},
                            "drift": {k: sum(r["label"] == k for r in drift) for k in sorted({r["label"] for r in drift})}},
           "agreement_with_claude_strict": {"baseline": agreement(base, "claude", "strict"), "drift_same_answer": agreement(drift, "claude", "strict")},
           "headline_under_claude": headline([dict(r, acc=r["claude"]) for r in base], drift, "acc", "drift_claude"),
           "headline_under_strict": headline([dict(r, acc=r["strict"]) for r in base], drift, "acc", "drift_strict")}
    if rubric == "tiered_v1":
        out["agreement_with_claude_lenient"] = {"baseline": agreement(base, "claude", "lenient"), "drift_same_answer": agreement(drift, "claude", "lenient")}
        out["headline_under_lenient"] = headline([dict(r, acc=r["lenient"]) for r in base], drift, "acc", "drift_lenient")
    RESULTS.mkdir(exist_ok=True)
    name = jid.lower().replace(".", "_")
    (RESULTS / f"judge_validation_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("label_counts", "agreement_with_claude_strict", "headline_under_claude", "headline_under_strict")}, indent=2))
    print("CHECK: if a headline gap or the category null changes under the second judge, tell the writing agent now.")


if __name__ == "__main__":
    main()
