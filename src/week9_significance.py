"""Week 9: significance testing driver.

Runs the actual formal significance tests behind Weeks 5, 6, and 8's
headline numbers -- turns "44% vs 17%" into a real p-value, and checks
whether the category/alpha differences are real or noise (the same
question the CIs in ecd_tradeoff.py were already hinting at for the alpha
sweep -- this makes it formal).

Usage (from src/, after eval_baseline.py and eval_drift.py have produced
their output -- ecd_tradeoff.py's output is optional, its test is skipped
cleanly if that file doesn't exist yet):
    python week9_significance.py
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations

from significance import (
    cochrans_q,
    holm_bonferroni,
    mann_whitney_u,
    mcnemar_from_pairs,
    two_proportion_fisher,
)


def load_jsonl(path: str) -> list[dict] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return None


def test_independent_rate_by_source_one_row_per_case(results: list[dict], outcome_key: str) -> dict:
    """MedQA vs. MIMIC where the input has exactly ONE row per case (e.g.
    eval_results_claude.jsonl -- Week 5 baseline accuracy). Independent
    groups, Fisher's exact is correct here. Do NOT use this on
    drift_results_claude.jsonl -- that has 5 rows per case (one per
    category), which would pseudo-replicate; use
    test_drift_rate_by_source_case_level for that instead."""
    by_source = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        by_source[r["source"]]["total"] += 1
        by_source[r["source"]]["success"] += int(r[outcome_key])

    sources = list(by_source.keys())
    if len(sources) != 2:
        return {"skipped": f"expected exactly 2 sources, found {sources}"}
    a, b = sources
    return {
        "comparison": f"{a} vs {b}",
        **two_proportion_fisher(
            by_source[a]["success"], by_source[a]["total"],
            by_source[b]["success"], by_source[b]["total"],
        ),
    }


def test_drift_rate_by_source_case_level(results: list[dict]) -> dict:
    """MedQA vs. MIMIC drift rate, corrected for pseudo-replication: found
    during paper review that testing this on all 994 trial-rows directly
    (5 per case) treats each case's 5 category-trials as independent, which
    they aren't (same patient, correlated evidence). Fix: collapse each case
    to its OWN drift proportion across its categories first (one number per
    case, in [0,1]), then compare the two groups' per-case proportions with
    Mann-Whitney U -- the case, not the trial, is the true independent unit."""
    by_case = defaultdict(lambda: {"source": None, "drifted": 0, "total": 0})
    for r in results:
        c = by_case[r["case_id"]]
        c["source"] = r["source"]
        c["total"] += 1
        c["drifted"] += int(r["drifted"])

    by_source_rates = defaultdict(list)
    for case in by_case.values():
        by_source_rates[case["source"]].append(case["drifted"] / case["total"])

    sources = list(by_source_rates.keys())
    if len(sources) != 2:
        return {"skipped": f"expected exactly 2 sources, found {sources}"}
    a, b = sources
    return {
        "comparison": f"{a} vs {b}",
        "note": "per-case drift proportion, Mann-Whitney U (not trial-level Fisher's -- avoids pseudo-replication across a case's 5 category-trials)",
        **mann_whitney_u(by_source_rates[a], by_source_rates[b]),
    }


def build_paired_matrix(
    results: list[dict], condition_key: str, outcome_key: str, id_fields: tuple[str, ...] = ("case_id",)
) -> tuple[list[list[bool]], list, int]:
    """Pivots subject x condition -> outcome. The subject key matters:
    tradeoff_results.jsonl can have the SAME case_id appear multiple times
    (once per category it drifted under, each running its own full alpha
    sweep) -- pivoting on case_id alone silently overwrites one category's
    alpha row with another's. id_fields=("case_id","category") makes each
    (case, category) combination its own subject, which is what's actually
    independently alpha-swept. Week 6's category test doesn't have this
    issue (case_id is already unique there), so its default stays case_id
    alone. Drops any subject missing a value for any condition -- McNemar's/
    Cochran's Q need a complete matrix, a partial row isn't a valid paired
    observation."""
    by_subject = defaultdict(dict)
    for r in results:
        key = tuple(r[f] for f in id_fields)
        by_subject[key][r[condition_key]] = r[outcome_key]

    conditions = sorted({r[condition_key] for r in results}, key=str)
    complete_subjects = [
        key for key, row in by_subject.items() if all(c in row for c in conditions)
    ]
    dropped = len(by_subject) - len(complete_subjects)
    matrix = [[by_subject[key][c] for c in conditions] for key in complete_subjects]
    return matrix, conditions, dropped


def test_paired_conditions(
    results: list[dict],
    condition_key: str,
    outcome_key: str,
    baseline_condition=None,
    id_fields: tuple[str, ...] = ("case_id",),
) -> dict:
    """Category vs. category, or alpha vs. alpha -- these ARE paired (same
    case tested under every condition, see eval_drift.py / ecd_tradeoff.py's
    trial construction), so this uses Cochran's Q (joint) + pairwise
    McNemar's with Holm-Bonferroni correction, not independent-group tests.

    baseline_condition: if given, only tests each other condition against
    this one (e.g. every alpha vs. alpha=0.0) instead of all pairs -- fewer
    comparisons, more power after Holm correction, and matches the actual
    question ("does ECD beat undefended") better than every pair."""
    matrix, conditions, dropped = build_paired_matrix(results, condition_key, outcome_key, id_fields)
    if dropped:
        print(f"dropped {dropped} incomplete subjects (not present under all {len(conditions)} conditions)")

    overall = cochrans_q(matrix)

    if baseline_condition is not None and baseline_condition in conditions:
        pairs = [(baseline_condition, c) for c in conditions if c != baseline_condition]
    else:
        pairs = list(combinations(conditions, 2))

    pairwise_raw = {}
    for c1, c2 in pairs:
        i1, i2 = conditions.index(c1), conditions.index(c2)
        before = [row[i1] for row in matrix]
        after = [row[i2] for row in matrix]
        pairwise_raw[f"{c1} vs {c2}"] = mcnemar_from_pairs(before, after)["p_value"]

    return {
        "n_complete_cases": len(matrix),
        "dropped_incomplete": dropped,
        "cochrans_q": overall,
        "pairwise_mcnemar_holm_corrected": holm_bonferroni(pairwise_raw) if pairwise_raw else {},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="../data/processed/eval_results_claude.jsonl")
    parser.add_argument("--drift-results", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument("--tradeoff-results", default="../data/processed/tradeoff_results.jsonl")
    parser.add_argument("--out", default="../data/processed/significance_report.json")
    args = parser.parse_args()

    report = {}

    baseline_results = load_jsonl(args.baseline_results)
    if baseline_results:
        report["week5_baseline_accuracy_by_source"] = test_independent_rate_by_source_one_row_per_case(
            baseline_results, outcome_key="correct"
        )
    else:
        print(f"skipping Week 5 test -- {args.baseline_results} not found")

    drift_results = load_jsonl(args.drift_results)
    if drift_results:
        report["week6_drift_rate_by_source"] = test_drift_rate_by_source_case_level(drift_results)
        report["week6_drift_rate_by_category"] = test_paired_conditions(
            drift_results, condition_key="category", outcome_key="drifted"
        )
    else:
        print(f"skipping Week 6 tests -- {args.drift_results} not found")

    tradeoff_results = load_jsonl(args.tradeoff_results)
    if tradeoff_results:
        # id_fields includes "category": the same case_id can recur once per
        # category it drifted under, each running its own independent alpha
        # sweep -- case_id alone would silently collapse/overwrite those
        report["week8_recovery_rate_by_alpha"] = test_paired_conditions(
            tradeoff_results,
            condition_key="alpha",
            outcome_key="recovered",
            baseline_condition=0.0,
            id_fields=("case_id", "category"),
        )
    else:
        print(f"skipping Week 8 test -- {args.tradeoff_results} not found (fine if that run isn't done yet)")

    print(json.dumps(report, indent=2, default=str))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"significance report -> {args.out}")
