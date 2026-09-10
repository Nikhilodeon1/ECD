"""Significance testing driver: reads eval/drift/tradeoff result files,
writes significance_report.json. Skips the tradeoff test if that file is absent.
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
    """MedQA vs MIMIC, one row per case (baseline accuracy). Fisher's exact."""
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
    """MedQA vs MIMIC drift rate: per-case drift proportion, Mann-Whitney U
    (case is the independent unit, not the trial)."""
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
    return {"comparison": f"{a} vs {b}", **mann_whitney_u(by_source_rates[a], by_source_rates[b])}


def build_paired_matrix(
    results: list[dict], condition_key: str, outcome_key: str, id_fields: tuple[str, ...] = ("case_id",)
) -> tuple[list[list[bool]], list, int]:
    """Pivot subject x condition -> outcome. id_fields=("case_id","category")
    for the alpha sweep, where a case_id recurs once per drifted category.
    Drops subjects missing any condition."""
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
    """Paired conditions (category or alpha): Cochran's Q + pairwise McNemar
    with Holm-Bonferroni. baseline_condition restricts pairs to X vs baseline."""
    matrix, conditions, dropped = build_paired_matrix(results, condition_key, outcome_key, id_fields)
    if dropped:
        print(f"dropped {dropped} incomplete subjects")

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
        report["week8_recovery_rate_by_alpha"] = test_paired_conditions(
            tradeoff_results,
            condition_key="alpha",
            outcome_key="recovered",
            baseline_condition=0.0,
            id_fields=("case_id", "category"),  # case_id recurs per drifted category
        )
    else:
        print(f"skipping Week 8 test -- {args.tradeoff_results} not found")

    print(json.dumps(report, indent=2, default=str))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"significance report -> {args.out}")
