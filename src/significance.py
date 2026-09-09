"""Week 9: significance testing helpers.

Which test to use depends on whether the comparison is PAIRED or INDEPENDENT
-- this matters a lot in this project and it's easy to get wrong:

- MedQA vs. MIMIC (baseline accuracy, drift rate): INDEPENDENT groups --
  a given case is one source or the other, never both. Use Fisher's exact
  test (two_proportion_fisher), not McNemar's.
- Category vs. category (VCS/ISO/PLV/MMS/SNHR): PAIRED -- every
  baseline-correct case gets tested under all 5 categories (see
  eval_drift.py's trial construction: `for case in correct_cases: for
  category in categories`). Use McNemar's test pairwise, Cochran's Q
  jointly across all 5.
- Alpha vs. alpha (ECD tradeoff sweep): PAIRED for the same reason -- every
  drifted case gets tested at every alpha (see ecd_tradeoff.py's
  sweep_drift_recovery: `for case in drifted_cases: for alpha in alphas`).
  Same tests as categories.

Flagged back in Week 2's technical report as the right toolset
(scipy.stats + statsmodels) -- this is that plan actually built out.
"""
from scipy.stats import binomtest, chi2, fisher_exact
from statsmodels.stats.multitest import multipletests


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar's test p-value from the two discordant-pair
    counts: b = condition-A-success/condition-B-fail, c = the reverse.
    Concordant pairs (both same) don't enter McNemar's at all -- only
    disagreements carry information about which condition is better."""
    if b + c == 0:
        return 1.0  # no discordant pairs at all -- no evidence of a difference
    return binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue


def mcnemar_from_pairs(before: list[bool], after: list[bool]) -> dict:
    """Same-subject paired outcomes under two conditions -- e.g. same case's
    drift outcome under category X vs category Y, or under alpha=0 vs
    alpha=1. before/after must be the same length and same subject order."""
    assert len(before) == len(after), "paired lists must be the same length"
    b = sum(1 for x, y in zip(before, after) if x and not y)
    c = sum(1 for x, y in zip(before, after) if not x and y)
    return {"b": b, "c": c, "p_value": mcnemar_exact(b, c)}


def cochrans_q(binary_matrix: list[list[bool]]) -> dict:
    """Tests whether >=2 PAIRED binary conditions (columns) differ jointly
    across the same subjects (rows) -- the multi-condition generalization of
    McNemar's test. binary_matrix[i][j] = subject i's outcome under
    condition j. Use this first to check "do these 5 categories/alphas
    differ at all" before running pairwise McNemar's to find which ones."""
    n = len(binary_matrix)
    if n == 0:
        return {"q": 0.0, "df": 0, "p_value": 1.0}
    k = len(binary_matrix[0])
    col_totals = [sum(row[j] for row in binary_matrix) for j in range(k)]
    row_totals = [sum(row) for row in binary_matrix]

    denom = k * sum(row_totals) - sum(r * r for r in row_totals)
    if denom == 0:
        # every subject scored the same on every condition (all 0s or all 1s
        # per row) -- no variability to test, not a real difference
        return {"q": 0.0, "df": k - 1, "p_value": 1.0}

    q = (k - 1) * (k * sum(t * t for t in col_totals) - sum(col_totals) ** 2) / denom
    df = k - 1
    p_value = chi2.sf(q, df)
    return {"q": round(q, 3), "df": df, "p_value": p_value}


def two_proportion_fisher(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    """INDEPENDENT-groups comparison (e.g. MedQA vs. MIMIC) -- Fisher's exact
    over a 2x2 table, not McNemar's (these are different cases, not paired
    measurements of the same subject)."""
    table = [[success_a, n_a - success_a], [success_b, n_b - success_b]]
    odds_ratio, p_value = fisher_exact(table)
    return {
        "rate_a": round(success_a / n_a, 3) if n_a else 0,
        "rate_b": round(success_b / n_b, 3) if n_b else 0,
        "odds_ratio": round(odds_ratio, 3),
        "p_value": p_value,
    }


def holm_bonferroni(labeled_pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Multiple-comparison correction for a set of pairwise tests (e.g. all
    10 category pairs, or all alpha-vs-alpha=0 pairs) -- controls the false
    positive rate that would otherwise inflate from running many McNemar
    tests at once. Returns each label's corrected p-value and reject flag."""
    labels = list(labeled_pvalues.keys())
    pvals = [labeled_pvalues[k] for k in labels]
    reject, corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return {
        label: {"p_corrected": round(p, 4), "significant": bool(r)}
        for label, p, r in zip(labels, corrected, reject)
    }
