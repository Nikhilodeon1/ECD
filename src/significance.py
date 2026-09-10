"""Significance testing helpers. Test choice by data structure:

- one binary row per subject, independent groups: two_proportion_fisher
- proportion per subject, independent groups (drift rate by source, since
  each case has 5 category-trials): mann_whitney_u on per-case aggregates
- paired binary across >2 conditions (categories, alpha): cochrans_q +
  pairwise mcnemar_from_pairs with holm_bonferroni
"""
from scipy.stats import binomtest, chi2, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant-pair counts b, c."""
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue


def mcnemar_from_pairs(before: list[bool], after: list[bool]) -> dict:
    """Paired binary outcomes under two conditions (same subject order)."""
    assert len(before) == len(after)
    b = sum(1 for x, y in zip(before, after) if x and not y)
    c = sum(1 for x, y in zip(before, after) if not x and y)
    return {"b": b, "c": c, "p_value": mcnemar_exact(b, c)}


def cochrans_q(binary_matrix: list[list[bool]]) -> dict:
    """Joint test that >=2 paired binary conditions (columns) differ across
    subjects (rows). matrix[i][j] = subject i's outcome under condition j."""
    n = len(binary_matrix)
    if n == 0:
        return {"q": 0.0, "df": 0, "p_value": 1.0}
    k = len(binary_matrix[0])
    col_totals = [sum(row[j] for row in binary_matrix) for j in range(k)]
    row_totals = [sum(row) for row in binary_matrix]

    denom = k * sum(row_totals) - sum(r * r for r in row_totals)
    if denom == 0:  # no within-subject variability
        return {"q": 0.0, "df": k - 1, "p_value": 1.0}

    q = (k - 1) * (k * sum(t * t for t in col_totals) - sum(col_totals) ** 2) / denom
    return {"q": round(q, 3), "df": k - 1, "p_value": chi2.sf(q, k - 1)}


def two_proportion_fisher(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    """Independent groups, one binary row per subject. Not for multi-row-per-subject data."""
    table = [[success_a, n_a - success_a], [success_b, n_b - success_b]]
    odds_ratio, p_value = fisher_exact(table)
    return {
        "rate_a": round(success_a / n_a, 3) if n_a else 0,
        "rate_b": round(success_b / n_b, 3) if n_b else 0,
        "odds_ratio": round(odds_ratio, 3),
        "p_value": p_value,
    }


def mann_whitney_u(group_a: list[float], group_b: list[float]) -> dict:
    """Independent groups, one proportion per subject (avoids pseudo-replication)."""
    if not group_a or not group_b:
        return {"n_a": len(group_a), "n_b": len(group_b), "u": None, "p_value": 1.0}
    u, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": round(sum(group_a) / len(group_a), 3),
        "mean_b": round(sum(group_b) / len(group_b), 3),
        "u": round(u, 3),
        "p_value": p_value,
    }


def holm_bonferroni(labeled_pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction over a set of labeled pairwise p-values."""
    labels = list(labeled_pvalues.keys())
    pvals = [labeled_pvalues[k] for k in labels]
    reject, corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return {
        label: {"p_corrected": round(p, 4), "significant": bool(r)}
        for label, p, r in zip(labels, corrected, reject)
    }
