"""v3 summary: drift per arm, excess over neutral, framing contrasts (docs/prereg-v3.md section 5).

Reads one judged file from v3_run.py and writes an aggregate-only JSON (no text) to results/.
Cases are the clusters: every interval resamples cases, never trials.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from paths import RESULTS, read_jsonl
from v3_run import ARMS, CATS, FRAMING

B = 10000


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def boot_mean_diff(x: np.ndarray, y: np.ndarray, seed: int = 42) -> list[float]:
    """Paired per-case values; percentile CI of mean(x) - mean(y) over case resamples."""
    rng = np.random.default_rng(seed)
    d = x.astype(float) - y.astype(float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for src in sorted({r["source"] for r in rows} | {"all"}):
        rs = [r for r in rows if src == "all" or r["source"] == src]
        base = {r["case_id"]: r["correct"] for r in rs if r["arm"] == "BASE"}
        drift = {arm: {r["case_id"]: not r["same_as_base"] for r in rs if r["arm"] == arm} for arm in ARMS if arm != "H"}
        s = {"n_cases": len(base), "baseline_correct": sum(base.values()),
             "baseline_accuracy": round(sum(base.values()) / len(base), 3) if base else None, "arms": {}}
        for arm, d in drift.items():
            k, n = sum(d.values()), len(d)
            s["arms"][arm] = {"n": n, "drift": round(k / n, 3) if n else None, "wilson": wilson(k, n)}
        neutral = drift.get("N", {})
        for arm in ("R",) + CATS + FRAMING:
            shared = sorted(set(drift.get(arm, {})) & set(neutral))
            if len(shared) >= 10:
                x = np.array([drift[arm][c] for c in shared]); y = np.array([neutral[c] for c in shared])
                s["arms"][arm]["excess_over_neutral"] = round(float(x.mean() - y.mean()), 3)
                s["arms"][arm]["excess_ci"] = boot_mean_diff(x, y)
        pa = drift.get("A_unattributed", {})
        contrasts = []
        for arm in ("B_attending", "C_nursing"):
            shared = sorted(set(drift.get(arm, {})) & set(pa))
            if len(shared) < 10:
                continue
            x = np.array([drift[arm][c] for c in shared]); y = np.array([pa[c] for c in shared])
            b, c = int((x & ~y).sum()), int((~x & y).sum())
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            contrasts.append({"contrast": f"{arm} minus A_unattributed", "n": len(shared),
                              "diff": round(float(x.mean() - y.mean()), 3), "ci": boot_mean_diff(x, y),
                              "discordant_up": b, "discordant_down": c, "p_exact": round(p, 4)})
        for cobj, ph in zip(contrasts, holm([c["p_exact"] for c in contrasts])):
            cobj["p_holm"] = round(ph, 4)
        s["framing_contrasts"] = contrasts
        h = {r["case_id"]: r["correct_after"] for r in rs if r["arm"] == "H"}
        wrong = [c for c in h if not base.get(c, True)]
        right = [c for c in h if base.get(c, False)]
        s["helpful"] = {"corrects_wrong": {"n": len(wrong), "rate": round(sum(h[c] for c in wrong) / len(wrong), 3) if wrong else None},
                        "stays_correct": {"n": len(right), "rate": round(sum(h[c] for c in right) / len(right), 3) if right else None}}
        out[src] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = read_jsonl(args.judged)
    out = {"script": "src/v3_summary.py", "judged_file": Path(args.judged).name, "by_source": summarize(rows),
           "note": "drift = judge says the arm answer differs from the baseline answer; baseline-correct cases only"}
    path = Path(args.out) if args.out else RESULTS / f"v3_summary_{Path(args.judged).stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
