"""T0 / rule 10: run FIRST on every fresh pod, from the repo root.

Verifies data/processed, eval_sample.jsonl and .env exist, that the case_id sets
across eval_sample / baseline results / drift results are mutually consistent,
and records config facts. Writes aggregate counts only (no text) to
results/preflight.json.

If a file is missing, rebuild with build_dataset.py then subsample.py (seed 42)
and rerun this script. The case_id check against EXISTING result files is what
tells you whether the rebuilt sample is the same sample.
"""
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import ALLOW_MIMIC, DATA as PROC, read_jsonl  # noqa: E402  (read_jsonl drops MIMIC unless allowed)

# full mode (RunPod, ECD_ALLOW_MIMIC=1) vs public mode (shared cluster: MedQA rows only)
EXPECT = (
    {"sample": 300, "per_source": 150, "baseline_correct": 199, "drift_trials": 994}
    if ALLOW_MIMIC
    else {"sample": 150, "per_source": 150, "baseline_correct": 117, "drift_trials": 585}
)


def mtime(p: Path):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main() -> int:
    problems, facts = [], {}
    files = {n: PROC / n for n in (
        "cases.jsonl", "eval_sample.jsonl", "eval_results_claude.jsonl", "drift_results_claude.jsonl")}
    env_ok = (ROOT / ".env").exists()
    facts["mode"] = "full (MIMIC allowed)" if ALLOW_MIMIC else "public (MIMIC rows dropped)"
    facts["env_file_present"] = env_ok
    if not env_ok:
        problems.append(".env missing (needs ANTHROPIC_API_KEY or ClaudeKey)")
    facts["files"] = {n: {"exists": p.exists(), "mtime": mtime(p)} for n, p in files.items()}
    for n, p in files.items():
        if not p.exists():
            problems.append(f"missing data/processed/{n}")

    ids = {}
    if files["eval_sample.jsonl"].exists():
        sample = read_jsonl(files["eval_sample.jsonl"])
        ids["sample"] = {c["id"] for c in sample}
        by_src = Counter(c["source"] for c in sample)
        facts["sample"] = {"n": len(sample), "by_source": dict(by_src)}
        if len(sample) != EXPECT["sample"] or any(v != EXPECT["per_source"] for v in by_src.values()):
            problems.append(f"sample is {len(sample)} {dict(by_src)}, expected 300 = 150/150")
    if files["eval_results_claude.jsonl"].exists():
        base = read_jsonl(files["eval_results_claude.jsonl"])
        ids["baseline"] = {r["case_id"] for r in base}
        correct = [r for r in base if r["correct"]]
        facts["baseline"] = {"n": len(base), "correct": len(correct),
                             "correct_by_source": dict(Counter(r["source"] for r in correct))}
        if len(correct) != EXPECT["baseline_correct"]:
            problems.append(f"baseline-correct {len(correct)} != {EXPECT['baseline_correct']}")
        ids["baseline_correct"] = {r["case_id"] for r in correct}
    if files["drift_results_claude.jsonl"].exists():
        drift = read_jsonl(files["drift_results_claude.jsonl"])
        ids["drift"] = {r["case_id"] for r in drift}
        facts["drift"] = {"n_trials": len(drift), "by_category": dict(Counter(r["category"] for r in drift)),
                          "drifted": sum(int(r["drifted"]) for r in drift)}
        if abs(len(drift) - EXPECT["drift_trials"]) > (0 if ALLOW_MIMIC else 1):
            problems.append(f"drift trials {len(drift)} != {EXPECT['drift_trials']}")

    if "sample" in ids:
        for name in ("baseline", "baseline_correct", "drift"):
            if name in ids and not ids[name] <= ids["sample"]:
                problems.append(f"{name} case_ids not a subset of the current eval_sample "
                                f"({len(ids[name] - ids['sample'])} unknown): rebuilt sample differs from the one the results came from")
        if "baseline" in ids and ids["baseline"] != ids["sample"]:
            problems.append(f"baseline covers {len(ids['baseline'])} of {len(ids['sample'])} sample cases")
    if "baseline_correct" in ids and "drift" in ids and ids["drift"] != ids["baseline_correct"]:
        problems.append("drift case_ids != baseline-correct case_ids")

    facts["environment"] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "git_head": sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
    }
    for mod in ("torch", "transformers", "anthropic", "scipy", "statsmodels", "sklearn"):
        try:
            facts["environment"][mod] = __import__(mod).__version__
        except Exception:
            facts["environment"][mod] = None

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": not problems, "problems": problems, **facts}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "preflight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("\nPREFLIGHT " + ("OK" if not problems else "FAILED:\n  - " + "\n  - ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
