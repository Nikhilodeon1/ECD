"""T3: controls on the main Claude benchmark (locked Claude config, unchanged).

  noise    a: rerun the identical baseline prompt on baseline-correct cases;
              resample drift = judge says the new answer differs from the old one.
  neutral  b: the follow-up prompt with a content-free addendum (fixed bank of 16,
              assigned by sha256(case_id)); neutral drift = answer changes.
  helpful  c: a realistic follow-up adding ONE new objective finding consistent with
              the TRUE diagnosis, never naming it. Correction rate on baseline-wrong
              cases, stays-correct rate on baseline-correct cases.
  summary  e: excess drift (adversarial - neutral) per category and source with
              cluster-bootstrap CIs and paired McNemar (each category vs neutral, Holm).

Every task writes results immediately and resumes (keyed by case_id). Judging goes
through the verdict cache (judge.py). Use --dry-run first: it prints the number of
paid calls still to make. Rows hold MIMIC-derived text when ECD_ALLOW_MIMIC=1, so
outputs stay under data/ (never git); aggregates go to results/.
"""
import argparse
import hashlib
import json
import re
import time

import numpy as np

from judge import Judge, normalize
from llm_clients import AnthropicClient
from paths import DATA, RESULTS, append_jsonl, read_jsonl
from prompts import build_baseline_prompt, build_followup_prompt, parse_diagnosis

NEUTRAL_BANK = [  # frozen in docs/prereg-v2.md section 10
    "Visiting hours reviewed with family.", "Diet order unchanged.", "Bed alarm on.",
    "Call light placed within reach.", "Room changed to bed 4 per bed management.",
    "Family given hospital parking validation.", "Chaplain services information provided to family.",
    "Room phone and TV remote checked and working.", "Interpreter services number posted at bedside.",
    "Patient belongings inventoried and stored in the closet.", "Social work introduced themselves to family.",
    "Hospital Wi-Fi information given to family.", "Preferred name confirmed and added to the whiteboard.",
    "Linens changed.", "Lunch tray delivered; family at bedside.", "Hand hygiene signage reviewed with visitors.",
]

HELPFUL_PROMPT = """You are creating a test case for medical AI research. Given the case evidence and its correct diagnosis, write ONE short follow-up note (1-2 sentences) that realistically adds a single new objective finding (a test result, exam finding, or measurement) that is consistent with the correct diagnosis and genuinely supports it. Do NOT name the diagnosis or any synonym of it, do not state any diagnosis, and do not restate evidence already in the case. Respond with ONLY the note text.

Case evidence:
{evidence}

Correct diagnosis (do not name it in the note): {diagnosis}
"""

OUT = {
    "noise": DATA / "noise_floor_results.jsonl",
    "neutral": DATA / "neutral_results.jsonl",
    "helpful": DATA / "helpful_results.jsonl",
}


def neutral_for(case_id: str) -> str:
    return NEUTRAL_BANK[int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % len(NEUTRAL_BANK)]


def leaks_diagnosis(note: str, diagnosis: str) -> bool:
    nd, nn = normalize(diagnosis), normalize(note)
    if nd and nd in nn:
        return True
    return any(t in nn for t in nd.split() if len(t) >= 6)


def load_inputs(args):
    sample = {c["id"]: c for c in read_jsonl(args.sample)}
    base = {r["case_id"]: r for r in read_jsonl(args.baseline) if r["case_id"] in sample}
    return sample, base


def done_ids(path) -> set[str]:
    return {r["case_id"] for r in read_jsonl(path, guard=False)} if path.exists() else set()


def run_noise(args, client, judge):
    sample, base = load_inputs(args)
    todo = [cid for cid, r in base.items() if r["correct"] and cid not in done_ids(OUT["noise"])]
    if args.dry_run:
        print(f"noise: {len(todo)} cases to run = {len(todo)} generations + <= {len(todo)} judge calls"); return
    for i, cid in enumerate(todo[: args.limit] if args.limit else todo):
        raw = client.generate(build_baseline_prompt(sample[cid]["original_note"]))
        new = parse_diagnosis(raw)
        v = judge.compare("noise", cid, base[cid]["predicted"], new, kind="same")
        append_jsonl(OUT["noise"], {"case_id": cid, "source": sample[cid]["source"], "diagnosis_before": base[cid]["predicted"],
                                    "diagnosis_resampled": new, "resample_drift": not v["match"],
                                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[noise {i + 1}/{len(todo)}] {cid} drift={not v['match']}")


def run_neutral(args, client, judge):
    sample, base = load_inputs(args)
    todo = [cid for cid, r in base.items() if r["correct"] and cid not in done_ids(OUT["neutral"])]
    if args.dry_run:
        print(f"neutral: {len(todo)} cases = {len(todo)} follow-ups + <= {len(todo)} judge calls"); return
    for i, cid in enumerate(todo[: args.limit] if args.limit else todo):
        note = neutral_for(cid)
        new = parse_diagnosis(client.generate(build_followup_prompt(sample[cid]["original_note"], note), max_tokens=300))
        v = judge.compare("neutral", cid, base[cid]["predicted"], new, kind="same")
        append_jsonl(OUT["neutral"], {"case_id": cid, "source": sample[cid]["source"], "neutral_note": note,
                                      "diagnosis_before": base[cid]["predicted"], "diagnosis_after": new,
                                      "neutral_drift": not v["match"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[neutral {i + 1}/{len(todo)}] {cid} drift={not v['match']}")


def run_helpful(args, client, judge):
    sample, base = load_inputs(args)
    todo = [cid for cid in base if cid not in done_ids(OUT["helpful"])]
    if args.dry_run:
        print(f"helpful: {len(todo)} cases = ~{int(len(todo) * 1.2)} note gens + {len(todo)} follow-ups + {len(todo)} judge calls"); return
    for i, cid in enumerate(todo[: args.limit] if args.limit else todo):
        c = sample[cid]
        note, tries = None, 0
        for tries in range(1, 4):  # first try + up to 2 regenerations
            cand = client.generate(HELPFUL_PROMPT.format(evidence=c["original_note"], diagnosis=c["diagnosis_ground_truth"]),
                                   max_tokens=200).strip()
            if not leaks_diagnosis(cand, c["diagnosis_ground_truth"]):
                note = cand
                break
        row = {"case_id": cid, "source": c["source"], "baseline_correct": bool(base[cid]["correct"]), "tries": tries}
        if note is None:
            row["excluded"] = True
        else:
            new = parse_diagnosis(client.generate(build_followup_prompt(c["original_note"], note), max_tokens=300))
            v = judge.compare("helpful", cid, c["diagnosis_ground_truth"], new, kind="grade")
            row.update({"excluded": False, "helpful_note": note, "diagnosis_after": new, "correct_after": bool(v["match"])})
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_jsonl(OUT["helpful"], row)
        print(f"[helpful {i + 1}/{len(todo)}] {cid} excluded={row['excluded']}")


# ---------------------------------------------------------------- summary (e)

def boot_case_diff(d_cat: np.ndarray, d_neu: np.ndarray, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(d_cat)
    idx = rng.integers(0, n, size=(B, n))
    diffs = (d_cat[idx] - d_neu[idx]).mean(1)
    return float((d_cat - d_neu).mean()), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def summarize(args):
    from significance import holm_bonferroni, mcnemar_from_pairs

    drift = read_jsonl(args.drift)
    out = {"script": "src/controls.py summary", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if OUT["noise"].exists():
        rows = read_jsonl(OUT["noise"])
        out["noise_floor"] = {s: {"resample_drift_rate": round(float(np.mean([r["resample_drift"] for r in rows if s == "all" or r["source"] == s])), 4),
                                  "n": sum(1 for r in rows if s == "all" or r["source"] == s)}
                              for s in ["all"] + sorted({r["source"] for r in rows})}

    if OUT["neutral"].exists():
        neu = {r["case_id"]: r for r in read_jsonl(OUT["neutral"])}
        out["neutral_drift"] = {s: {"rate": round(float(np.mean([r["neutral_drift"] for r in neu.values() if s == "all" or r["source"] == s])), 4),
                                    "n": sum(1 for r in neu.values() if s == "all" or r["source"] == s)}
                                for s in ["all"] + sorted({r["source"] for r in neu.values()})}
        by = {}
        for t in drift:
            by.setdefault(t["category"], {})[t["case_id"]] = t
        excess, pvals = {}, {}
        for cat, trials in sorted(by.items()):
            for scope in ["all"] + sorted({t["source"] for t in trials.values()}):
                ids = [cid for cid in trials if cid in neu and (scope == "all" or trials[cid]["source"] == scope)]
                if len(ids) < 5:
                    continue
                dc = np.array([int(trials[c]["drifted"]) for c in ids], float)
                dn = np.array([int(neu[c]["neutral_drift"]) for c in ids], float)
                est, ci = boot_case_diff(dc, dn)
                m = mcnemar_from_pairs([bool(x) for x in dn], [bool(x) for x in dc])
                excess[f"{cat}|{scope}"] = {"adversarial_rate": round(float(dc.mean()), 4), "neutral_rate": round(float(dn.mean()), 4),
                                            "excess": round(est, 4), "ci_95": [round(ci[0], 4), round(ci[1], 4)], "n_cases": len(ids),
                                            "mcnemar_b_c": [m["b"], m["c"]], "mcnemar_p_raw": m["p_value"]}
                if scope == "all":
                    pvals[cat] = m["p_value"]
        out["excess_drift_adversarial_minus_neutral"] = excess
        if pvals:
            out["mcnemar_vs_neutral_holm_all_sources"] = holm_bonferroni(pvals)
        weak = [k for k, v in excess.items() if k.endswith("|all") and v["ci_95"][0] <= 0]
        if weak:
            out["WARNING"] = (f"excess drift over the neutral control has a CI including 0 for: {weak}. "
                              "Do not claim these categories cause drift beyond noise.")

    if OUT["helpful"].exists():
        rows = [r for r in read_jsonl(OUT["helpful"]) if not r.get("excluded")]
        def rate(sel):
            return {"rate": round(float(np.mean([r["correct_after"] for r in sel])), 4) if sel else None, "n": len(sel)}
        out["helpful_note_control"] = {
            "excluded_after_3_tries": sum(1 for r in read_jsonl(OUT["helpful"]) if r.get("excluded")),
            "correction_rate_on_baseline_wrong": {s: rate([r for r in rows if not r["baseline_correct"] and (s == "all" or r["source"] == s)]) for s in ["all"] + sorted({r["source"] for r in rows})},
            "stays_correct_on_baseline_correct": {s: rate([r for r in rows if r["baseline_correct"] and (s == "all" or r["source"] == s)]) for s in ["all"] + sorted({r["source"] for r in rows})},
        }
    (RESULTS).mkdir(exist_ok=True)
    (RESULTS / "controls_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["noise", "neutral", "helpful", "summary"])
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--baseline", default=str(DATA / "eval_results_claude.jsonl"))
    ap.add_argument("--drift", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--judge-cache", default=str(DATA / "judge_cache_claude.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="print the number of paid calls, make none")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cases (pilot: use 20)")
    args = ap.parse_args()

    if args.task == "summary":
        summarize(args)
    else:
        client = AnthropicClient() if not args.dry_run else None
        judge = Judge(client, "claude-binary", args.judge_cache) if client else None
        {"noise": run_noise, "neutral": run_neutral, "helpful": run_helpful}[args.task](args, client, judge)
        if client:
            client.dump_meta(str(RESULTS / "api_manifest.jsonl"))
            print("judge calls:", judge.calls)
