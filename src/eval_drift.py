"""Week 6: adversarial injection across all 5 taxonomy categories.

For every case Claude got right in the Week 5 baseline, generates an
adversarial note per category (adversarial_notes.py), re-asks for a
diagnosis with the note appended, and checks whether the diagnosis changed
from the model's own original answer -- that's Diagnosis Drift. Compared
against the model's own prior answer, not against ground truth, since drift
is about the model changing its mind, not about accuracy per se (though for
this sample the two are closely related since we only test cases that
started out correct).

Usage (from src/, after Week 5's eval_baseline.py has already produced
eval_results_claude.jsonl):
    python eval_drift.py

Each trial needs 3 sequentially-dependent calls (generate note -> use it to
get a followup diagnosis -> grade whether that drifted), but trials are
independent of EACH OTHER, so this batches stage-wise across all trials
instead of one trial at a time: generate all 960 notes, then all 960
followup diagnoses, then all 960 drift checks. In CLAUDE_MODE=manual that's
3 sets of paste-many-get-many round trips instead of 2,880 individual ones.
Behaviorally identical to the old interleaved version in normal API mode.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from adversarial_notes import CATEGORY_DEFINITIONS, build_generate_note_prompt
from grade import build_drift_check_prompt, parse_verdict
from llm_clients import get_claude_client
from prompts import build_followup_prompt, parse_diagnosis


def load_baseline_correct(baseline_results_path: str, sample_path: str) -> list[dict]:
    with open(baseline_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    with open(sample_path, encoding="utf-8") as f:
        cases_by_id = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    correct = []
    for r in results:
        if not r["correct"]:
            continue
        case = cases_by_id.get(r["case_id"])
        if case is None:
            continue
        correct.append({**case, "baseline_diagnosis": r["predicted"]})
    return correct


def run_drift_eval(
    correct_cases: list[dict],
    client,
    out_path: str,
    long_chunk_size: int = 40,
    short_chunk_size: int = 400,
) -> list[dict]:
    categories = list(CATEGORY_DEFINITIONS.keys())
    # every (case, category) combination is one trial -- flatten into one
    # list so each stage below batches across ALL trials at once
    trials = [(case, category) for case in correct_cases for category in categories]
    print(f"{len(trials)} total trials across 3 batched stages")

    # stage 1: generate an adversarial note per trial
    # long prompts (embed the full case evidence, ~1200 tokens worst-case) --
    # 40/chunk keeps well under a 200k context window with room for output
    note_prompts = [
        build_generate_note_prompt(case["original_note"], case["baseline_diagnosis"], category)
        for case, category in trials
    ]
    adversarial_notes = [
        n.strip() if n else ""
        for n in client.generate_batch(note_prompts, max_tokens=200, chunk_size=long_chunk_size)
    ]

    # stage 2: get a followup diagnosis using each generated note
    followup_prompts = [
        build_followup_prompt(case["original_note"], note)
        for (case, category), note in zip(trials, adversarial_notes)
    ]
    raw_afters = client.generate_batch(followup_prompts, max_tokens=300, chunk_size=long_chunk_size)
    diagnoses_after = [parse_diagnosis(r) if r else "" for r in raw_afters]

    # stage 3: grade whether each diagnosis actually changed
    # short prompts (two diagnosis strings, ~250 tokens) -- can go big
    drift_prompts = [
        build_drift_check_prompt(case["baseline_diagnosis"], diag_after)
        for (case, category), diag_after in zip(trials, diagnoses_after)
    ]
    drift_raws = client.generate_batch(drift_prompts, max_tokens=250, chunk_size=short_chunk_size)

    results = []
    for (case, category), note, diag_after, drift_raw in zip(
        trials, adversarial_notes, diagnoses_after, drift_raws
    ):
        if not note or not diag_after or not drift_raw:
            print(f"skipping {case['id']}/{category}: missing a stage result (see batch warnings above)")
            continue
        drifted = not parse_verdict(drift_raw)
        results.append(
            {
                "case_id": case["id"],
                "source": case["source"],
                "category": category,
                "diagnosis_before": case["baseline_diagnosis"],
                "adversarial_note": note,
                "diagnosis_after": diag_after,
                "drifted": drifted,
            }
        )
        print(f"{case['id']} / {category} drifted={drifted}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def drift_rate_table(results: list[dict]) -> dict:
    by_cat = defaultdict(lambda: {"drifted": 0, "total": 0})
    by_source = defaultdict(lambda: {"drifted": 0, "total": 0})
    overall = {"drifted": 0, "total": 0}

    for r in results:
        by_cat[r["category"]]["total"] += 1
        by_cat[r["category"]]["drifted"] += int(r["drifted"])
        by_source[r["source"]]["total"] += 1
        by_source[r["source"]]["drifted"] += int(r["drifted"])
        overall["total"] += 1
        overall["drifted"] += int(r["drifted"])

    def rate(d):
        return round(d["drifted"] / d["total"], 3) if d["total"] else 0

    return {
        "overall_ddr": rate(overall),
        "overall_n": overall["total"],
        "by_category": {k: {"ddr": rate(v), "n": v["total"]} for k, v in by_cat.items()},
        "by_source": {k: {"ddr": rate(v), "n": v["total"]} for k, v in by_source.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="../data/processed/eval_results_claude.jsonl")
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--out", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument(
        "--long-chunk-size", type=int, default=40, help="manual mode only -- note/followup prompts"
    )
    parser.add_argument(
        "--short-chunk-size", type=int, default=400, help="manual mode only -- drift-check prompts"
    )
    args = parser.parse_args()

    correct_cases = load_baseline_correct(args.baseline_results, args.sample)
    print(
        f"{len(correct_cases)} baseline-correct cases x 5 categories "
        f"= {len(correct_cases) * 5} drift trials"
    )

    client = get_claude_client()
    results = run_drift_eval(
        correct_cases,
        client,
        args.out,
        long_chunk_size=args.long_chunk_size,
        short_chunk_size=args.short_chunk_size,
    )

    table = drift_rate_table(results)
    print(json.dumps(table, indent=2))

    table_path = Path(args.out).with_name("drift_rate_table.json")
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"drift rate table -> {table_path}")
