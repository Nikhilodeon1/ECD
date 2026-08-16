"""Week 8: accuracy-drift tradeoff curve for ECD.

Sweeps alpha across cases that drifted under Claude in Week 6, running
Llama-Med + ECD on each and measuring the DRIFT RECOVERY RATE at each
alpha: does the ECD-defended output match the original pre-drift (correct)
diagnosis? Grading uses Claude as an LLM-judge (src/grade.py), not string
parsing -- Week 7 showed this open model doesn't reliably follow the
structured "Diagnosis: X" format, so exact parsing would undercount
correct-but-differently-worded recoveries.

Clean-case accuracy is measured ONCE, not swept per alpha. When there's no
adversarial note, original_prompt == full_prompt, so both ECD forward
passes see identical input and produce identical logits -- the blend
(1+a)*x - a*x collapses to exactly x for any alpha. This is a provable
property of the algorithm (see docs/week8-tradeoff-results.md), not an
approximation, so re-running clean cases at every alpha would just
re-confirm the same identity at real GPU/API cost for no new information.

Usage (from src/, on the GPU pod):
    python ecd_tradeoff.py --n-drifted 20 --n-clean 20 --alphas 0,0.5,1,1.5,2
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from grade import build_grade_prompt, parse_verdict
from llm_clients import LlamaMedClient, get_claude_client
from prompts import build_baseline_prompt, build_followup_prompt, parse_diagnosis


def load_drifted_cases(
    drift_results_path: str, sample_path: str, n: int, source: str | None = None
) -> list[dict]:
    """source: filter to 'medqa' (fully public, safe on any cluster) or
    'mimic-iv-note' (real PhysioNet-restricted evidence text, only run this
    where that's cleared). None = no filter, mixes both."""
    with open(drift_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    with open(sample_path, encoding="utf-8") as f:
        cases_by_id = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    drifted = []
    for r in results:
        if not r["drifted"]:
            continue
        case = cases_by_id.get(r["case_id"])
        if case is None:
            continue
        if source is not None and case["source"] != source:
            continue
        drifted.append({**r, "original_note": case["original_note"]})
        if len(drifted) >= n:
            break
    return drifted


def load_clean_cases(sample_path: str, n: int, source: str | None = None) -> list[dict]:
    with open(sample_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if source is not None:
        cases = [c for c in cases if c["source"] == source]
    return cases[:n]


def grade_match(judge_client, ground_truth: str, predicted: str) -> bool:
    prompt = build_grade_prompt(ground_truth, predicted)
    raw = judge_client.generate(prompt, max_tokens=250)
    return parse_verdict(raw)


def measure_clean_accuracy(clean_cases: list[dict], llama_client, judge_client) -> dict:
    """Run once -- see module docstring for why this doesn't need to be
    swept per alpha. Uses alpha=0 (arbitrary; mathematically equivalent to
    any other alpha here since original_prompt == full_prompt)."""
    correct = 0
    for c in clean_cases:
        prompt = build_baseline_prompt(c["original_note"])
        raw = llama_client.generate_ecd(prompt, prompt, alpha=0.0, max_tokens=200)
        predicted = parse_diagnosis(raw)
        if grade_match(judge_client, c["diagnosis_ground_truth"], predicted):
            correct += 1
    return {"accuracy": round(correct / len(clean_cases), 3), "n": len(clean_cases)}


def sweep_drift_recovery(
    drifted_cases: list[dict], alphas: list[float], llama_client, judge_client, out_path: str
) -> dict:
    results = []
    total = len(drifted_cases) * len(alphas)
    i = 0
    for case in drifted_cases:
        original_prompt = build_baseline_prompt(case["original_note"])
        full_prompt = build_followup_prompt(case["original_note"], case["adversarial_note"])

        for alpha in alphas:
            i += 1
            try:
                raw = llama_client.generate_ecd(original_prompt, full_prompt, alpha=alpha, max_tokens=200)
                predicted = parse_diagnosis(raw)
                recovered = grade_match(judge_client, case["diagnosis_before"], predicted)
            except Exception as e:
                print(f"[{i}/{total}] {case['case_id']} alpha={alpha} failed: {e}")
                continue

            results.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "alpha": alpha,
                    "diagnosis_before": case["diagnosis_before"],
                    "ecd_output": predicted,
                    "recovered": recovered,
                }
            )
            print(f"[{i}/{total}] {case['case_id']} alpha={alpha} recovered={recovered}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    by_alpha = defaultdict(lambda: {"recovered": 0, "total": 0})
    for r in results:
        by_alpha[r["alpha"]]["total"] += 1
        by_alpha[r["alpha"]]["recovered"] += int(r["recovered"])

    curve = {
        str(alpha): {
            "recovery_rate": round(v["recovered"] / v["total"], 3) if v["total"] else 0,
            "n": v["total"],
        }
        for alpha, v in sorted(by_alpha.items())
    }
    return curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-results", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--n-drifted", type=int, default=20)
    parser.add_argument("--n-clean", type=int, default=20)
    parser.add_argument("--alphas", default="0,0.5,1,1.5,2")
    parser.add_argument("--out", default="../data/processed/tradeoff_results.jsonl")
    parser.add_argument("--out-curve", default="../data/processed/tradeoff_curve.json")
    parser.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-8B")
    parser.add_argument(
        "--source",
        default=None,
        choices=["medqa", "mimic-iv-note"],
        help=(
            "filter to one source. 'medqa' is fully public -- safe on any cluster "
            "(e.g. Nautilus, free GPUs). 'mimic-iv-note' contains real PhysioNet-"
            "restricted evidence text -- only run that on a cleared environment "
            "(e.g. your RunPod pod). Omit to run both mixed."
        ),
    )
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    drifted_cases = load_drifted_cases(args.drift_results, args.sample, args.n_drifted, args.source)
    clean_cases = load_clean_cases(args.sample, args.n_clean, args.source)
    print(
        f"{len(drifted_cases)} drifted cases x {len(alphas)} alphas = "
        f"{len(drifted_cases) * len(alphas)} recovery trials, plus {len(clean_cases)} clean-accuracy trials"
    )

    llama_client = LlamaMedClient(model_name=args.model_name)
    judge_client = get_claude_client()

    clean_accuracy = measure_clean_accuracy(clean_cases, llama_client, judge_client)
    print(f"clean-case accuracy (alpha-invariant): {json.dumps(clean_accuracy)}")

    curve = sweep_drift_recovery(drifted_cases, alphas, llama_client, judge_client, args.out)

    final = {"clean_accuracy": clean_accuracy, "recovery_by_alpha": curve}
    print(json.dumps(final, indent=2))
    Path(args.out_curve).write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"tradeoff curve -> {args.out_curve}")

    try:
        import matplotlib.pyplot as plt

        sorted_items = sorted(curve.items(), key=lambda kv: float(kv[0]))
        alphas_sorted = [float(k) for k, _ in sorted_items]
        recovery_rates = [v["recovery_rate"] for _, v in sorted_items]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(alphas_sorted, recovery_rates, marker="o", label="drift recovery rate")
        ax.axhline(
            clean_accuracy["accuracy"], color="gray", linestyle="--",
            label=f"clean-case accuracy (alpha-invariant, n={clean_accuracy['n']})",
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
        ax.set_title("ECD accuracy-drift tradeoff")
        ax.legend()
        plt.tight_layout()
        plot_path = "../docs/week8-tradeoff-curve.png"
        plt.savefig(plot_path, dpi=150)
        print(f"plot -> {plot_path}")
    except Exception as e:
        print(f"plotting failed (non-fatal, data is already saved): {e}")
