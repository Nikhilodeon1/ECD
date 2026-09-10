"""ECD accuracy-drift tradeoff: sweep alpha over drifted cases (Llama-Med +
ECD), grade recovery via Claude judge. GPU required; only judge calls hit
the API. Clean-case accuracy measured once (alpha-invariant when there is
no adversarial note: full context == original context).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from grade import build_grade_prompt, parse_verdict
from llm_clients import AnthropicClient, LlamaMedClient
from prompts import build_llama_baseline_prompt, build_llama_followup_prompt, parse_llama_diagnosis


def binomial_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial CI."""
    if n == 0:
        return (0.0, 0.0)
    ci = binomtest(successes, n).proportion_ci(confidence_level=confidence, method="exact")
    return (round(ci.low, 3), round(ci.high, 3))


def load_drifted_cases(
    drift_results_path: str, sample_path: str, n: int, source: str | None = None
) -> list[dict]:
    """source='mimic-iv-note' must only run where that data is cleared to be."""
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


def measure_clean_accuracy(clean_cases: list[dict], llama_client, judge_client, out_path: str) -> dict:
    """Run once (alpha-invariant). Writes each result immediately, resumes on rerun."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    results = []
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r["case_id"])
        if done_ids:
            print(f"resuming clean-accuracy: {len(done_ids)} cases already done")

    with p.open("a", encoding="utf-8") as f:
        for c in clean_cases:
            case_id = c["id"]
            if case_id in done_ids:
                continue
            prompt = build_llama_baseline_prompt(c["original_note"])
            try:
                raw = llama_client.generate_ecd(prompt, prompt, alpha=0.0, max_tokens=200)
                predicted = parse_llama_diagnosis(raw)
                correct = grade_match(judge_client, c["diagnosis_ground_truth"], predicted)
            except Exception as e:
                print(f"clean case {case_id} failed: {e}")
                continue
            result = {"case_id": case_id, "correct": correct}
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    n = len(results)
    correct_n = sum(int(r["correct"]) for r in results)
    return {
        "accuracy": round(correct_n / n, 3) if n else 0,
        "n": n,
        "ci_95": binomial_ci(correct_n, n),
    }


def sweep_drift_recovery(
    drifted_cases: list[dict], alphas: list[float], llama_client, judge_client, out_path: str
) -> dict:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    already_done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    already_done.add((r["case_id"], r["alpha"]))
        if already_done:
            print(f"resuming recovery sweep: {len(already_done)} trials already done")

    total = len(drifted_cases) * len(alphas)
    i = 0
    with out.open("a", encoding="utf-8") as f:
        for case in drifted_cases:
            original_prompt = build_llama_baseline_prompt(case["original_note"])
            full_prompt = build_llama_followup_prompt(case["original_note"], case["adversarial_note"])

            for alpha in alphas:
                i += 1
                if (case["case_id"], alpha) in already_done:
                    continue
                try:
                    raw = llama_client.generate_ecd(
                        original_prompt, full_prompt, alpha=alpha, max_tokens=200
                    )
                    predicted = parse_llama_diagnosis(raw)
                    recovered = grade_match(judge_client, case["diagnosis_before"], predicted)
                except Exception as e:
                    print(f"[{i}/{total}] {case['case_id']} alpha={alpha} failed: {e}")
                    continue

                result = {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "alpha": alpha,
                    "diagnosis_before": case["diagnosis_before"],
                    "ecd_output": predicted,
                    "recovered": recovered,
                }
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(f"[{i}/{total}] {case['case_id']} alpha={alpha} recovered={recovered}")

    by_alpha = defaultdict(lambda: {"recovered": 0, "total": 0})
    for r in results:
        by_alpha[r["alpha"]]["total"] += 1
        by_alpha[r["alpha"]]["recovered"] += int(r["recovered"])

    curve = {
        str(alpha): {
            "recovery_rate": round(v["recovered"] / v["total"], 3) if v["total"] else 0,
            "n": v["total"],
            "ci_95": binomial_ci(v["recovered"], v["total"]),
        }
        for alpha, v in sorted(by_alpha.items())
    }
    return curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-results", default="../data/processed/drift_results_claude.jsonl")
    parser.add_argument("--sample", default="../data/processed/eval_sample.jsonl")
    parser.add_argument("--n-drifted", type=int, default=55)
    parser.add_argument("--n-clean", type=int, default=55)
    parser.add_argument("--alphas", default="0,0.5,1,1.5,2")
    parser.add_argument("--out", default="../data/processed/tradeoff_results.jsonl")
    parser.add_argument("--out-clean", default="../data/processed/tradeoff_clean_accuracy.jsonl")
    parser.add_argument("--out-curve", default="../data/processed/tradeoff_curve.json")
    parser.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-8B")
    parser.add_argument("--source", default=None, choices=["medqa", "mimic-iv-note"])
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    drifted_cases = load_drifted_cases(args.drift_results, args.sample, args.n_drifted, args.source)
    clean_cases = load_clean_cases(args.sample, args.n_clean, args.source)
    print(
        f"{len(drifted_cases)} drifted cases x {len(alphas)} alphas = "
        f"{len(drifted_cases) * len(alphas)} recovery trials, plus {len(clean_cases)} clean-accuracy trials"
    )

    llama_client = LlamaMedClient(model_name=args.model_name)
    judge_client = AnthropicClient()

    clean_accuracy = measure_clean_accuracy(clean_cases, llama_client, judge_client, args.out_clean)
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
        print(f"plotting failed (data already saved): {e}")
