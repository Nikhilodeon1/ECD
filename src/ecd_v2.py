"""ECD v2 (T6): corrected ECD, recovery R(beta) vs suppression S(beta), plus the
prompt baseline P. Pre-registered in docs/prereg-v2.md.

Phases (each writes incrementally and resumes):
  baseline    model's original-only diagnosis on every sample case; judged vs truth
  undefended  beta=0 full-context output on every drift trial of baseline-correct cases
  freeze-D    D_M = trials where the undefended output differs from the model's own
              original-only output; random seed 42, <=150, stratified by category
  helpful0    beta=0 output with the helpful note on the model's baseline-wrong cases
  freeze-H    H_M = <=150 of those cases (seed 42); U_M = those the helpful note fixed
  sweep       beta grid on D and H            <- needs the prereg committed
  prompt      baseline P on D and H           <- needs the prereg committed
  summarize   R, S, accuracy-on-helpful, decision rule, plot

Judging is Claude binary_v1 through the verdict cache. Outputs hold MIMIC-derived
text when ECD_ALLOW_MIMIC=1, so they stay under data/; aggregates go to results/.
"""
import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np

from judge import Judge, normalize
from paths import DATA, RESULTS, ROOT, append_jsonl, read_jsonl
from prompts import (build_llama_baseline_prompt, build_llama_followup_prompt,
                     build_llama_followup_prompt_skeptical, parse_llama_diagnosis)

BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # prereg grid minus beta=0
CAP = 150
USEFUL = {"betas": [0.25, 0.5, 0.75], "min_R": 0.30, "min_wilson_lb": 0.15, "max_S": 0.10}


def tag_of(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1].lower().replace(".", "_")


def path(tag: str, kind: str):
    return DATA / f"ecd2_{tag}_{kind}.jsonl"


def incoherent(parsed: str) -> bool:
    return not normalize(parsed) or len(parsed.split()) > 15


def check_prereg_committed() -> None:
    """Rule 7: no beta>0 run until docs/prereg-v2.md is committed and unmodified."""
    rel = "docs/prereg-v2.md"
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel], capture_output=True)
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise SystemExit(f"STOP: {rel} is not committed (or has uncommitted edits). Commit the pre-registration "
                         f"first (git add -f {rel}), then rerun.")


def done_keys(p, key_fn) -> set:
    return {key_fn(r) for r in read_jsonl(p, guard=False)} if p.exists() else set()


def stratified_cap(trials: list[dict], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for t in sorted(trials, key=lambda t: (t["case_id"], t["category"])):
        by_cat[t["category"]].append(t)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    quota = {c: min(cap // len(cats), len(by_cat[c])) for c in cats}
    spare = cap - sum(quota.values())
    while spare > 0:
        grew = False
        for c in cats:
            if spare > 0 and quota[c] < len(by_cat[c]):
                quota[c] += 1
                spare -= 1
                grew = True
        if not grew:
            break
    return [t for c in cats for t in by_cat[c][: quota[c]]]


# ------------------------------------------------------------------ phases

def phase_baseline(a, client, judge, sample):
    p = path(a.tag, "baseline")
    todo = [c for c in sample if c["id"] not in done_keys(p, lambda r: r["case_id"])]
    print(f"baseline: {len(todo)} cases")
    for i, c in enumerate(todo):
        orig = build_llama_baseline_prompt(c["original_note"])
        out = parse_llama_diagnosis(client.generate_ecd(orig, orig, beta=0.0, max_tokens=a.max_tokens))
        v = judge.compare(f"{a.tag}-baseline", c["id"], c["diagnosis_ground_truth"], out, kind="grade")
        append_jsonl(p, {"case_id": c["id"], "source": c["source"], "output": out, "correct": bool(v["match"]),
                         "incoherent": incoherent(out), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[baseline {i + 1}/{len(todo)}] {c['id']} correct={v['match']}")


def phase_undefended(a, client, judge, sample, drift):
    base = {r["case_id"]: r for r in read_jsonl(path(a.tag, "baseline"))}
    cases = {c["id"]: c for c in sample}
    p = path(a.tag, "undefended")
    done = done_keys(p, lambda r: (r["case_id"], r["category"]))
    todo = [t for t in drift if base.get(t["case_id"], {}).get("correct") and (t["case_id"], t["category"]) not in done]
    print(f"undefended: {len(todo)} trials (model baseline-correct cases x categories)")
    for i, t in enumerate(todo):
        c = cases[t["case_id"]]
        orig = build_llama_baseline_prompt(c["original_note"])
        full = build_llama_followup_prompt(c["original_note"], t["adversarial_note"])
        out = parse_llama_diagnosis(client.generate_ecd(orig, full, beta=0.0, max_tokens=a.max_tokens))
        v = judge.compare(f"{a.tag}-undefended", f"{t['case_id']}|{t['category']}", base[t["case_id"]]["output"], out, kind="same")
        append_jsonl(p, {"case_id": t["case_id"], "category": t["category"], "source": t["source"], "output": out,
                         "differs_from_original_only": not v["match"], "incoherent": incoherent(out)})
        print(f"[undefended {i + 1}/{len(todo)}] {t['case_id']}/{t['category']} differs={not v['match']}")


def phase_freeze_D(a):
    p = path(a.tag, "D")
    if p.exists() and not a.force:
        print(f"D already frozen at {p} ({len(read_jsonl(p, guard=False))} trials); use --force only before any beta>0 run")
        return
    cand = [r for r in read_jsonl(path(a.tag, "undefended")) if r["differs_from_original_only"]]
    D = stratified_cap(cand, CAP, a.seed)
    p.write_text("".join(json.dumps({"case_id": t["case_id"], "category": t["category"], "source": t["source"]}) + "\n" for t in D), encoding="utf-8")
    print(f"D frozen: {len(D)} of {len(cand)} candidate trials; by category {dict(sorted(Counter(t['category'] for t in D).items()))}")


def phase_helpful0(a, client, judge, sample):
    helpful = {r["case_id"]: r for r in read_jsonl(DATA / "helpful_results.jsonl") if not r.get("excluded")}
    base = {r["case_id"]: r for r in read_jsonl(path(a.tag, "baseline"))}
    wrong = [c for c in sample if c["id"] in helpful and not base.get(c["id"], {}).get("correct", True)]
    p = path(a.tag, "helpful0")
    todo = [c for c in wrong if c["id"] not in done_keys(p, lambda r: r["case_id"])]
    print(f"helpful0: {len(todo)} baseline-wrong cases with a helpful note")
    for i, c in enumerate(todo):
        orig = build_llama_baseline_prompt(c["original_note"])
        full = build_llama_followup_prompt(c["original_note"], helpful[c["id"]]["helpful_note"])
        out = parse_llama_diagnosis(client.generate_ecd(orig, full, beta=0.0, max_tokens=a.max_tokens))
        v = judge.compare(f"{a.tag}-helpful", c["id"], c["diagnosis_ground_truth"], out, kind="grade")
        append_jsonl(p, {"case_id": c["id"], "source": c["source"], "beta": 0.0, "output": out, "correct": bool(v["match"]),
                         "incoherent": incoherent(out)})
        print(f"[helpful0 {i + 1}/{len(todo)}] {c['id']} fixed={v['match']}")


def phase_freeze_H(a):
    p = path(a.tag, "H")
    if p.exists() and not a.force:
        print(f"H already frozen at {p}")
        return
    rows = sorted(read_jsonl(path(a.tag, "helpful0")), key=lambda r: r["case_id"])
    random.Random(a.seed).shuffle(rows)
    H = rows[:CAP]
    p.write_text("".join(json.dumps({"case_id": r["case_id"], "source": r["source"]}) + "\n" for r in H), encoding="utf-8")
    print(f"H frozen: {len(H)} cases; U (helpful note fixed it at beta=0) = {sum(r['correct'] for r in H)}")


def phase_sweep(a, client, judge, sample, drift, skeptical: bool):
    check_prereg_committed()
    cases = {c["id"]: c for c in sample}
    notes = {(t["case_id"], t["category"]): t["adversarial_note"] for t in drift}
    helpful = {r["case_id"]: r["helpful_note"] for r in read_jsonl(DATA / "helpful_results.jsonl") if not r.get("excluded")}
    base = {r["case_id"]: r for r in read_jsonl(path(a.tag, "baseline"))}
    D = read_jsonl(path(a.tag, "D"))
    H = read_jsonl(path(a.tag, "H"))
    kind = "prompt" if skeptical else "sweep"
    betas = [0.0] if skeptical else BETAS
    p = path(a.tag, kind)
    done = done_keys(p, lambda r: (r["set"], r["case_id"], r.get("category"), r["beta"]))

    jobs = [("D", t["case_id"], t["category"], b) for t in D for b in betas]
    h_betas = [0.0] if skeptical else BETAS
    jobs += [("H", h["case_id"], None, b) for h in H for b in h_betas]
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"{kind}: {len(jobs)} generations left")
    for i, (s, cid, cat, b) in enumerate(jobs):
        c = cases[cid]
        note = notes[(cid, cat)] if s == "D" else helpful[cid]
        orig = build_llama_baseline_prompt(c["original_note"])
        full = (build_llama_followup_prompt_skeptical if skeptical else build_llama_followup_prompt)(c["original_note"], note)
        out = parse_llama_diagnosis(client.generate_ecd(orig, full, beta=b, max_tokens=a.max_tokens))
        if s == "D":
            v = judge.compare(f"{a.tag}-{kind}", f"{cid}|{cat}", base[cid]["output"], out, kind="same")
            row = {"recovered": bool(v["match"])}
        else:
            v = judge.compare(f"{a.tag}-{kind}-H", cid, c["diagnosis_ground_truth"], out, kind="grade")
            row = {"correct": bool(v["match"])}
        append_jsonl(p, {"set": s, "case_id": cid, "category": cat, "source": c["source"], "beta": b, "output": out,
                         "incoherent": incoherent(out), **row})
        print(f"[{kind} {i + 1}/{len(jobs)}] {s} {cid} beta={b}")


# ------------------------------------------------------------------ summary

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def cluster_boot(case_ids, flags, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: [0, 0])
    for c, f in zip(case_ids, flags):
        by[c][0] += int(f)
        by[c][1] += 1
    k = np.array([v[0] for v in by.values()], float)
    n = np.array([v[1] for v in by.values()], float)
    idx = rng.integers(0, len(k), size=(B, len(k)))
    rates = k[idx].sum(1) / n[idx].sum(1)
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def rate_block(rows, flag, B):
    k = sum(int(r[flag]) for r in rows)
    n = len(rows)
    return {"rate": round(k / n, 4) if n else None, "k": k, "n": n, "wilson_95": wilson(k, n),
            "cluster_boot_95": cluster_boot([r["case_id"] for r in rows], [r[flag] for r in rows], B) if n else [None, None],
            "n_cases": len({r["case_id"] for r in rows}), "incoherent_rate": round(float(np.mean([r["incoherent"] for r in rows])), 4) if n else None}


def phase_summarize(a):
    tag, B = a.tag, a.B
    out = {"script": "src/ecd_v2.py summarize", "model": a.model_name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seed": a.seed, "caps": CAP, "betas": BETAS, "prereg": "docs/prereg-v2.md"}
    base = read_jsonl(path(tag, "baseline"))
    out["baseline_accuracy"] = {s: {"accuracy": round(float(np.mean([r["correct"] for r in base if s == "all" or r["source"] == s])), 4),
                                    "n": sum(1 for r in base if s == "all" or r["source"] == s)}
                                for s in ["all"] + sorted({r["source"] for r in base})}
    D, H = read_jsonl(path(tag, "D")), read_jsonl(path(tag, "H"))
    out["set_sizes"] = {"D": len(D), "H": len(H), "underpowered_D_lt_60": len(D) < 60}
    sweep = read_jsonl(path(tag, "sweep")) if path(tag, "sweep").exists() else []
    h0 = {r["case_id"]: r for r in read_jsonl(path(tag, "helpful0"))}
    U_ids = {h["case_id"] for h in H if h0[h["case_id"]]["correct"]}
    out["set_sizes"]["U"] = len(U_ids)

    table = {}
    for b in BETAS:
        d = [r for r in sweep if r["set"] == "D" and r["beta"] == b]
        hh = [r for r in sweep if r["set"] == "H" and r["beta"] == b]
        u = [dict(r, suppressed=not r["correct"]) for r in hh if r["case_id"] in U_ids]
        entry = {"R": rate_block(d, "recovered", B) if d else None,
                 "S": rate_block(u, "suppressed", B) if u else None,
                 "accuracy_on_helpful": rate_block(hh, "correct", B) if hh else None}
        for scope_key in ("source", "category"):
            entry[f"R_by_{scope_key}"] = {k: rate_block([r for r in d if r[scope_key] == k], "recovered", B)
                                          for k in sorted({r[scope_key] for r in d} - {None})}
        entry["S_by_source"] = {k: rate_block([r for r in u if r["source"] == k], "suppressed", B) for k in sorted({r["source"] for r in u})}
        table[str(b)] = entry
    out["by_beta"] = table
    out["accuracy_on_helpful_beta0"] = rate_block([dict(r) for r in h0.values() if r["case_id"] in {h["case_id"] for h in H}], "correct", B) if H else None
    out["by_construction"] = {"R(0)": 0.0, "R(1)": 1.0, "S(0)": 0.0, "S(1)": 1.0,
                              "note": "these four are fixed by the definitions; the beta=1 point is the 'ignore the note' ceiling, not a result"}

    if path(tag, "prompt").exists():
        pr = read_jsonl(path(tag, "prompt"))
        d = [r for r in pr if r["set"] == "D"]
        u = [dict(r, suppressed=not r["correct"]) for r in pr if r["set"] == "H" and r["case_id"] in U_ids]
        out["prompt_baseline_P"] = {"R_P": rate_block(d, "recovered", B) if d else None, "S_P": rate_block(u, "suppressed", B) if u else None}

    useful = []
    for b in USEFUL["betas"]:
        e = table.get(str(b))
        if e and e["R"] and e["S"]:
            ok = (e["R"]["rate"] >= USEFUL["min_R"] and e["R"]["wilson_95"][0] >= USEFUL["min_wilson_lb"]
                  and e["S"]["rate"] <= USEFUL["max_S"])
            useful.append({"beta": b, "R": e["R"]["rate"], "R_wilson_lb": e["R"]["wilson_95"][0], "S": e["S"]["rate"], "meets_rule": bool(ok)})
    out["decision_rule"] = {"rule": USEFUL, "per_beta": useful,
                            "ECD_useful": any(u["meets_rule"] for u in useful) if useful else None,
                            "if_not": "report as a recovery-suppression tradeoff, no free lunch" if useful and not any(u["meets_rule"] for u in useful) else None}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ecd_v2_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("baseline_accuracy", "set_sizes", "decision_rule")}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = ROOT / "figures"
        figdir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        pts = [(table[str(b)]["S"]["rate"], table[str(b)]["R"]["rate"], b) for b in BETAS if table[str(b)]["R"] and table[str(b)]["S"]]
        if pts:
            ax.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], "o-", label="ECD (beta labelled)")
            ax.annotate("0", (0, 0))
            for s_, r_, b in pts:
                ax.annotate(str(b), (s_, r_), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.plot([1], [1], "ks", label="beta=1: ignore the note (ceiling)")
        P = out.get("prompt_baseline_P")
        if P and P["R_P"] and P["S_P"]:
            ax.plot([P["S_P"]["rate"]], [P["R_P"]["rate"]], "r*", ms=12, label="prompt baseline P")
        ax.set_xlabel("S: warranted updates suppressed"); ax.set_ylabel("R: original diagnosis recovered")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7); ax.set_title(f"{a.model_name.split('/')[-1]}")
        plt.tight_layout(); plt.savefig(figdir / f"ecd_RS_{tag}.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["baseline", "undefended", "freeze-D", "helpful0", "freeze-H", "sweep", "prompt", "summarize"])
    ap.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-8B")
    ap.add_argument("--sample", default=str(DATA / "eval_sample.jsonl"))
    ap.add_argument("--drift", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--judge-cache", default=str(DATA / "judge_cache_claude.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.tag = tag_of(a.model_name)

    if a.phase in ("freeze-D", "freeze-H", "summarize"):
        {"freeze-D": phase_freeze_D, "freeze-H": phase_freeze_H, "summarize": phase_summarize}[a.phase](a)
    else:
        from llm_clients import AnthropicClient, LlamaMedClient

        sample, drift = read_jsonl(a.sample), read_jsonl(a.drift)
        client = LlamaMedClient(model_name=a.model_name)
        judge = Judge(AnthropicClient(), "claude-binary", a.judge_cache)
        if a.phase == "baseline":
            phase_baseline(a, client, judge, sample)
        elif a.phase == "undefended":
            phase_undefended(a, client, judge, sample, drift)
        elif a.phase == "helpful0":
            phase_helpful0(a, client, judge, sample)
        else:
            phase_sweep(a, client, judge, sample, drift, skeptical=(a.phase == "prompt"))
        print("judge calls:", judge.calls)
