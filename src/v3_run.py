"""v3 pipeline (protocol: docs/prereg-v3.md). Every stage resumes from its output file.

  notes    generator model: one wrong diagnosis X per case, five validated category notes, one
           helpful note (note_spec rules, up to --max-tries regenerations, all attempts logged)
  verify   a second model (different family from the generator) signs off on X and each note
  answers  tested model: BASE (all cases), then the arms on baseline-correct cases
           (R re-ask, N neutral, H helpful [all cases], five categories, three framing arms)
  judge    judge model: baseline correctness vs gold, and arm answer vs baseline answer

Stages are separate commands so only one model is in GPU memory at a time. Typical order:
  notes -> verify -> answers --arms BASE -> judge -> answers --arms REST -> judge -> v3_summary

Backends are chosen with --model/--generator/--judge/--verifier specs:
  claude:<model-id>      Anthropic API (budget-gated by --budget-usd; MIMIC text allowed, ZDR)
  hf:<repo>              open-weight chat model, local GPU (ChatClient, batched)
  open:<repo>            open-weight judge/verifier (OpenJudgeClient, primed)
MIMIC rows are dropped unless ECD_ALLOW_MIMIC=1 (paths.py), i.e. only on a cleared host.
Outputs hold note text, so they live under data/ and never go to git.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from controls import HELPFUL_PROMPT, leaks_diagnosis, neutral_for
from judge import Judge
from note_spec import (FRAMING_PREFIX, HEDGED_PREFIX, SPECS, VERIFY_NOTE_PROMPT, VERIFY_WRONG_DX_PROMPT,
                       build_note_prompt, build_wrong_dx_prompt, check_note, check_wrong_dx, framing_notes,
                       hedged_notes, parse_yes_no)
from paths import DATA, RESULTS, append_jsonl, read_jsonl
from prompts import build_baseline_prompt, build_followup_prompt, parse_diagnosis

V3 = DATA / "v3"
CATS = tuple(SPECS)
FRAMING = tuple(FRAMING_PREFIX)
HEDGED = tuple(HEDGED_PREFIX)  # secondary, exploratory (see note_spec)
ARMS = ("R", "N", "H") + CATS + FRAMING + HEDGED
EVIDENCE_WORDS = 800


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", spec).strip("-")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def evidence_of(case: dict) -> str:
    return " ".join(case["original_note"].split()[:EVIDENCE_WORDS])


def chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def done_keys(path: Path, keys: tuple[str, ...]) -> set[tuple]:
    if not Path(path).exists():
        return set()
    return {tuple(r[k] for k in keys) for r in read_jsonl(path, guard=False)}


# ---------------------------------------------------------------- backends and budget

class BudgetExceeded(RuntimeError):
    pass


class Budgeted:
    """Wraps AnthropicClient. Refuses calls once cumulative v3 spend (an estimate from token
    counts at the given rates) reaches the cap. Prior spend is read from the v3 manifest."""

    parallel_ok = True  # API calls may be threaded; local GPU backends may not

    def __init__(self, inner, cap_usd: float, manifest, in_rate: float = 3.0, out_rate: float = 15.0):
        self.inner, self.cap, self.manifest = inner, cap_usd, Path(manifest)
        self.in_rate, self.out_rate = in_rate, out_rate
        self.prior = 0.0
        if self.manifest.exists():
            with self.manifest.open(encoding="utf-8") as f:
                self.prior = sum(json.loads(line).get("cost_usd_estimate", 0.0) for line in f if line.strip())

    def run_cost(self) -> float:
        m = self.inner.meta
        return (m["input_tokens"] * self.in_rate + m["output_tokens"] * self.out_rate) / 1e6

    def spent(self) -> float:
        return self.prior + self.run_cost()

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        if self.spent() >= self.cap:
            raise BudgetExceeded(f"estimated v3 spend ${self.spent():.2f} reached the cap ${self.cap:.2f}")
        return self.inner.generate(prompt, max_tokens=max_tokens)

    def close(self) -> None:
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**self.inner.meta, "cost_usd_estimate": round(self.run_cost(), 4),
                                "timestamp": now()}) + "\n")


class Lazy:
    """Defers building a backend until it is first used, so a stage with nothing left to do
    never loads a model (matters when a Colab session resumes a finished stage)."""

    def __init__(self, factory):
        self._factory, self._built = factory, None

    def __getattr__(self, name):  # only reached for attributes not set in __init__
        if self._built is None:
            self._built = self._factory()
        return getattr(self._built, name)


def make_backend(spec: str, budget_usd: float | None = None, prime: str | None = None, batch_size: int = 16):
    kind, _, name = spec.partition(":")
    if kind == "claude":
        from llm_clients import AnthropicClient

        client = AnthropicClient(name or "claude-sonnet-5")
        return Budgeted(client, budget_usd if budget_usd is not None else 9.0, RESULTS / "api_manifest_v3.jsonl")
    if kind == "hf":
        from llm_clients import ChatClient

        return ChatClient(name, batch_size=batch_size)
    if kind == "open":
        from llm_clients import OpenJudgeClient

        return OpenJudgeClient(name, **({"prime": prime} if prime else {}))
    raise ValueError(f"unknown backend spec {spec!r}")


def gen_many(backend, prompts: list[str], max_tokens: int, workers: int = 8) -> list[str]:
    if hasattr(backend, "generate_batch"):
        return backend.generate_batch(prompts, max_tokens=max_tokens)
    if workers <= 1 or len(prompts) <= 1 or not getattr(backend, "parallel_ok", False):
        return [backend.generate(p, max_tokens=max_tokens) for p in prompts]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: backend.generate(p, max_tokens=max_tokens), prompts))


def is_claude(spec: str) -> bool:
    return spec.startswith("claude:")


# ---------------------------------------------------------------- notes

def run_validated(gen, items: list[dict], build_prompt, check, max_tries: int, max_tokens: int = 100) -> dict:
    """items carry a 'key'. build_prompt(item, failed|None) -> prompt; check(item, text) -> failed rules.
    Returns key -> {text, passed, attempts}; whole batches are regenerated round by round."""
    state = {it["key"]: {"text": "", "passed": False, "attempts": []} for it in items}
    feedback: dict = {}
    pending = list(items)
    for _ in range(max_tries):
        if not pending:
            break
        outs = gen_many(gen, [build_prompt(it, feedback.get(it["key"])) for it in pending], max_tokens)
        nxt = []
        for it, out in zip(pending, outs):
            text = out.strip().strip('"').strip()
            failed = check(it, text)
            st = state[it["key"]]
            st["text"] = text
            st["attempts"].append(failed)
            if failed:
                feedback[it["key"]] = failed
                nxt.append(it)
            else:
                st["passed"] = True
        pending = nxt
    return state


def _write_notes(path, cases_by_id, state, generator):
    for key, st in state.items():
        cid, kind = key
        append_jsonl(path, {"case_id": cid, "source": cases_by_id[cid]["source"], "kind": kind,
                            "text": st["text"], "passed": st["passed"], "n_attempts": len(st["attempts"]),
                            "attempts": st["attempts"], "generator": generator, "timestamp": now()})


def run_notes(cases: list[dict], gen, out_path, generator: str, max_tries: int = 3, chunk: int = 64) -> None:
    out_path = Path(out_path)
    by_id = {c["id"]: c for c in cases}

    def pending(kind, ids=None):
        done = done_keys(out_path, ("case_id", "kind")) if out_path.exists() else set()
        return [c for c in cases if (c["id"], kind) not in done and (ids is None or c["id"] in ids)]

    # 1. wrong diagnosis X
    for part in chunks(pending("X"), chunk):
        items = [{"key": (c["id"], "X"), "case": c} for c in part]
        state = run_validated(
            gen, items, lambda it, f: build_wrong_dx_prompt(evidence_of(it["case"]), it["case"]["diagnosis_ground_truth"]),
            lambda it, t: [k for k, ok in check_wrong_dx(t, it["case"]["diagnosis_ground_truth"]).items() if not ok],
            max_tries, max_tokens=24)
        _write_notes(out_path, by_id, state, generator)

    if not out_path.exists():
        return
    xs = {r["case_id"]: r["text"] for r in read_jsonl(out_path, guard=False) if r["kind"] == "X" and r["passed"]}

    # 2. five categories (need X)
    for cat in CATS:
        for part in chunks(pending(cat, set(xs)), chunk):
            items = [{"key": (c["id"], cat), "case": c, "x": xs[c["id"]]} for c in part]
            state = run_validated(
                gen, items,
                lambda it, f, cat=cat: build_note_prompt(evidence_of(it["case"]), it["case"]["diagnosis_ground_truth"],
                                                          it["x"], cat, f),
                lambda it, t, cat=cat: [k for k, ok in check_note(cat, t, it["case"]["diagnosis_ground_truth"]).items() if not ok],
                max_tries)
            _write_notes(out_path, by_id, state, generator)

    # 3. helpful note (reuses the v2 prompt and diagnosis-leak check)
    for part in chunks(pending("H"), chunk):
        items = [{"key": (c["id"], "H"), "case": c} for c in part]
        state = run_validated(
            gen, items,
            lambda it, f: HELPFUL_PROMPT.format(evidence=evidence_of(it["case"]), diagnosis=it["case"]["diagnosis_ground_truth"]),
            lambda it, t: (["empty"] if not t else []) + (["no_gold_diagnosis"] if leaks_diagnosis(t, it["case"]["diagnosis_ground_truth"]) else []),
            max_tries, max_tokens=120)
        _write_notes(out_path, by_id, state, generator)


# ---------------------------------------------------------------- verify

def run_verify(cases: list[dict], notes_path, verifier, out_path, chunk: int = 64) -> None:
    by_id = {c["id"]: c for c in cases}
    done = done_keys(out_path, ("case_id", "kind"))
    rows = [r for r in read_jsonl(notes_path, guard=False)
            if r["passed"] and r["case_id"] in by_id and (r["case_id"], r["kind"]) not in done and r["kind"] != "H"]
    for part in chunks(rows, chunk):
        prompts = []
        for r in part:
            gold = by_id[r["case_id"]]["diagnosis_ground_truth"]
            prompts.append(VERIFY_WRONG_DX_PROMPT.format(gold=gold, wrong_dx=r["text"]) if r["kind"] == "X"
                           else VERIFY_NOTE_PROMPT.format(spec=SPECS[r["kind"]][0], note=r["text"]))
        for r, out in zip(part, gen_many(verifier, prompts, 8)):
            append_jsonl(out_path, {"case_id": r["case_id"], "kind": r["kind"], "verified": parse_yes_no(out),
                                    "timestamp": now()})


def usable_notes(notes_path, verify_path=None) -> dict[tuple, str]:
    """(case_id, kind) -> text for notes that passed the rules (and the verifier, if run)."""
    verdict = {}
    if verify_path and Path(verify_path).exists():
        verdict = {(r["case_id"], r["kind"]): r["verified"] for r in read_jsonl(verify_path, guard=False)}
    out = {}
    for r in read_jsonl(notes_path, guard=False):
        key = (r["case_id"], r["kind"])
        if r["passed"] and (not verdict or verdict.get(key) is True or r["kind"] == "H"):
            out[key] = r["text"]
    return out


# ---------------------------------------------------------------- answers

def judged_correct(judged_path) -> dict[str, bool]:
    if not judged_path or not Path(judged_path).exists():
        return {}
    return {r["case_id"]: bool(r["correct"]) for r in read_jsonl(judged_path, guard=False) if r["arm"] == "BASE"}


def arm_prompt(arm: str, case: dict, notes: dict) -> str | None:
    cid, ev = case["id"], case["original_note"]
    if arm == "R":
        return build_baseline_prompt(ev)
    if arm == "N":
        return build_followup_prompt(ev, neutral_for(cid))
    if arm in FRAMING or arm in HEDGED:
        x = notes.get((cid, "X"))
        if not x:
            return None
        return build_followup_prompt(ev, {**framing_notes(x), **hedged_notes(x)}[arm])
    note = notes.get((cid, arm))
    return build_followup_prompt(ev, note) if note else None


def run_answers(cases: list[dict], model, model_spec: str, arms: list[str], notes: dict, judged_path,
                out_path, chunk: int = 64) -> None:
    claude = is_claude(model_spec)
    done = done_keys(out_path, ("case_id", "arm"))
    correct = judged_correct(judged_path)
    for arm in arms:
        if arm == "BASE":
            todo = [c for c in cases if (c["id"], "BASE") not in done]
        else:
            if not correct:
                raise SystemExit("no judged baseline yet: run answers --arms BASE, then judge")
            todo = [c for c in cases if (c["id"], arm) not in done and c["id"] in correct
                    and (arm == "H" or correct[c["id"]])]
        prompts = [(c, build_baseline_prompt(c["original_note"]) if arm == "BASE" else arm_prompt(arm, c, notes)) for c in todo]
        prompts = [(c, p) for c, p in prompts if p]
        tok = (500 if arm in ("BASE", "R") else 300) if claude else 160
        for part in chunks(prompts, chunk):
            outs = gen_many(model, [p for _, p in part], tok)
            for (c, _), out in zip(part, outs):
                append_jsonl(out_path, {"case_id": c["id"], "source": c["source"], "arm": arm, "answer": out,
                                        "timestamp": now()})


# ---------------------------------------------------------------- judge

def run_judge(cases: list[dict], answers_path, judge: Judge, out_path, workers: int = 1) -> None:
    by_id = {c["id"]: c for c in cases}
    done = done_keys(out_path, ("case_id", "arm"))
    answers = {(r["case_id"], r["arm"]): r for r in read_jsonl(answers_path, guard=False)}
    base_pred = {cid: parse_diagnosis(r["answer"]) for (cid, arm), r in answers.items() if arm == "BASE"}

    def work(key):
        cid, arm = key
        gold = by_id[cid]["diagnosis_ground_truth"]
        pred = parse_diagnosis(answers[key]["answer"])
        row = {"case_id": cid, "source": by_id[cid]["source"], "arm": arm, "timestamp": now()}
        if arm == "BASE":
            row["correct"] = bool(judge.compare("v3base", cid, gold, pred, kind="grade")["match"])
        else:
            row["same_as_base"] = bool(judge.compare(f"v3{arm}", cid, base_pred[cid], pred, kind="same")["match"])
            if arm == "H":
                row["correct_after"] = bool(judge.compare("v3H", cid, gold, pred, kind="grade")["match"])
        return row

    def todo_keys():
        return [k for k in answers if k not in done and k[0] in by_id and (k[1] == "BASE" or k[0] in base_pred)]

    # BASE first so arm rows have a reference; workers apply within each pass
    for first in (True, False):
        keys = [k for k in todo_keys() if (k[1] == "BASE") == first]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(work, keys))
        else:
            rows = [work(k) for k in keys]
        for row in rows:
            append_jsonl(out_path, row)
        done = done_keys(out_path, ("case_id", "arm"))


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["notes", "verify", "answers", "judge"])
    ap.add_argument("--sample", required=True, help="jsonl of cases (id, source, original_note, diagnosis_ground_truth)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--generator", help="spec for notes")
    ap.add_argument("--verifier", help="spec for verify (different family from the generator)")
    ap.add_argument("--model", help="spec of the tested model (answers, judge file naming)")
    ap.add_argument("--judge", help="spec of the judge")
    ap.add_argument("--arms", default="BASE", help="BASE | REST | comma list")
    ap.add_argument("--notes", help="notes file (default derived from --generator)")
    ap.add_argument("--verify-file", help="verify file; if given, only verified notes are used")
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16, help="GPU batch for hf: models (A100 40GB: 48 for 8B, 16 for 14B)")
    ap.add_argument("--budget-usd", type=float, default=9.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = read_jsonl(args.sample)
    if args.limit:
        cases = cases[: args.limit]
    V3.mkdir(parents=True, exist_ok=True)
    notes_path = Path(args.notes) if args.notes else (V3 / f"notes__{slug(args.generator)}.jsonl" if args.generator else None)
    judged_path = V3 / f"judged__{slug(args.model)}__{slug(args.judge)}.jsonl" if args.model and args.judge else None
    backend = None
    try:
        if args.stage == "notes":
            if args.dry_run:
                print(f"notes: {len(cases)} cases -> about {len(cases) * 7} generations (+ retries)"); return
            backend = Lazy(lambda: make_backend(args.generator, args.budget_usd, batch_size=args.batch_size))
            run_notes(cases, backend, notes_path, args.generator, args.max_tries, args.chunk)
        elif args.stage == "verify":
            backend = Lazy(lambda: make_backend(args.verifier, args.budget_usd, prime="Answer:", batch_size=args.batch_size))
            run_verify(cases, notes_path, backend, V3 / f"verify__{slug(args.generator)}__{slug(args.verifier)}.jsonl", args.chunk)
        elif args.stage == "answers":
            notes = usable_notes(notes_path, args.verify_file) if notes_path and Path(notes_path).exists() else {}
            arms = list(ARMS) if args.arms == "REST" else args.arms.split(",")
            if args.dry_run:
                print(f"answers: arms {arms}, {len(cases)} cases, {len(notes)} usable notes"); return
            backend = Lazy(lambda: make_backend(args.model, args.budget_usd, batch_size=args.batch_size))
            run_answers(cases, backend, args.model, arms, notes, judged_path, V3 / f"answers__{slug(args.model)}.jsonl", args.chunk)
        else:
            backend = Lazy(lambda: make_backend(args.judge, args.budget_usd, batch_size=args.batch_size))
            judge = Judge(backend, slug(args.judge), V3 / f"judge_cache__{slug(args.judge)}.jsonl")
            run_judge(cases, V3 / f"answers__{slug(args.model)}.jsonl", judge, judged_path,
                      workers=args.workers if is_claude(args.judge) else 1)
            print(f"judge calls {judge.calls}")
    finally:
        real = getattr(backend, "_built", backend) if isinstance(backend, Lazy) else backend
        if isinstance(real, Budgeted):
            print(f"estimated Claude spend this run ${real.run_cost():.2f}, cumulative ${real.spent():.2f}")
            real.close()


if __name__ == "__main__":
    main()
