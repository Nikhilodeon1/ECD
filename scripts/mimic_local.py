"""MIMIC replication on the laptop (no GPU). Resumable, local only.

MIMIC-IV-Note rules (DUA, see docs/prereg-v3.md):
  - raw note text and every output derived from it stay on this machine under data/ (gitignored)
  - never Colab, never the shared cluster, never GitHub
  - cloud model calls only to Anthropic (zero data retention verified)
  - only aggregate summary JSONs under results/ may leave this machine

Tested model: Claude (budget-gated). Notes: a small open model on CPU (a different family from the
tested model). Verifier: a small open model of a third family. Judge: Claude, which is a deviation
from the preregistered open judge and is checked on public text (docs/prereg-v3.md, Deviations).

Every stage resumes from its own output file, so stop with Ctrl+C at any time and rerun the same
command. Notes are written in small chunks (--chunk 8) so an interruption loses at most a few minutes.

  python scripts/mimic_local.py --dry-run            print the stages
  python scripts/mimic_local.py                      run everything that is not finished
  python scripts/mimic_local.py --only notes verify  run some stages
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA = Path(os.environ.get("ECD_DATA_DIR", ROOT / "data" / "processed"))
SAMPLE = DATA / "mimic_sample.jsonl"
V3 = DATA / "v3"

GENERATOR = "hf:Qwen/Qwen2.5-3B-Instruct"        # CPU-sized, different family from Claude
VERIFIER = "open:microsoft/Phi-3.5-mini-instruct"  # third family
MODEL = "claude:claude-sonnet-5"                   # locked config of the main benchmark
JUDGE = "claude:claude-sonnet-5"
CATS = "ISO,VCS,SNHR"                              # the categories that moved answers on public text
ARMS = "R,N,ISO,VCS,SNHR,A_unattributed,B_attending,C_nursing"
ARMS_HEDGED = ",A_hedged,B_hedged_attending,C_hedged_nursing"  # exploratory; added with --with-hedged

# --api-only: no GPU, nothing leaves the laptop except calls to Anthropic (ZDR verified). Claude writes,
# verifies and judges, so notes and tested model share a family; logged in docs/prereg-v3.md and bounded
# by a public-text generator comparison. Fewer arms to fit the API budget.
API_GENERATOR = "claude:claude-haiku-5-5"
API_VERIFIER = "claude:claude-haiku-5-5"
API_JUDGE = "claude:claude-sonnet-5"
API_CATS = "ISO"
ARMS_API = "R,N,ISO,A_unattributed,B_attending,C_nursing"
STAGES = ["sample", "notes", "verify", "base", "judge_base", "arms", "judge_arms", "summary"]


def slug(spec: str) -> str:
    from v3_run import slug as s
    return s(spec)


def refuse_if_not_local() -> None:
    """MIMIC text must never be processed on Colab or a shared cluster."""
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content/drive").exists() or os.environ.get("SLURM_JOB_ID"):
        sys.exit("refusing to run: this looks like Colab or a cluster. MIMIC stays on the laptop.")


def plan(a) -> list[tuple[str, list[str]]]:
    py = sys.executable
    run = [py, str(SRC / "v3_run.py")]
    vf = V3 / f"verify__{slug(a.generator)}__{slug(a.verifier)}.jsonl"
    judged = V3 / f"judged__{slug(MODEL)}__{slug(a.judge)}.jsonl"
    common = ["--sample", str(SAMPLE)] + (["--limit", str(a.limit)] if a.limit else [])
    gen = ["--generator", a.generator, "--verify-file", str(vf)]
    return [
        ("sample", [py, "-c", "print('sample written by this script')"]),
        ("notes", run + ["notes", *common, "--generator", a.generator, "--cats", a.cats, "--no-helpful",
                         "--chunk", str(a.notes_chunk), "--batch-size", str(a.batch_size)]),
        ("verify", run + ["verify", *common, "--generator", a.generator, "--verifier", a.verifier,
                           "--chunk", str(a.notes_chunk)]),
        ("base", run + ["answers", *common, "--model", MODEL, "--arms", "BASE", *gen, "--budget-usd", str(a.budget_usd)]),
        ("judge_base", run + ["judge", *common, "--model", MODEL, "--judge", a.judge, "--budget-usd", str(a.budget_usd)]),
        ("arms", run + ["answers", *common, "--model", MODEL, "--arms", (ARMS_API if a.api_only else ARMS) + (ARMS_HEDGED if a.with_hedged else ""),
                        "--judge", a.judge, *gen,
                        "--budget-usd", str(a.budget_usd)]),
        ("judge_arms", run + ["judge", *common, "--model", MODEL, "--judge", a.judge, "--budget-usd", str(a.budget_usd)]),
        ("summary", [py, str(SRC / "v3_summary.py"), "--judged", str(judged)]),
    ]


def build_sample(src: Path, out: Path) -> int:
    """MIMIC rows of the existing eval sample -> one local file."""
    from paths import read_jsonl

    rows = [r for r in read_jsonl(src, guard=False) if r.get("source") == "mimic-iv-note"]
    if not rows:
        sys.exit(f"no MIMIC rows in {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", choices=STAGES)
    ap.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    ap.add_argument("--generator", default=GENERATOR, help="notes model spec (pod: hf:Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--verifier", default=VERIFIER, help="verifier spec (pod: open:microsoft/phi-4)")
    ap.add_argument("--judge", default=JUDGE, help="judge spec (pod: open:microsoft/phi-4, the preregistered open judge)")
    ap.add_argument("--cats", default=CATS)
    ap.add_argument("--notes-chunk", type=int, default=8, help="cases per saved chunk (pod: 64)")
    ap.add_argument("--batch-size", type=int, default=4, help="CPU generation batch")
    ap.add_argument("--budget-usd", type=float, default=5.0,
                    help="cumulative Claude cap (estimate from tokens) across all stages; raise it to resume later")
    ap.add_argument("--api-only", action="store_true",
                    help="laptop mode: Claude as generator, verifier and judge (no GPU, no downloads)")
    ap.add_argument("--with-hedged", action="store_true",
                    help="also run the three exploratory hedged framing arms (about $1 more)")
    ap.add_argument("--eval-sample", default=str(DATA / "eval_sample.jsonl"))
    a = ap.parse_args()

    refuse_if_not_local()
    if a.api_only:
        a.generator, a.verifier, a.judge, a.cats = API_GENERATOR, API_VERIFIER, API_JUDGE, API_CATS
    env = {**os.environ, "ECD_ALLOW_MIMIC": "1", "ECD_HF_DTYPE": os.environ.get("ECD_HF_DTYPE", "bfloat16")}
    todo = a.only or STAGES
    for label, argv in plan(a):
        if label not in todo:
            continue
        print(f"\n=== {label}\n{' '.join(argv)}", flush=True)
        if a.dry_run:
            continue
        if label == "sample":
            print(f"{build_sample(Path(a.eval_sample), SAMPLE)} MIMIC cases -> {SAMPLE.name}")
            continue
        rc = subprocess.run(argv, env=env).returncode
        if rc != 0:
            sys.exit(f"stage '{label}' failed (exit {rc}); finished work is kept, rerun to resume")


if __name__ == "__main__":
    main()
