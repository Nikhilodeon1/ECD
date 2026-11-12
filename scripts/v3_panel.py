"""Run the whole v3 sequence for a panel of models on Colab, syncing to Google Drive.

Work happens on local disk (fast, and a Drive FUSE mount can lose buffered appends when a
runtime dies); this script copies the outputs to Drive after every stage and restores them from
Drive at the start, so a new session continues where the last one stopped. Every v3_run stage
also resumes from its own output file and loads no model when it has nothing left to do.

  ECD_DATA_DIR / ECD_RESULTS_DIR   local working dirs (for example /content/ecd_data)
  --drive-dir                      Drive folder holding the synced copies

Public text only. Never set ECD_ALLOW_MIMIC here.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from v3_run import slug  # noqa: E402


def sync(src: Path, dst: Path) -> int:
    """Copy new or changed files from src to dst (size or mtime differs). Returns files copied."""
    n = 0
    if not src.exists():
        return 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        t = dst / f.relative_to(src)
        if t.exists() and t.stat().st_size == f.stat().st_size and t.stat().st_mtime >= f.stat().st_mtime:
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
        n += 1
    return n


def plan(a, data: Path, results: Path) -> list[tuple[str, list[str]]]:
    py, run = sys.executable, [sys.executable, str(SRC / "v3_run.py")]
    common = ["--sample", a.sample, "--batch-size", str(a.batch_size)] + (["--limit", str(a.limit)] if a.limit else [])
    vf = str(data / "v3" / f"verify__{slug(a.generator)}__{slug(a.verifier)}.jsonl")
    cmds = [("notes", run + ["notes", *common, "--generator", a.generator]),
            ("verify", run + ["verify", *common, "--generator", a.generator, "--verifier", a.verifier])]
    for m in a.models:
        gen_args = ["--generator", a.generator, "--verify-file", vf]
        judged = str(data / "v3" / f"judged__{slug(m)}__{slug(a.judge)}.jsonl")
        cmds += [(f"{m} baseline answers", run + ["answers", *common, "--model", m, "--arms", "BASE", *gen_args]),
                 (f"{m} judge baseline", run + ["judge", *common, "--model", m, "--judge", a.judge]),
                 (f"{m} arm answers", run + ["answers", *common, "--model", m, "--arms", "REST", "--judge", a.judge, *gen_args]),
                 (f"{m} judge arms", run + ["judge", *common, "--model", m, "--judge", a.judge]),
                 (f"{m} summary", [py, str(SRC / "v3_summary.py"), "--judged", judged])]
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--drive-dir", help="Drive folder for synced copies, e.g. /content/drive/MyDrive/ecd")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip() or "no GPU found"
        print(f"GPU: {gpu}")
        try:
            mem_mib = int(gpu.split(",")[-1].strip().split()[0])
        except (ValueError, IndexError):
            mem_mib = 0
        if mem_mib < 38000:
            print("WARNING: under 38 GB of GPU memory. 7-9B models in bf16/fp16 need about 16 GB for weights alone, "
                  "so they barely fit a 15 GB T4 and cannot batch; 14B models need about 30 GB. Switch the runtime "
                  "(Runtime > Change runtime type) to an A100 or G4 before continuing.")

    data, results = Path(os.environ["ECD_DATA_DIR"]), Path(os.environ["ECD_RESULTS_DIR"])
    drive = Path(a.drive_dir) if a.drive_dir else None
    if drive and not a.dry_run:
        print(f"restored {sync(drive / 'data', data)} data and {sync(drive / 'results', results)} result files from Drive")
    for label, argv in plan(a, data, results):
        print(f"\n=== {label}\n{' '.join(argv)}")
        if a.dry_run:
            continue
        rc = subprocess.run(argv).returncode
        if drive:
            sync(data, drive / "data"); sync(results, drive / "results")  # keep Drive current even after a failure
        if rc != 0:
            sys.exit(f"stage '{label}' failed (exit {rc}); fix and rerun, finished work is kept")


if __name__ == "__main__":
    main()
