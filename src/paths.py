"""Location-independent paths and the MIMIC guard.

Nothing here depends on the working directory or on where the repo is cloned.

ECD_DATA_DIR    where data/processed-style files live (default <repo>/data/processed)
ECD_RESULTS_DIR aggregate outputs (default <repo>/results)
ECD_ALLOW_MIMIC set to 1 ONLY on a cleared host (the RunPod volume). Unset, every
                loader here drops MIMIC-derived rows, so MIMIC text cannot reach a
                shared cluster by accident.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ECD_DATA_DIR", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("ECD_RESULTS_DIR", ROOT / "results"))
ALLOW_MIMIC = os.environ.get("ECD_ALLOW_MIMIC") == "1"


def is_mimic(row: dict) -> bool:
    if row.get("source") == "mimic-iv-note":
        return True
    ident = str(row.get("id") or row.get("case_id") or "")
    return ident.upper().startswith("MIMIC")


def guard_rows(rows: list[dict], what: str = "rows") -> list[dict]:
    """Drop MIMIC-derived rows unless ECD_ALLOW_MIMIC=1."""
    if ALLOW_MIMIC:
        return rows
    kept = [r for r in rows if not is_mimic(r)]
    if len(kept) != len(rows):
        print(f"[guard] ECD_ALLOW_MIMIC not set: dropped {len(rows) - len(kept)} MIMIC {what}", file=sys.stderr)
    return kept


def read_jsonl(path, guard: bool = True) -> list[dict]:
    import json

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return guard_rows(rows, f"rows from {Path(path).name}") if guard else rows


def append_jsonl(path, row: dict) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def data_path(name: str) -> Path:
    return DATA / name


def results_path(name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / name
