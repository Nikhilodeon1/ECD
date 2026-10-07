"""Run on the RunPod volume (ECD_ALLOW_MIMIC=1). Writes a MedQA-ONLY copy of the
result files so they can be moved to a shared cluster. MedQA is public, MIMIC is not.

    python scripts/export_public_subset.py        # -> data/public_export/ + ecd_public_medqa.tgz

Every output row is asserted non-MIMIC before it is written.
"""
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import DATA, is_mimic  # noqa: E402

FILES = ["eval_sample.jsonl", "eval_results_claude.jsonl", "drift_results_claude.jsonl"]
OUT = ROOT / "data" / "public_export"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in FILES:
        src = DATA / name
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        kept = 0
        with src.open(encoding="utf-8") as f, (OUT / name).open("w", encoding="utf-8") as g:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if is_mimic(row) or row.get("source") not in (None, "medqa"):
                    continue
                assert "MIMIC" not in json.dumps(row)[:400].upper() or row.get("source") == "medqa", "possible MIMIC leak"
                g.write(json.dumps(row) + "\n")
                kept += 1
        counts[name] = kept
        print(f"{name}: kept {kept} MedQA rows")
    tgz = ROOT / "ecd_public_medqa.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        for name in counts:
            t.add(OUT / name, arcname=f"processed/{name}")
    print(f"wrote {tgz} ({tgz.stat().st_size/1e3:.0f} KB). Unpack on the shared pod into <repo>/data/ :  tar xzf ecd_public_medqa.tgz -C <repo>/data/")


if __name__ == "__main__":
    main()
