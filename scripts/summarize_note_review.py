"""Tabulate the labeled note-review sheet into counts only (no note text).

    python scripts/summarize_note_review.py [path/to/note_review_sheet.csv]

Default sheet: data/audit/note_review_sheet.csv (contains MIMIC-derived note text: keep it
on the laptop, never commit it). Output: results/note_review_summary.json (aggregates only).
Label each of the four columns y or n; every row must be filled.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["matches_category", "adds_new_clinical_fact_beyond_rule", "chart_plausible", "harmful_implication"]
YES, NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def regex_agreement(rows):
    """Agreement of matches_category with the pattern audit (needs the drift file, MIMIC allowed)."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import note_audit as na
        from paths import DATA, read_jsonl

        trials = {t["case_id"] + "|" + t["category"]: t for t in read_jsonl(str(DATA / "drift_results_claude.jsonl"))}
        out, tot = {}, 0
        for r in rows:
            t = trials[r["item_id"]]
            ok = all(na.check_note(t["category"], t["adversarial_note"], t["diagnosis_before"]).values())
            agree = ok == (r["matches_category"].strip().lower() in YES)
            out.setdefault(r["category"], [0, 0])
            out[r["category"]][0] += int(agree)
            out[r["category"]][1] += 1
            tot += int(agree)
        return {"agree": tot, "n": len(rows), "by_category": {k: f"{a}/{n}" for k, (a, n) in sorted(out.items())}}
    except Exception as e:  # not available off the cleared laptop
        return {"unavailable": str(e)[:80]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(ROOT / "data" / "audit" / "note_review_sheet.csv"))
    ap.add_argument("--annotator", default="single non-clinician human")
    ap.add_argument("--out", default=str(ROOT / "results" / "note_review_summary.json"))
    a = ap.parse_args()
    path = Path(a.path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:  # category is encoded in item_id ("case|CATEGORY") when the column is absent
        r.setdefault("category", r["item_id"].split("|")[-1])
    bad = [(r["item_id"], c) for r in rows for c in COLS if r[c].strip().lower() not in YES | NO]
    if bad:
        print(f"{len(bad)} unlabeled or invalid cells (use y or n), e.g. {bad[:3]}")
        return 1
    count = defaultdict(lambda: defaultdict(int))
    n_cat = defaultdict(int)
    for r in rows:
        n_cat[r["category"]] += 1
        for c in COLS:
            if r[c].strip().lower() in YES:
                count[r["category"]][c] += 1
                count["ALL"][c] += 1
    n_cat["ALL"] = len(rows)
    out = {"script": "scripts/summarize_note_review.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "annotator": a.annotator, "n": len(rows),
           "yes_counts": {k: {"n": n_cat[k], **dict(count[k])} for k in sorted(n_cat)},
           "agreement_with_pattern_audit": regex_agreement(rows)}
    dest = Path(a.out)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    a = out["yes_counts"]["ALL"]
    print(f"{a.get('matches_category', 0)}/{len(rows)} matched their category; "
          f"{a.get('adds_new_clinical_fact_beyond_rule', 0)}/{len(rows)} added a new clinical fact beyond the rule; "
          f"{a.get('chart_plausible', 0)}/{len(rows)} chart-plausible; "
          f"{a.get('harmful_implication', 0)}/{len(rows)} harmful implication")
    print(json.dumps(out["yes_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
