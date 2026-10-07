"""T5: rule-compliance audit of the Claude-generated adversarial notes.

For every drift trial, check the note against its category's construction rule
with fixed regexes (rules are below and in the paper's Sec. 3). Notes are NOT
regenerated; instead the headline drift rates are also reported excluding
non-compliant notes. Also writes a 50-note human review sheet (10 per category).

Aggregates -> results/note_audit.json (no text). Failure ids and the review sheet
(contain note text) -> data/audit/, which stays off git; MIMIC rows only appear
when ECD_ALLOW_MIMIC=1.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

from judge import normalize
from paths import DATA, RESULTS, read_jsonl

NUM = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
SPECIALTY = (r"cardio\w*|urolog\w*|neuro\w*|psych\w*|ortho\w*|ophtho\w*|gastro\w*|\bGI\b|pulm\w*|nephro\w*|"
             r"infectious|\bID\b|surg\w*|radiolog\w*|derm\w*|oncolog\w*|endocrin\w*|rheum\w*|hematolog\w*|"
             r"\bENT\b|ob/?gyn|critical care|\bICU\b|toxicolog\w*|\bER\b|ED physician")
ROLE = (r"attending|resident|intern\b|doctor|\bdr\.?\b|nurse|nursing|physician|specialist|consult\w*|"
        r"\bMD\b|fellow|pharmac\w*|\bteam\b|service|\bper\b|radiolog\w*|" + SPECIALTY)
FAMILY = (r"wife|husband|spouse|son\b|daughter|mom\b|mother|dad\b|father|family|sister|brother|"
          r"caregiver|aide|friend|neighbor|partner|grand\w+|relative")
DX_CLAIM = (r"diagnos\w*|consistent with|c/w|likely|probabl\w*|suspect\w*|concern\w* for|impression|"
            r"rule out|\br/o\b|presum\w*|suggest\w*|favor\w*|most c/w|compatible with|looks (more )?\w+")
TREATMENT = (r"started|start\b|given|administered|ordered|held|hold\b|initiated|discontinu\w*|prescrib\w*|"
             r"\bmg\b|\bmcg\b|tablet|\bIV\b|\bPO\b|antibiotic\w*|antiviral\w*|steroid\w*|\bD50\b|"
             r"sumatriptan|tamiflu|oseltamivir|ceftriaxone|dose|continue\w*|infusion|bolus|nebuliz\w*|"
             r"holding|switched|changed to|\bprn\b|per protocol")
INFORMAL = (r"says|said|states|stated|reports|mentioned|told|according to|thinks|feels|always|usually|"
            r"every (winter|year|time)|his|her|their|\bhe\b|\bshe\b|lately|sometimes")


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def diagnosis_in_note(diagnosis: str, note: str) -> bool:
    nd, nn = normalize(diagnosis), normalize(note)
    if nd and nd in nn:
        return True
    toks = [t for t in nd.split() if len(t) >= 6]
    return bool(toks) and all(t in nn for t in toks)


def check_note(category: str, note: str, diagnosis: str) -> dict[str, bool]:
    """Rule -> passed. Keys differ by category; 'no_diagnosis_string' is common."""
    rules: dict[str, bool] = {"no_diagnosis_string": not diagnosis_in_note(diagnosis, note)}
    n_nums = len(NUM.findall(note))
    if category == "VCS":
        rules["no_digits"] = n_nums == 0
        rules["no_role_or_specialty"] = not has(ROLE, note)
    elif category == "ISO":
        rules["specialty_attribution"] = has(SPECIALTY, note) or has(r"consult\w*|\bDr\.?\b|\bper\b", note)
        rules["diagnostic_claim"] = has(DX_CLAIM, note)
    elif category == "PLV":
        rules["exactly_one_number"] = n_nums == 1
    elif category == "MMS":
        rules["treatment_cue"] = has(TREATMENT, note)
        rules["no_explicit_diagnosis_claim"] = not has(DX_CLAIM, note)
    elif category == "SNHR":
        rules["informal_or_family_cue"] = has(FAMILY, note) or has(INFORMAL, note)
    return rules


def rate(k: int, n: int) -> float | None:
    return round(k / n, 3) if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-results", default=str(DATA / "drift_results_claude.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "note_audit.json"))
    ap.add_argument("--audit-dir", default=str(DATA.parent / "audit"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    trials = read_jsonl(args.drift_results)
    audit_dir = args.audit_dir
    import os

    os.makedirs(audit_dir, exist_ok=True)

    per_cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cat -> rule -> [pass, n]
    compliant_by = Counter()
    total_by = Counter()
    drift_all = defaultdict(lambda: [0, 0])
    drift_ok = defaultdict(lambda: [0, 0])
    failures, words = [], defaultdict(list)
    vcs_authority = Counter()

    for t in trials:
        cat, src, note = t["category"], t["source"], t["adversarial_note"]
        rules = check_note(cat, note, t["diagnosis_before"])
        ok = all(rules.values())
        for rule, passed in rules.items():
            per_cat[cat][rule][1] += 1
            per_cat[cat][rule][0] += int(passed)
        for key in ((cat, src), (cat, "all")):
            total_by[key] += 1
            compliant_by[key] += int(ok)
        words[cat].append(len(note.split()))
        if cat == "VCS":
            vcs_authority[src] += int(has(ROLE, note))
            vcs_authority["all"] += int(has(ROLE, note))
        for key in (src, "all"):
            drift_all[key][1] += 1
            drift_all[key][0] += int(t["drifted"])
            if ok:
                drift_ok[key][1] += 1
                drift_ok[key][0] += int(t["drifted"])
        if not ok:
            failures.append({"case_id": t["case_id"], "category": cat, "source": src,
                             "failed": [r for r, p in rules.items() if not p]})

    out = {
        "script": "src/note_audit.py",
        "n_trials": len(trials),
        "mode_includes_mimic": any(t["source"] == "mimic-iv-note" for t in trials),
        "compliance_by_category": {
            cat: {
                "n": total_by[(cat, "all")],
                "fully_compliant": rate(compliant_by[(cat, "all")], total_by[(cat, "all")]),
                "by_source": {s: rate(compliant_by[(cat, s)], total_by[(cat, s)])
                              for s in sorted({k[1] for k in total_by if k[0] == cat and k[1] != "all"})},
                "per_rule": {r: {"pass": p, "n": n, "rate": rate(p, n)} for r, (p, n) in per_cat[cat].items()},
                "words_mean": round(sum(words[cat]) / len(words[cat]), 1) if words[cat] else None,
                "words_max": max(words[cat]) if words[cat] else None,
            }
            for cat in sorted(per_cat)
        },
        "vcs_notes_naming_an_authority_or_role": {k: v for k, v in vcs_authority.items()},
        "drift_rate_all_notes": {k: {"rate": rate(a, n), "n": n} for k, (a, n) in drift_all.items()},
        "drift_rate_compliant_notes_only": {k: {"rate": rate(a, n), "n": n} for k, (a, n) in drift_ok.items()},
        "note": "regex rules are heuristics; failures flag notes for the human review sheet, not proof of rule breaks",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(f"{audit_dir}/note_failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")

    # human review sheet: 10 random notes per category (text stays off git)
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for t in trials:
        by_cat[t["category"]].append(t)
    rows = []
    for cat in sorted(by_cat):
        pick = rng.sample(by_cat[cat], min(10, len(by_cat[cat])))
        for t in pick:
            rows.append({"item_id": f"{t['case_id']}|{cat}", "category": cat, "source": t["source"],
                         "note": t["adversarial_note"],
                         "matches_category": "", "adds_new_clinical_fact_beyond_rule": "",
                         "chart_plausible": "", "harmful_implication": "", "comments": ""})
    rng.shuffle(rows)
    with open(f"{audit_dir}/note_review_sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: out[k] for k in ("n_trials", "vcs_notes_naming_an_authority_or_role",
                                          "drift_rate_all_notes", "drift_rate_compliant_notes_only")}, indent=2))
    print(f"-> {args.out}; failures + review sheet in {audit_dir}/ (not for git)")


if __name__ == "__main__":
    main()
