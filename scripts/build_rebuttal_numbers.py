"""Assemble docs/rebuttal-numbers.md from results/*.json (the single source of truth
for the writing agent). Every row is read from a results file written by a script;
nothing is typed in by hand. Rows whose file does not exist yet are listed as
INCOMPLETE rather than dropped.

    python scripts/build_rebuttal_numbers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
rows = []


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def ci(c):
    return f"[{fmt(c[0])}, {fmt(c[1])}]" if c else "-"


def add(i, qty, value, interval, n, file, script, section, reviewers):
    rows.append((i, qty, fmt(value), ci(interval) if isinstance(interval, (list, tuple)) else (interval or "-"), fmt(n), file, script, section, reviewers))


def missing(i, qty, file, script, section, reviewers):
    rows.append((i, qty, "INCOMPLETE", f"{file} not produced", "-", file, script, section, reviewers))


# ---- baseline accuracy
b = load("baseline_stats.json")
for s in (b or {}).get("by_source", {}):
    v = b["by_source"][s]
    add(f"B1-{s}", f"baseline accuracy, {s}", v["accuracy"], v["wilson_95"], v["n"], "baseline_stats.json", "src/baseline_stats.py", "Tab. 2", "R4-6, R2-5")
if b and "fisher_gap" in b:
    add("B2", "accuracy gap MedQA vs MIMIC, Fisher p / odds ratio", b["fisher_gap"]["p_value"], None, b["fisher_gap"]["odds_ratio"], "baseline_stats.json", "src/baseline_stats.py", "Sec. 6.1", "G-4")
    L = b["length_adjusted_logit"]
    add("B3", "MIMIC vs MedQA accuracy odds ratio, adjusted for log evidence length", L["odds_ratio_mimic_vs_medqa"], L["ci_95"], b["n"], "baseline_stats.json", "src/baseline_stats.py", "Sec. 6.1 (weakens: gap shrinks)", "G-4")
if not b:
    missing("B1", "baseline accuracy by source", "baseline_stats.json", "src/baseline_stats.py", "Tab. 2", "R4-6")

# ---- drift
d = load("stats_v2.json")
if d:
    add("D1", "overall drift rate, all trials", d["drift_rate_overall"]["rate"], d["drift_rate_overall"]["ci_95"], d["drift_rate_overall"]["n_trials"], "stats_v2.json", "src/stats_v2.py", "Abstract, Sec. 6.1", "R1-1")
    for s, v in d["drift_rate_by_source"].items():
        add(f"D2-{s}", f"drift rate, {s} (cluster bootstrap)", v["rate"], v["ci_95"], v["n_trials"], "stats_v2.json", "src/stats_v2.py", "Tab. 2", "G-4")
    if "risk_difference_mimic_minus_medqa" in d:
        r = d["risk_difference_mimic_minus_medqa"]
        add("D3", "drift risk difference MIMIC - MedQA", r["diff"], r["ci_95"], None, "stats_v2.json", "src/stats_v2.py", "Sec. 6.1", "G-4")
    for c, v in d["drift_rate_by_category"].items():
        add(f"D4-{c}", f"drift rate, category {c}", v["rate"], v["ci_95"], v["n_trials"], "stats_v2.json", "src/stats_v2.py", "Sec. 6.2", "R3-1")
    g = d["gee"]
    for k, v in g["coefficients"].items():
        if k.startswith("C(source") or k == "log_words":
            add("D5", f"GEE odds ratio: {k}", v["odds_ratio"], v["ci_95"], g["n_cases"], "stats_v2.json", "src/stats_v2.py", "Sec. 6.1", "G-4")
    add("D6", "GEE joint Wald test for category (p)", g["category_joint_wald"]["p"], None, g["n_cases"], "stats_v2.json", "src/stats_v2.py", "Sec. 6.2", "R2-1")
    sg = d["source_gap_survives_length_adjustment"]
    add("D7", "does the source gap survive length adjustment?", sg.get("survives"), sg.get("ci_95"), None, "stats_v2.json", "src/stats_v2.py", "Sec. 6.1, 7", "G-4")
else:
    missing("D1", "drift rates, GEE", "stats_v2.json", "src/stats_v2.py", "Sec. 6.1-6.2", "G-4, R3-1")

# ---- controls
c = load("controls_summary.json")
if c:
    for s, v in c.get("noise_floor", {}).items():
        add(f"C1-{s}", f"noise floor: same prompt re-asked, answer changes ({s})", v["resample_drift_rate"], None, v["n"], "controls_summary.json", "src/controls.py noise", "Sec. 6.1 (new)", "G-1")
    for s, v in c.get("neutral_drift", {}).items():
        add(f"C2-{s}", f"neutral addendum drift ({s})", v["rate"], None, v["n"], "controls_summary.json", "src/controls.py neutral", "Sec. 6.1 (new)", "R3-1")
    for k, v in c.get("excess_drift_adversarial_minus_neutral", {}).items():
        add(f"C3-{k}", f"excess drift over neutral, {k}", v["excess"], v["ci_95"], v["n_cases"], "controls_summary.json", "src/controls.py summary", "Sec. 6.2", "R3-1")
    h = c.get("helpful_note_control")
    if h:
        for s, v in h["correction_rate_on_baseline_wrong"].items():
            add(f"C4-{s}", f"helpful note: correction rate on baseline-wrong ({s})", v["rate"], None, v["n"], "controls_summary.json", "src/controls.py helpful", "Sec. 7", "R2-3")
        for s, v in h["stays_correct_on_baseline_correct"].items():
            add(f"C5-{s}", f"helpful note: stays correct on baseline-correct ({s})", v["rate"], None, v["n"], "controls_summary.json", "src/controls.py helpful", "Sec. 7", "R2-3")
    if "WARNING" in c:
        add("C6", "WARNING from controls summary", c["WARNING"], None, None, "controls_summary.json", "src/controls.py summary", "Sec. 6.2", "R3-1")
else:
    missing("C1", "noise / neutral / helpful controls", "controls_summary.json", "src/controls.py", "Sec. 6.1-7", "G-1, R3-1, R2-3")

# ---- note audit
n = load("note_audit.json")
if n:
    for cat, v in n["compliance_by_category"].items():
        add(f"N1-{cat}", f"note fully rule-compliant, {cat}", v["fully_compliant"], None, v["n"], "note_audit.json", "src/note_audit.py", "Sec. 3, 8", "R3-3, R1-3")
    add("N2", "VCS notes naming an authority/role", n["vcs_notes_naming_an_authority_or_role"].get("all"), None, None, "note_audit.json", "src/note_audit.py", "Sec. 3 (contradicts 'unattributed')", "G-2")
    for s, v in n["drift_rate_compliant_notes_only"].items():
        add(f"N3-{s}", f"drift rate on rule-compliant notes only ({s})", v["rate"], None, v["n"], "note_audit.json", "src/note_audit.py", "Sec. 6.1", "R3-3")
else:
    missing("N1", "note rule-compliance", "note_audit.json", "src/note_audit.py", "Sec. 3, 8", "R3-3, R1-3, G-2")

# ---- MIMIC trivial baselines
m = load("mimic_baselines.json")
if m:
    for k, v in m["strict_string_match"].items():
        add(f"M1-{k}", f"MIMIC accuracy of trivial predictor: {k} (strict match)", v.get("accuracy"), v.get("wilson_95"), m["n_mimic_sample"], "mimic_baselines.json", "src/baselines_mimic.py", "Sec. 6.1", "R4-6")
else:
    missing("M1", "MIMIC trivial baselines", "mimic_baselines.json", "src/baselines_mimic.py", "Sec. 6.1", "R4-6")

# ---- power check
p = load("power_check.json")
if p:
    for k, v in p["scenarios"].items():
        add("P1", f"McNemar n for 80% power, {k}", v["connor_n"], None, f"exact-search n = {v['smallest_n_for_80pct_exact_power']}; exact power at Connor n = {v['exact_power_at_connor_n']}", "power_check.json", "src/power_check.py", "App. D", "G-5")

# ---- judge validation
for name in sorted(RES.glob("judge_validation_*.json")):
    j = json.loads(name.read_text(encoding="utf-8"))
    for task, sc in j["agreement_with_claude_strict"].items():
        for s, v in sc.items():
            add(f"J1-{j['judge']}-{task}-{s}", f"agreement with Claude, {j['judge']}, {task}, {s}: Cohen's kappa", v["kappa"], v["ci_95"], v["n"], name.name, "src/judge_validation.py", "Sec. 8, App.", "R4-6, R1-5, R3-2")
    for label, key in (("under Claude", "headline_under_claude"), (f"under {j['judge']} (strict)", "headline_under_strict")):
        hl = j[key]
        if "accuracy_gap_fisher" in hl:
            add(f"J2-{j['judge']}", f"accuracy gap p, {label}", hl["accuracy_gap_fisher"]["p_value"], None, None, name.name, "src/judge_validation.py", "Sec. 6.1", "R4-6")
        if "drift_gap_mann_whitney" in hl:
            add(f"J3-{j['judge']}", f"drift gap p (Mann-Whitney), {label}", hl["drift_gap_mann_whitney"]["p_value"], None, None, name.name, "src/judge_validation.py", "Sec. 6.1", "R4-6")
        if hl.get("category_cochrans_q"):
            add(f"J4-{j['judge']}", f"category Cochran's Q p, {label}", hl["category_cochrans_q"]["p_value"], None, hl["category_cochrans_q"]["n_cases"], name.name, "src/judge_validation.py", "Sec. 6.2", "R4-6")

# ---- ECD
g1 = load("g1_gate.json")
if g1:
    add("E0", "G1 gate: match rate vs original-only greedy by beta", json.dumps(g1["match_rate_vs_original_only_greedy"]), None, g1["n_trials"], "g1_gate.json", "src/g1_gate.py", "Sec. 5", "R2-6, R4-1")
for name in sorted(RES.glob("ecd_v2_*.json")):
    e = json.loads(name.read_text(encoding="utf-8"))
    tag = name.stem.replace("ecd_v2_", "")
    add(f"E1-{tag}", f"{e['model']} baseline accuracy (competence)", json.dumps({k: v["accuracy"] for k, v in e["baseline_accuracy"].items()}), None, e["set_sizes"], name.name, "src/ecd_v2.py", "Sec. 6.4", "R1-2, R4-3")
    for b_, v in e["by_beta"].items():
        if v["R"]:
            add(f"E2-{tag}-{b_}", f"recovery R(beta={b_})", v["R"]["rate"], v["R"]["cluster_boot_95"], v["R"]["n"], name.name, "src/ecd_v2.py", "Sec. 6.4", "R2-6, R4-1/2")
        if v["S"]:
            add(f"E3-{tag}-{b_}", f"suppression S(beta={b_})", v["S"]["rate"], v["S"]["cluster_boot_95"], v["S"]["n"], name.name, "src/ecd_v2.py", "Sec. 6.4", "R2-3")
    P = e.get("prompt_baseline_P")
    if P and P["R_P"] and P["S_P"]:
        add(f"E4-{tag}", "prompt baseline P: R_P", P["R_P"]["rate"], P["R_P"]["cluster_boot_95"], P["R_P"]["n"], name.name, "src/ecd_v2.py", "Sec. 6.4", "R1-4")
        add(f"E5-{tag}", "prompt baseline P: S_P", P["S_P"]["rate"], P["S_P"]["cluster_boot_95"], P["S_P"]["n"], name.name, "src/ecd_v2.py", "Sec. 6.4", "R1-4")
    add(f"E6-{tag}", "decision rule: ECD useful?", e["decision_rule"].get("ECD_useful"), None, None, name.name, "src/ecd_v2.py", "Sec. 6.4, 9", "R4-1/2/3")
if not list(RES.glob("ecd_v2_*.json")):
    missing("E1", "corrected ECD R/S table", "ecd_v2_<model>.json", "src/ecd_v2.py", "Sec. 6.4", "R2-6, R4-1/2/3, R1-2, R1-4")

head = "| ID | quantity | value | 95% CI | n | results file | script | paper section | reviewers |\n|---|---|---|---|---|---|---|---|---|\n"
body = "\n".join("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in rows)
out = ROOT / "docs" / "rebuttal-numbers.md"
out.parent.mkdir(exist_ok=True)
out.write_text("# Rebuttal numbers (generated; do not edit by hand)\n\nBuilt by `scripts/build_rebuttal_numbers.py` from `results/*.json`. "
               "Re-run after any result changes. INCOMPLETE rows mean the result file does not exist yet.\n\n" + head + body + "\n", encoding="utf-8")
print(f"wrote {out} ({len(rows)} rows, {sum(1 for r in rows if r[2] == 'INCOMPLETE')} incomplete)")
