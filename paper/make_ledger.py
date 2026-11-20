"""Builds EVIDENCE.md from the transcribed run summaries. Every value below is copied from a
summary JSON written by src/v3_summary.py (public runs: Colab, files on Drive; MIMIC run: results/).
Derived quantities are computed here, never typed. Rounding follows the paper (one decimal in percent)."""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1] / "results"
mimic = json.load(open(R / "v3_summary_judged__claude-claude-sonnet-5__claude-claude-sonnet-5.json"))["by_source"]["mimic-iv-note"]

# Transcribed from the Colab run output of 2026-11-13
# (Drive MyDrive/ecd/results/v3_summary_judged__<model>__open-microsoft-phi-4.json)
# arms: name -> (n, drift, (excess, ci_lo, ci_hi) or None); contrasts: (diff, lo, hi, up, down, p_exact, p_holm)
public = {
    "gemma4": dict(n_cases=400, base_ok=114, acc=0.285, arms={
        "R": (114, .026, None), "N": (114, .228, None),
        "VCS": (87, .368, (.138, .034, .241)), "ISO": (108, .370, (.148, .065, .241)),
        "PLV": (93, .226, (.011, -.054, .086)), "MMS": (89, .202, (-.011, -.101, .079)),
        "SNHR": (109, .303, (.092, .009, .174)),
        "A": (108, .694, (.491, .389, .593)), "B": (108, .611, (.407, .306, .509)), "C": (108, .491, (.287, .185, .389)),
        "Ah": (108, .454, None), "Bh": (108, .380, None), "Ch": (108, .352, None)},
        r_excess=(-.202, -.281, -.123),
        contr={"B-A": (-.083, -.139, -.037, 0, 9, .0039, .0039), "C-A": (-.204, -.287, -.130, 1, 23, 0.0, 0.0),
               "Bh-Ah": (-.074, -.139, -.009, 3, 11, .0574, .0574), "Ch-Ah": (-.102, -.185, -.019, 5, 16, .0266, .0532)},
        helpful=dict(wrong=(213, .197), right=(94, .798)), adopt=(66, 61, 47, 108)),
    "llama31": dict(n_cases=400, base_ok=83, acc=0.207, arms={
        "R": (83, .048, None), "N": (83, .265, None),
        "VCS": (65, .277, (.077, -.01, .17)), "ISO": (78, .590, (.346, .20, .47)),
        "PLV": (66, .273, (.045, -.04, .14)), "MMS": (60, .317, (.033, -.07, .13)),
        "SNHR": (79, .304, (.051, -.04, .14)),
        "A": (77, .831, (.571, .44, .69)), "B": (77, .818, (.558, .43, .68)), "C": (77, .805, (.545, .42, .66)),
        "Ah": (77, .649, None), "Bh": (77, .662, None), "Ch": (77, .610, None)},
        r_excess=(-.217, -.33, -.12),
        contr={"B-A": (-.013, -.052, .026, 1, 2, None, 1.0), "C-A": (-.026, -.078, .026, 1, 3, None, 1.0),
               "Bh-Ah": (.013, -.026, .065, 2, 1, 1.0, 1.0), "Ch-Ah": (-.039, -.091, .0, 0, 3, .25, .5)},
        helpful=dict(wrong=(242, .194), right=(65, .785)), adopt=(66, 62, 61, 77)),
}
notes_pass = {"X": (400, .963, 1.08), "ISO": (385, .943, 1.16), "SNHR": (385, .987, 1.09), "PLV": (385, .896, 1.25),
              "VCS": (385, .849, 1.54), "MMS": (385, .842, 1.66), "H": (400, .767, 1.49)}


def pct(x):
    return f"{100 * x:.1f}"


out = ["# Evidence ledger", "",
       "Every number in paper/main.tex comes from a row here. Source for public rows: "
       "`v3_summary_judged__<model>__open-microsoft-phi-4.json` from the Colab run of 2026-11-13, transcribed from the "
       "printed output (copy the Drive files into results/ and extend make_ledger.py to read them to re-check). "
       "Source for MIMIC rows: `results/v3_summary_judged__claude-claude-sonnet-5__claude-claude-sonnet-5.json`. "
       "Derived values are computed in paper/make_ledger.py.", "",
       "Prespecified in docs/prereg-v3.md: R, N, the five categories, framing arms A, B, C, contrasts B-A and C-A, "
       "the helpful note. Exploratory: hedged arms (Ah, Bh, Ch) and the adoption string match.", ""]
for key, name in (("gemma4", "Gemma 4 12B"), ("llama31", "Llama 3.1 8B")):
    d = public[key]
    dg = 0 if key == "llama31" else 1  # the Llama summary was printed with two-decimal intervals, so report whole points
    out += [f"## {name}, public cases (n_cases {d['n_cases']}, baseline-correct {d['base_ok']}, accuracy {pct(d['acc'])}%)", ""]
    out += ["| id | arm | n | drift (%) | excess over neutral, points [95% CI] | status |", "|---|---|---|---|---|---|"]
    for i, (a, (n, dr, ex)) in enumerate(d["arms"].items()):
        e = f"{100 * ex[0]:+.1f} [{100 * ex[1]:+.{dg}f}, {100 * ex[2]:+.{dg}f}]" if ex else ""
        st = "exploratory" if a in ("Ah", "Bh", "Ch") else "prespecified"
        out.append(f"| {key}-{i} | {a} | {n} | {pct(dr)} | {e} | {st} |")
    nr = d["arms"]["N"][1] - d["arms"]["R"][1]
    lo, hi = -d["r_excess"][2], -d["r_excess"][1]
    out += ["", f"Derived: N minus R = {100 * nr:.1f} points; CI from the paired R-minus-N bootstrap with the sign flipped: "
            f"[{100 * lo:.{dg}f}, {100 * hi:.{dg}f}] (R excess {100 * d['r_excess'][0]:+.1f} "
            f"[{100 * d['r_excess'][1]:+.{dg}f}, {100 * d['r_excess'][2]:+.{dg}f}]).", ""]
    out += ["| contrast | diff (points) | 95% CI | discordant up/down | exact p | Holm p |", "|---|---|---|---|---|---|"]
    for c, (df, lo_, hi_, up, dn, p, ph) in d["contr"].items():
        out.append(f"| {c} | {100 * df:+.1f} | [{100 * lo_:+.1f}, {100 * hi_:+.1f}] | {up}/{dn} | "
                   f"{p if p is not None else 'n/a'} | {ph} |")
    h = d["helpful"]
    out += ["", f"Helpful note: corrects a wrong baseline in {pct(h['wrong'][1])}% of {h['wrong'][0]} cases; stays correct in "
            f"{pct(h['right'][1])}% of {h['right'][0]} baseline-correct cases (so {100 - 100 * h['right'][1]:.1f}% do not).",
            f"Adoption (exploratory string match; the answer contains the inserted wrong diagnosis): unattributed "
            f"{d['adopt'][0]} of {d['adopt'][3]}, attending {d['adopt'][1]} of {d['adopt'][3]}, nursing "
            f"{d['adopt'][2]} of {d['adopt'][3]}.", ""]
out += ["## Claude Sonnet 5 on MIMIC-IV-Note (150 cases)", "",
        f"Baseline-correct {mimic['baseline_correct']} of {mimic['n_cases']} ({pct(mimic['baseline_accuracy'])}%).", "",
        "| arm | n | drift (%) | excess over neutral, points [95% CI] |", "|---|---|---|---|"]
for a, x in mimic["arms"].items():
    if not x["n"]:
        continue
    e = (f"{100 * x['excess_over_neutral']:+.1f} [{100 * x['excess_ci'][0]:+.1f}, {100 * x['excess_ci'][1]:+.1f}]"
         if "excess_over_neutral" in x else "")
    out.append(f"| {a} | {x['n']} | {pct(x['drift'])} | {e} |")
r, n = mimic["arms"]["R"], mimic["arms"]["N"]
out += ["", f"Derived: N minus R = {100 * (n['drift'] - r['drift']):.1f} points, CI [{-100 * r['excess_ci'][1]:.1f}, "
        f"{-100 * r['excess_ci'][0]:.1f}] (sign-flipped R excess).", ""]
out += ["| contrast | diff (points) | 95% CI | discordant up/down | exact p | Holm p |", "|---|---|---|---|---|---|"]
for c in mimic["framing_contrasts"]:
    out.append(f"| {c['contrast']} | {100 * c['diff']:+.1f} | [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] | "
               f"{c['discordant_up']}/{c['discordant_down']} | {c['p_exact']} | {c['p_holm']} |")
out += ["", "## Note generation (public cases, Qwen2.5-7B-Instruct, up to 3 attempts)", "",
        "| kind | notes | passed rules (%) | mean attempts |", "|---|---|---|---|"]
for k, (n_, p, a) in notes_pass.items():
    out.append(f"| {k} | {n_} | {pct(p)} | {a} |")
out += ["", "Verifier (phi-4) answered yes to 97.6% of 2124 notes. In a calibration check it answered yes to 96 of 96 notes "
        "judged against their own category and to 11 of 96 judged against a wrong category.", "",
        "## Setup facts", "",
        "- Public cases: MedCaseReasoning test split (897 rows on the dataset viewer), 400 sampled, at most 3 per diagnosis, "
        "seed 42. Presentation 60 to 700 words, diagnosis at most 8 words, diagnosis string not in the presentation.",
        "- MIMIC cases: 150 MIMIC-IV-Note cases from the evaluation sample (median 333 words, 12 over 800 words and truncated).",
        "- Neutral arm: 16 fixed non-clinical lines, one chosen by a hash of the case id. Re-ask arm: the baseline prompt sampled again.",
        "- Open models: greedy decoding, 160 new tokens. Claude: thinking disabled, 500 tokens for baseline and re-ask, 300 for note arms.",
        "- Bootstrap: 10000 case resamples. Exact McNemar test on discordant pairs. Holm correction over two contrasts per family.",
        "- MIMIC run: 1430 Claude API calls (745 Haiku 5.5, 685 Sonnet 5), estimated cost 6.01 dollars before the final judging "
        "pass and 6.51 dollars after it, priced at Sonnet rates.",
        "- Models: Gemma 4 12B instruction-tuned, Llama 3.1 8B Instruct, Qwen2.5 7B Instruct (generator), phi-4 14B (verifier and "
        "public judge), Claude Sonnet 5 (tested model and MIMIC judge), Claude Haiku 5.5 (MIMIC notes and verifier).",
        "- Token limits and sizes: 160 new tokens for open models, 500 and 300 for Claude, evidence cut to 800 words, "
        "public presentations of 60 to 700 words, diagnoses of at most 8 words, notes of at most 40 words, 16 neutral lines, 3 attempts.",
        "- Holm p printed as 0.0 in the summary file is reported in the paper as below 0.001.",
        "- Judge reading check: of the first 25 baseline answers that phi-4 judged incorrect for Gemma 4, the author's non-clinician "
        "reading found about 3 to 4 that look like a parent or subtype of the reference diagnosis.",
        "- Count of verifier notes: 2124.",
        "- Judge validation for the v3 judges: none. Clinician audit: none."]
pathlib.Path(__file__).with_name("EVIDENCE.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("ok")
