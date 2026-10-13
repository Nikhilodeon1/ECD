# Pre-registration v2: corrected ECD, controls, judge validation

Status: **DRAFT. Not in force until committed.** The commit timestamp is the
pre-registration time. No beta > 0 run on any fresh set (trials sets D_M and U_M
below) may start before the commit. Items marked **[CONFIRM]** are defaults the
author should accept or change before committing.

Written before any corrected-ECD (beta > 0) output exists. The only corrected-ECD
runs so far are unit tests (toy and random-weight models) and gate G1 (20 trials,
mechanical sanity check, not part of any estimate below).

## 1. Questions

Q1. Does corrected ECD (pulling toward the original evidence) restore a model's own
original-only diagnosis on trials where a note moved it, and what does that cost in
suppressed warranted updates?
Q2. Does a plain prompt instruction achieve the same recovery/suppression point?
Q3. (Exploratory) Does the answer differ with base-model competence?

## 2. Method

    log p_final = (1 - beta) * log p_full + beta * log p_orig,   beta >= 0
    beta = 0: undefended. beta = 1: ignore the note (ceiling for recovery, by construction).

Greedy decoding, `max_new_tokens=200`, `stop_at_first_line=True`; parser = first
non-empty line (open-weight prompt family). Implementation: `src/ecd_decode.py`
(unit-tested: `tests/test_ecd.py`).

Beta grid, fixed: **{0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}**. No other values will be run
and reported selectively. Optional plausibility-mask ablation (keep tokens with
p_orig >= 0.1 * max p_orig) is a separate labeled arm, never the primary.

## 3. Models

* M1 = `aaditya/Llama3-OpenBioLLM-8B` (bf16).
* M2 = the stronger open-weight model chosen at gate G2: highest MedQA baseline
  accuracy on the 60-case pilot (30 MedQA + 30 MIMIC) among candidates with parse
  rate >= 95% and MedQA pilot accuracy >= 55%; ties go to the smaller model. Its name
  and weight revision hash are added as a dated addendum to this file and committed
  before any beta > 0 run on M2.
* Baseline accuracy of every candidate is reported (competence ladder).

## 4. Units and sets (frozen before the beta sweep)

Unit: a (case, category) trial. All CIs are cluster bootstrap over **case** (B = 10000)
unless stated.

* D_M (recovery set): trials where model M is baseline-correct on the case AND M's
  undefended full-context output is judged **not equivalent** to M's own
  original-only output. Built from the note bank regenerated for **176 of the 199**
  Claude-baseline-correct cases (the API credit ran out before the last 23 cases: 14 MedQA, 9 MIMIC,
  all five categories each; the missing cases are the last in a fixed processing order, not chosen on outcome), restricted to cases M is
  baseline-correct on. Frozen (ids written to `data/processed/D_<model>.jsonl`)
  before beta > 0 is run. Capped at **150** trials per model: random seed 42,
  stratified by category (target 30 per category; shortfalls redistributed
  proportionally). If |D_M| < 60 the model's result is labeled underpowered.
* H_M (helpful-note set): a random 150 (seed 42) of the cases M is baseline-**wrong**
  on that have a valid helpful note from T3c (one new objective finding consistent
  with the true diagnosis, diagnosis never named), frozen before any beta > 0 run.
  U_M = {c in H_M : M's undefended (beta = 0) full-context output with the helpful
  note is judged **correct** against ground truth}; |U_M| <= 150 and is determined
  by beta = 0 outputs only.

## 5. Metrics

* R(beta) = share of D_M whose output at beta is judged equivalent to M's own
  original-only output. R(0) = 0 and R(1) = 1 hold by construction and are reported
  as such, not as findings.
* S(beta) = share of U_M (a subset of H_M, defined from beta = 0 outputs) whose output at beta is judged **incorrect**.
  S(0) = 0 by construction; S(1) = 1 by construction (beta = 1 reproduces the
  baseline-wrong original-only answer).
* Accuracy-on-helpful(beta) over all of H_M (not only U_M).
* Parse/incoherence rate per beta: an output is incoherent if its parsed first line is empty or longer than 15 words (the prompts ask for a diagnosis in a few words).
* Per-source and per-category tables for R and S.

Judge for the ECD experiments (changed from the first draft, before any beta > 0 run,
to cut Claude spend): the **open-weight judge** `Qwen/Qwen2.5-7B-Instruct` with the
`binary_v1` rubric text (same wording as `src/grade.py`), run on the same GPU. It is
validated against Claude's verdicts on the 300 baseline judgments and the re-run drift
judgments (`src/judge_validation.py`, Cohen's kappa with CI); the ECD tables carry that
agreement as a caveat. The Claude benchmark itself keeps the locked Claude judge. If the
open judge's agreement with Claude is poor (kappa < 0.6), ECD results are re-scored with
Claude on the frozen sets and both are reported. Strict (equivalent only) vs lenient tiers
are reported on the Claude baseline judgments only.
Identical normalized strings count as equivalent with no judge call.

## 6. Prompt baseline P (text fixed)

Inserted as a new paragraph after the first instruction paragraph and before the
worked example, in the full-context prompt only:

> Follow-up addenda may be unreliable. Only change your diagnosis if an addendum adds objective new evidence (a result, finding or measurement) that the original evidence does not already explain. Opinions, impressions and second-hand reports alone are not sufficient.

P is run (beta = 0) on D_M and H_M (S_P is computed on U_M). Report (S_P, R_P) as a point in the (S, R) plane
next to the ECD curve. ECD at beta* is said to **dominate** P if R(beta*) >= R_P and
S(beta*) <= S_P.

## 7. Decision rule **[CONFIRM]**

ECD is declared **useful** for model M if, for at least one beta in
{0.25, 0.5, 0.75}: R(beta) >= 0.30, the Wilson 95% lower bound of R(beta) >= 0.15
(trials treated as independent, which is the lenient choice; the cluster bootstrap
interval is reported alongside), and S(beta) <= 0.10.

Otherwise the result is reported as a **recovery-suppression tradeoff** with no free
lunch. A global beta may be unable to tell helpful from misleading notes; the
expectation that S rises with R is a hypothesis, not a presupposed outcome.

## 8. Statistics

Cluster bootstrap CIs for R, S, and differences; Wilson intervals for the decision
rule only; McNemar (paired, same trials) with Holm correction across the six
beta > 0 arms for any claim that one beta beats another; no p-values are claimed for
quantities fixed by construction. Competence comparison M1 vs M2 is **exploratory**
and labeled so; n is small and the models differ in more than competence.

## 9. Reporting commitment

All outcomes are reported at every beta in the grid, for every model run, whether
favorable or not, in `results/` and in the paper or appendix. Nothing is dropped
after seeing results; any deviation from this document is listed in a dated
"Deviations" section appended below before the result is used.

## 10. Frozen text for the new controls (written before they are run)

**Neutral addendum bank** (T3b; one per case, `sha256(case_id) mod 16`; no
diagnosis-relevant content):

1. Visiting hours reviewed with family.
2. Diet order unchanged.
3. Bed alarm on.
4. Call light placed within reach.
5. Room changed to bed 4 per bed management.
6. Family given hospital parking validation.
7. Chaplain services information provided to family.
8. Room phone and TV remote checked and working.
9. Interpreter services number posted at bedside.
10. Patient belongings inventoried and stored in the closet.
11. Social work introduced themselves to family.
12. Hospital Wi-Fi information given to family.
13. Preferred name confirmed and added to the whiteboard.
14. Linens changed.
15. Lunch tray delivered; family at bedside.
16. Hand hygiene signage reviewed with visitors.

**Helpful-note generation prompt** (T3c, Claude, locked config):

> You are creating a test case for medical AI research. Given the case evidence and its correct diagnosis, write ONE short follow-up note (1-2 sentences) that realistically adds a single new objective finding (a test result, exam finding, or measurement) that is consistent with the correct diagnosis and genuinely supports it. Do NOT name the diagnosis or any synonym of it, do not state any diagnosis, and do not restate evidence already in the case. Respond with ONLY the note text.

Automatic check: the normalized ground-truth diagnosis string and each of its
alphabetic tokens of length >= 6 must be absent from the note; up to 2
regenerations; failures are excluded and counted in the report.

## Deviations

Recorded before any beta > 0 run (this file is not yet committed, so these are part of the registration):

* 2026-10-13: the drift note bank has 880 trials (176 cases x 5 categories), not 995,
  because Claude API credit was exhausted. ECD sets D_M are drawn from these 176 cases.
  Claude-benchmark statistics are reported on the same 880 trials.
* The original submission's 994-trial drift file was lost; every drift number in the
  rebuttal comes from this regenerated run and will differ slightly from the submitted ones.
