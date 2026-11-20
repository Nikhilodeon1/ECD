# Author pass

Rewrite the passages below in your own words and keep the facts and numbers. Text you write yourself is the best protection against a reviewer reading the paper as machine drafted, and it is also where your judgment belongs. Do not add claims. If a sentence reads awkwardly, fix the sentence and leave the number.

## Passages to rewrite

1. **Abstract, first sentence.** The four-source framing (noise, added text, content, attribution) is the paper's idea. Say it the way you would explain it to a labmate.
2. **Introduction, paragraph 1.** The opening scene and why you care about follow-up notes in charts. Add the motivation you actually have.
3. **Introduction, the "We do not claim" sentence.** This is the scope of your claim. Decide how strongly you want to put it.
4. **Related Work, the ends of each paragraph** ("Our setting is...", "We share the question and differ in the controls..."). Positioning against prior work is yours to judge. Check each against the paper it names.
5. **Design, Arms and Note generation.** Check that the descriptions match what you built. In particular the rules per category (the list in `src/note_spec.py`) and the verifier step.
6. **Results 4.1, the half-precision remark.** The padding and batch composition explanation is a guess. Keep it only if you believe it, or delete the sentence.
7. **Results 4.3, the interpretation of Gemma 4 following the attending version less.** This is the surprising result. Say what you think it means and keep the hedge.
8. **Discussion, paragraph 3 (bare statement versus attributed opinion).** This is a hypothesis, labelled as untested. Keep, change or cut.
9. **Discussion, the sentence on other authority manipulations.** List the ones you would try.
10. **Limitations.** Add any limit I missed, in your words. Confirm each statement is true.
11. **Conclusion, the last sentence (open question).** Pick the follow-up you want to be known for.

## Facts only you can confirm (nothing in the paper depends on a guess, but these claims need your yes)

- **Protocol before the runs.** The paper says the protocol was written before the runs and that all arms except the hedged ones were specified in advance. `docs/prereg-v3.md` was a draft when written and its first deviation entry is dated 2026-11-12. Confirm that statement matches your git history, and decide whether to release the document in the supplement (it is in a gitignored folder).
- **No power calculation.** The paper says none was run before the runs. The protocol promised a minimum detectable difference; check that you did not compute one.
- **Zero data retention.** The paper says MIMIC text went only to an API configured for zero data retention. You said that was verified; confirm the wording is accurate.
- **IRB or exemption statement.** TMLR asks nothing specific, but a clinical paper should say it. I did not write one because I do not know your status. Add a sentence on credentialed access and whether review was needed.
- **MedCaseReasoning license.** The Hugging Face card showed an MIT tag when the citation check ran, while your earlier note said CC BY 4.0. The paper does not state a license. Look at the card yourself before you release derived notes.
- **AI use.** TMLR allows LLMs as assistive tools and holds authors responsible, and requires no disclosure on the pages I read. If you want a statement anyway, write exactly what happened: Claude Code drafted the analysis code and this draft from your ledger, and you checked each number and reference. Do not claim a check you have not done.
- **The Gemma 4 and Llama numbers.** Done: `make_ledger.py` now reads the two summary JSONs in `results/` (copied from Drive on 2026-11-19; the Colab runs wrote them on 2026-11-13). All values I had transcribed from the pasted output matched them, and the Llama intervals are now exact. The note-pass-rate table, the verifier rate (97.6% of 2124) and the adoption counts still come from output you pasted, because the notes, answers and judged files were not supplied. Copy those from Drive (`data/v3/`) if you want them re-read too.

## Before you submit

- Swap the preamble of `main.tex` for the TMLR style file. The author guide says submissions must use the TMLR LaTeX stylefile and template (https://github.com/JmlrOrg/tmlr-style-file). The file is not in this repository.
- Anonymity: the paper contains no author names, links or institutions. Check the supplement too.
- TMLR has no page limit but wants length justified. The main text is 9 compiled pages with the appendix.
- TMLR asks for resubmissions after a rejection to be new submissions with a description of changes. The earlier version was rejected elsewhere, so check whether any text overlaps with a posted preprint of it.
- Reopen every URL in `CITATIONS_VERIFIED.md` once yourself. The second-pass check the skill asks for was not done.
- The exploratory result to look at again: in Gemma 4, the hedged attending and nursing contrasts have Holm p of 0.0574 and 0.0532, just above 0.05.

## What I left out on purpose

- Evidential Consistency Decoding (the earlier method). It was a separate claim with an unexplained gap at one setting, and including it would need its own evidence. It can be a short separate note.
- The earlier drift numbers on MedQA and MIMIC with the first note bank. Those notes broke their own construction rules in an audit, and the MIMIC neutral rate there (18.3%) differs from the one here (9.1%) under a different sample and judge. They are not comparable, so the paper does not mix them.
