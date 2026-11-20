#!/usr/bin/env python3
"""paperlint: style, citation and number audit for paper drafts (.tex or .md).

python3 -I paperlint.py paper.tex [more files] [--bib refs.bib]
        [--log CITATIONS_VERIFIED.md] [--ledger EVIDENCE.md]

Exit code 1 if any ERROR. ERRORs are cheap-to-fix surface items (dashes,
contractions, rare-in-human-papers words, unresolved markers, bad citations,
numbers missing from the ledger). Style warnings fire when a rate leaves the
range of 1,843 pre-2023 human-written papers (ICML, AISTATS, UAI, ML4H, MLHC,
MIDL, CHIL 2019-2022); limits sit near that corpus's 98th percentile. A flag means
"look at this", never "an AI wrote this". Run it on the whole paper, not a
section: density checks are skipped under 800 words.
"""
import os, re, sys, argparse, statistics as st

HARD = (r"\b(?:delv(?:e|es|ed|ing)|tapestry|meticulous(?:ly)?|groundbreaking|transformative|paramount|unparalleled"
        r"|invaluable|multifaceted|testament|embark(?:s|ed|ing)?|endeavou?rs?|bolster(?:s|ed|ing)?|plethora|burgeoning"
        r"|commendable|noteworthy|myriad|pivotal|underscor(?:e|es|ed|ing)|intricate(?:ly)?|intricacies|seamless(?:ly)?"
        r"|holistic(?:ally)?|realm|unlock(?:s|ed|ing)?|nuanced?|harness(?:es|ed|ing)?|cornerstone|cutting-edge"
        r"|game-chang\w+|ever-evolving|revolutioni[sz](?:e|es|ed|ing))\b"
        r"|pav(?:e|es|ed|ing) the way|at the forefront|in today's|rapidly evolving"
        r"|navigat(?:e|es|ing) the (?:complexit|landscape|challenge)"
        r"|(?:has|have) (?:attracted|garnered) (?:significant|growing|considerable|increasing) (?:attention|interest)")
WATCH = set("""capabilities enabling enhance enhances enhanced enhancing comprehensive remarkable remarkably
ensuring offers integrates integrating highlighting advancements addressing struggle struggles emerged primarily
achieving preserving maintaining demonstrating inherent thereby insights diverse extensive effectively exhibit
exhibits potential tailored foster fosters fostering showcase showcases showcasing vital crucial additionally
notably substantial employs facilitate facilitates facilitating""".split())
PART = (r",\s+(?:thereby |thus |while )?(?:enabling|allowing|ensuring|highlighting|demonstrating|providing|offering"
        r"|making|leading|resulting|reflecting|suggesting|indicating|underscoring|showing|achieving|improving"
        r"|reducing|enhancing|facilitating|capturing|revealing|paving|opening|yielding|supporting|promoting)\b")
TRANS = r"(?:^|(?<=[.!?] ))(?:Moreover|Furthermore|Additionally|In addition|Notably|Importantly|Interestingly|Crucially|Overall|Consequently|Hence|Thus)\b"
SUMMARY = r"^(?:Overall|In summary|In conclusion|Taken together|Together|Collectively|These (?:results|findings|observations) (?:highlight|underscore|demonstrate|suggest|show|indicate|reveal)|This (?:highlights|underscores|demonstrates|reveals|shows))\b"
CITE = r"\\(?:cite[a-z]*|autocite|parencite|textcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}"
ENVS = "equation|align|eqnarray|gather|multline|tikzpicture|lstlisting|verbatim|algorithm|algorithmic|thebibliography"
ABBR = re.compile(r"\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Tab|vs|cf|Dr|No|approx|resp|Alg|Appx)\.", re.I)
MARK = r"\[(?:CITE-NEEDED|VERIFY)[^\]]*\]|\bTODO\b|\bFIXME\b|\bXXX\b|\?\?"
RESID_ERR = (r"(?i:as (?:suggested|requested|recommended|noted|pointed out) by (?:the |a |an )?(?:reviewers?|referees?|editors?)"
             r"|\b(?:in response to|per) (?:the )?reviewers?\b"
             r"|\bimprov\w+ (?:on|over) (?:the |our )?(?:previous|earlier|prior) (?:run|runs|attempt|attempts|version|iteration)"
             r"|\b(?:earlier|previous|prior|updated|revised) version of (?:this|the present|our) (?:paper|work|manuscript)"
             r"|\bwe (?:now|then) (?:fix|fixed|correct|corrected|patch|patched)\b|\bI hope this|\bGreat question|\bas an AI\b)"
             r"|\bTL;DR|\bKey takeaways?\b|\(new\)|\bNEW:|\bUPDATED?:"
             r"|(?i:\bsupersed(?:ed|es)\b|\bwithdr(?:aw|awn|ew) (?:the |this |that )?(?:claim|prediction|hypothesis|conclusion)|\bdesign error\b|\boriginal runs?\b"
             r"|\b(?:earlier|first|previous) \d+-seed\b|\bno (?:decision )?rule (?:was )?written\b|\bdated (?:repository )?file\b)")
RESID_WARN = (r"(?i:\b(?:in|of|for) (?:this|the current|the present|the final|the latest|the new) (?:version|iteration|draft|revision|round)\b"
              r"|\b(?:next|previous|earlier|latest|updated|revised) (?:iteration|iterations|draft|drafts|revision)\b"
              r"|\bin the (?:next|previous|last|earlier) (?:run|round|version)\b"
              r"|\bwe (?:previously|earlier|originally|initially) (?:\w+ ){0,2}(?:found|reported|showed|claimed|tried|believed|assumed)\b"
              r"|\b(?:bug|bugs|buggy)\b|\b(?:fixed|corrected|patched) (?:a |the )?(?:bug|error|leak|mistake)\b"
              r"|\b(?:ChatGPT|Claude|Gemini|Copilot)\b)|\bHere (?:is|are)\b|\bCertainly\b")


def blank(pat, text, flags=0):
    """Replace matches with spaces, keeping newlines, so line numbers stay valid."""
    return re.sub(pat, lambda m: re.sub(r"[^\n]", " ", m.group()), text, flags=flags)


def load(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    return t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def views(text, ext):
    """Return (raw without comments, view for numbers, view for prose)."""
    if ext == ".tex":
        text = blank(r"(?<!\\)%[^\n]*", text)
        m = re.search(r"\\begin\{document\}", text)
        if m:
            text = blank(r"\A.*?\\begin\{document\}", text, re.S)
        text = blank(r"\\begin\{thebibliography\}.*", text, re.S)
        nums = blank(r"\\(?:vspace|hspace|setlength|includegraphics|resizebox|scalebox|rule)\*?(?:\[[^\]]*\])*\{[^}]*\}", text)
        nums = blank(r"\\begin\{(?:%s)(\*?)\}.*?\\end\{(?:%s)\1\}" % (ENVS, ENVS), nums, re.S)
        nums = blank(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\)", nums, re.S)
        nums = blank(CITE, nums)
        nums = blank(r"\\(?:ref|eqref|autoref|cref|Cref|label|url|href|input|include|usepackage|bibliography|bibliographystyle)\*?(?:\[[^\]]*\])*(?:\{[^}]*\})?", nums)
        prose = blank(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", nums, re.S)
        prose = blank(r"\\[a-zA-Z]+\*?|[{}\\~&]", prose)
        nums = blank(r"\\[a-zA-Z]+\*?|[{}\\~]", nums)
    else:
        text = blank(r"`{3}.*?`{3}|`[^`\n]*`", text, re.S)
        text = blank(r"^[ \t]*\|?[ \t]*:?-{3,}[-:| \t]*$", text, re.M)
        prose = blank(r"\]\([^)]*\)|^#+|\*\*|\*|\|", text, re.M)
        nums = blank(r"\]\([^)]*\)", text)
    prose = blank(r"\(\s*\)|\[\s*\]", blank(CITE, prose))
    prose = blank(r"\[@[^\]]*\]|\[(?:CITE-NEEDED|VERIFY)[^\]]*\]", prose)
    return text, nums, prose


def sentences(par):
    p = ABBR.sub(lambda m: m.group().replace(".", "\x00"), par)
    p = re.sub(r"(?<=\d)\.(?=\d)", "\x00", p)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", p)
    return [s.replace("\x00", ".").strip() for s in parts if len(s.split()) >= 3]


CANDID = r"(?:Importantly|Notably|Crucially|Interestingly|Remarkably|Surprisingly|Intriguingly|Strikingly),"
ANTI = (r"\bnot (?:just|merely|simply)\b|\bnot only\b[^.;]{0,120}?\bbut (?:also|rather)\b"
        r"|\b(?:is|are|was|were) not [^.;]{3,60}[;,] (?:it|they|this|these) (?:is|are)\b")
# (key, limit, unit, what it counts); limits sit near the 98th percentile of 1,843 human papers,
# widened slightly so clinical venues (more lists and summary sentences) are not over-flagged
GATES = [("watch", 0.55, "% of words", "Tier B words"),
         ("part", 2.5, "per 1k words", "participial tails (', enabling ...')"),
         ("trans", 10.0, "% of sentences", "sentences opening with a stock transition"),
         ("triad", 3.0, "per 1k words", "list-of-three patterns"),
         ("candid", 0.85, "per 1k words", "candid openers (Notably, Interestingly, ...)"),
         ("anti", 0.6, "per 1k words", "not-X-but-Y constructions"),
         ("summ", 1.0, "per 1k words", "summary sentences (Overall, These results highlight ...)")]
ACR_OK = set("""AI ML DL NLP CV CNN CNNs RNN RNNs LSTM GRU GAN GANs VAE VAEs SGD GPU GPUs CPU MLP MLPs KL AUC AUROC AUPRC ROC MSE MAE
RMSE SOTA IID BERT GPT LLM LLMs ICU EHR EHRs MIMIC ECG EEG MRI CT PET HIV COVID US UK USA ICML ICLR AAAI UAI AISTATS MIDL CHIL MLHC
TMLR PMLR IEEE ACM NIH FDA IRB PCA SVD SVM KNN LDA ODE ODEs PDE SDE MCMC MAP MLE ELBO EM ERM IRM DRO RL MDP POMDP DQN PPO TD UCB GP
RBF NTK API URL PDF HTML SQL JSON RAM GB TB MB LR RF XGB ReLU GELU FPR TPR PPV NPV ECE NLL FGSM PGD DNN DNNs OOD ID ICD SOFA APACHE
SAPS GCS BMI BP HR OK II III IV VI VII VIII IX XI XII IP DAG DAGs CI SD SE MCC QA VQA RGB SGLD HMC VI CCA ICA NMF DP FL IoU PSNR SSIM
LPIPS FID IS TV NN NNs GNN GNNs GCN GCNs PAC SAT SDP LP QP TPU FLOPs FLOP KG MNIST CIFAR SVHN ImageNet""".split())
ACR_DEF = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,9}s?)\)")
ACR_USE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})s?\b")
VAGUE = (r"(?:^|(?<=[.!?] ))(?:This|That|These|Those|It|Such)\s+(?:is|are|was|were|has|have|had|can|could|will|would|may|might|means"
         r"|shows|suggests|leads|makes|allows|implies|indicates|provides|helps|causes|explains|reflects|gives)\b")
PROC = (r"(?i:\b(?:fixed in advance|written in advance|written before the runs|before the runs|pre-?specified|pre-?registered|pre-?set"
        r"|decision rules?|(?:the|a|our) rule (?:was|were|required|counted|fixed)|rules? (?:required|written|fixed)|no rule|dated (?:repository )?file"
        r"|supersed\w+|withdr(?:aw|awn|ew)|design error|earlier \d+-seed|first \d+-seed|follow-up sweeps?|confirmatory run)\b)")
DISCL = (r"(?i:\b(?:we (?:have|did|do) not|we (?:cannot|could not|can not)|was not run|were not run"
         r"|not (?:been )?(?:tested|verified|measured|established|isolated|shown|run|done)|untested|unexplained|remains? (?:open|unexplained)"
         r"|cannot (?:separate|say|exclude|distinguish)|does not (?:show|establish|test|separate|prove)|is confounded|not independent"
         r"|inconclusive|not supported|not significant)\b)")
LOGGATES = [("process-log", PROC, 0.75, "process and pre-registration words (rule, fixed in advance, superseded, withdrawn)"),
            ("disclaimer-density", DISCL, 1.2, "disclaimers (we have not, not tested, confounded, unexplained)")]


def undefined_acronyms(prose):
    """Acronyms used twice or more with no '(ACR)' definition anywhere and not on the common list."""
    defs = {m.group(1).rstrip("s") for m in ACR_DEF.finditer(prose)}
    cnt = {}
    for m in ACR_USE.finditer(prose):
        a = m.group(1)
        if a not in ACR_OK and a not in defs: cnt[a] = cnt.get(a, 0) + 1
    return sorted(a for a, c in cnt.items() if c >= 2)


def stats(prose):
    pars = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", prose)]
    pars = [p for p in pars if len(p.split()) >= 6]
    sents = [[s for s in sentences(p)] for p in pars]
    allS = [s for ps in sents for s in ps]
    L = [len(s.split()) for s in allS]
    flat = " ".join(pars)
    low = re.findall(r"[A-Za-z][A-Za-z'-]*", flat.lower())
    n, w = len(L), max(len(low), 1)
    k = 1000.0 / w
    uniform = sum(1 for ps in sents if len(ps) >= 4 and st.pstdev([len(s.split()) for s in ps]) < 4.0)
    return dict(words=w, sentences=n, mean=st.mean(L) if L else 0, sd=st.pstdev(L) if L else 0,
                we=k * len(re.findall(r"\b(?:we|our|ours)\b", flat, re.I)),
                watch=100.0 * sum(x in WATCH for x in low) / w,
                part=k * len(re.findall(PART, flat)),
                trans=100.0 * sum(bool(re.match(TRANS, s)) for s in allS) / max(n, 1),
                triad=k * len(re.findall(r"\b\w+, \w+,? (?:and|or) \w+\b", flat)),
                candid=k * sum(bool(re.match(CANDID, s)) for s in allS),
                anti=k * len(re.findall(ANTI, flat, re.I)),
                summ=k * sum(bool(re.match(SUMMARY, s)) for s in allS),
                uniform_pars=uniform, pars=len(pars), sents=sents)


def lint(path, issues, cites):
    ext = os.path.splitext(path)[1].lower()
    text, nums, prose = views(load(path), ext)
    ln = lambda s, pos: s.count("\n", 0, pos) + 1
    snip = lambda s, m: re.sub(r"\s+", " ", s[max(0, m.start() - 30):m.start()] + "«" + m.group() + "»" + s[m.end():m.end() + 30]).strip()
    rules = [("ERROR", "dash", r"—|(?<=\s)–(?=\s)|---|(?<=\s)--(?=\s)", 0),
             ("ERROR", "banned-word", HARD, re.I),
             ("ERROR", "contraction", r"\b\w+(?:n't|'re|'ll|'ve|'d|'m)\b|\b(?:it|that|there|here|what|let)'s\b", re.I),
             ("ERROR", "marker", MARK, 0),
             ("ERROR", "process-residue", RESID_ERR, 0),
             ("WARN", "process-residue?", RESID_WARN, 0),
             ("WARN", "first-person-singular", r"(?<![\w$])I(?![\w$.])(?! \d)", 0),
             ("WARN", "generic-opener", r"\bIn recent years\b|\bhas become (?:increasingly|a)\b|\bmore and more\b|\brapid(?:ly)? (?:advances|growth|development|progress)\b", re.I)]
    for lvl, tag, pat, fl in rules:
        src = text if tag in ("marker", "dash") else prose
        for m in re.finditer(pat, src, fl):
            issues.append((lvl, path, ln(src, m.start()), tag, snip(src, m)))
    s = stats(prose)
    allS = [x for ps in s["sents"] for x in ps]
    at = lambda x: ln(prose, max(prose.find(x[:40]), 0))
    prev = False
    for x in allS:
        cur = bool(re.match(TRANS, x))
        if cur and prev:
            issues.append(("WARN", path, at(x), "stacked-transition", x[:90]))
        prev = cur
    inst = {"watch": lambda: [m for m in re.finditer(r"\b(?:%s)\b" % "|".join(sorted(WATCH)), prose, re.I)],
            "part": lambda: list(re.finditer(PART, prose)),
            "anti": lambda: list(re.finditer(ANTI, prose, re.I)),
            "triad": lambda: list(re.finditer(r"\b\w+, \w+,? (?:and|or) \w+\b", prose))}
    sent_pat = {"trans": TRANS, "candid": CANDID, "summ": SUMMARY}
    if s["words"] >= 800:
        tripped = [g for g in GATES if s[g[0]] > g[1]]
        lvl = "WARN" if len(tripped) >= 2 else "NOTE"
        if tripped:
            msg = ("%d density flags: outside the human range (1%% of human papers, 86%% of 2025 abstract batches)" % len(tripped)
                   if len(tripped) >= 2 else "1 density flag: about 1 in 10 human papers has one (1 in 6 at clinical venues), fix only if the prose reads badly")
            issues.append((lvl, path, 0, "density-summary", msg))
        for key, lim, unit, what in tripped:
            issues.append((lvl, path, 0, "density", f"{what}: {s[key]:.2f} {unit} (human p98 {lim})"))
            if key in inst:
                for m in inst[key]()[:8]: issues.append((lvl, path, ln(prose, m.start()), key, snip(prose, m)))
            else:
                for x in [y for y in allS if re.match(sent_pat[key], y)][:8]: issues.append((lvl, path, at(x), key, x[:90]))
        vg = 1000.0 / s["words"] * sum(bool(re.match(VAGUE, x)) for x in allS)
        if vg > 3.0:
            issues.append(("WARN", path, 0, "vague-reference", f"{vg:.1f} sentences per 1k words open with This/It/These + verb (human p98 3.0)"))
            for x in [y for y in allS if re.match(VAGUE, y)][:8]: issues.append(("WARN", path, at(x), "vague-reference", x[:90]))
        for key, pat, lim, what in LOGGATES:
            ms = list(re.finditer(pat, prose)); d = 1000.0 * len(ms) / s["words"]
            if d > lim:
                issues.append(("WARN", path, 0, key, f"{d:.2f} per 1k words: {what} (above {lim} in under 2% of human papers)"))
                for m in ms[:8]: issues.append(("WARN", path, ln(prose, m.start()), key, snip(prose, m)))
        ua = undefined_acronyms(prose)
        if ua:
            issues.append(("WARN" if len(ua) > 12 else "NOTE", path, 0, "undefined-acronym",
                           f"{len(ua)} used twice or more with no definition (human median about 4): " + ", ".join(ua[:15])))
        if s["sentences"] >= 20:
            notes = []
            if not 15 <= s["mean"] <= 32: notes.append("mean sentence length outside 15-32 words")
            if s["sd"] < 6: notes.append("sentence length SD under 6 (very uniform rhythm)")
            if s["we"] < 11: notes.append("under 11 we/our per 1k words (agentless or nominal style)")
            if s["uniform_pars"] > 0.3 * max(s["pars"], 1): notes.append("many paragraphs have near-identical sentence lengths")
            for n_ in notes: issues.append(("WARN", path, 0, "sanity", n_))
    else:
        issues.append(("WARN", path, 0, "short-text", f"{s['words']} words: density checks skipped, run on the whole paper"))
    print(f"{path}: {s['words']} words, {s['sentences']} sentences | mean {s['mean']:.1f} sd {s['sd']:.1f} | we {s['we']:.0f}/1k | "
          f"tierB {s['watch']:.2f}% | tails {s['part']:.1f}/1k | transitions {s['trans']:.1f}% | triads {s['triad']:.1f}/1k | "
          f"candid {s['candid']:.1f}/1k | antithesis {s['anti']:.1f}/1k | summaries {s['summ']:.1f}/1k")
    for m in re.finditer(CITE, text):
        cites += [(k.strip(), ln(text, m.start())) for k in m.group(1).split(",") if k.strip()]
    for m in re.finditer(r"\[((?:-?@[\w:.\-]+[;,\s]*)+)\]", text):
        cites += [(k, ln(text, m.start())) for k in re.findall(r"@([\w:.\-]+)", m.group(1))]
    return nums


def citation_audit(path, cites, bib, log, issues):
    keys = {}
    if bib:
        bt = load(bib)
        for blk in re.split(r"\n\s*(?=@\w+\s*\{)", bt):
            m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", blk)
            if not m or m.group(1).lower() in ("comment", "string", "preamble"): continue
            keys[m.group(2)] = blk
            for f, alt in (("author", ("editor",)), ("title", ()), ("year", ("date",))):
                if not any(re.search(r"\b%s\s*=" % a, blk, re.I) for a in (f,) + alt):
                    issues.append(("WARN", bib, 0, "bib-field", f"{m.group(2)} lacks {f}"))
            if not re.search(r"\b(?:journal|booktitle|publisher|eprint|howpublished|institution|url|doi)\s*=", blk, re.I):
                issues.append(("WARN", bib, 0, "bib-field", f"{m.group(2)} lacks venue/eprint/url"))
    status = {}
    if log:
        for line in load(log).splitlines():
            m = re.search(r"\b(UNVERIFIED|METADATA-ONLY|VERIFIED)\b", line)
            cells = [c.strip(" `*") for c in line.split("|") if c.strip()]
            if m and cells:
                k = cells[0] if "|" in line else re.sub(r"^[\s\-*`]*([\w:.\-]+).*$", r"\1", line)
                status[k] = m.group(1)
    seen = set()
    for k, line in cites:
        if k in seen: continue
        seen.add(k)
        if bib and k not in keys: issues.append(("ERROR", path, line, "cite-missing-in-bib", k))
        if log:
            s = status.get(k)
            if s is None: issues.append(("ERROR", path, line, "cite-not-verified", f"{k}: no record in log"))
            elif s == "UNVERIFIED": issues.append(("ERROR", path, line, "cite-not-verified", f"{k}: UNVERIFIED"))
            elif s == "METADATA-ONLY": issues.append(("WARN", path, line, "cite-metadata-only", f"{k}: exists, content not checked"))
    for k in keys:
        if k not in seen: issues.append(("WARN", bib, 0, "bib-unused", k))


def number_audit(path, nums, ledger, issues):
    def vals(s):
        out = set()
        for t in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", s):
            try: v = float(t.rstrip("%").replace(",", ""))
            except ValueError: continue
            out.update({round(v, 6), round(v * 100, 6), round(v / 100, 6)})
        return out
    ok = vals(load(ledger))
    nums = blank(r"\b(?:Sections?|Sec\.|Figures?|Fig\.|Tables?|Appendix|Appendices|Eq\.|Equations?|Theorem|Lemma|Proposition|Corollary|Algorithm|Hypothesis|Definition|Assumption|Step|Phase|Experiment)s?\s+[\dA-Z][\w.\-]*", nums)
    for m in re.finditer(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", nums):
        t = m.group()
        try: v = float(t.rstrip("%").replace(",", ""))
        except ValueError: continue
        if (v == int(v) and v <= 20 and not t.endswith("%")) or 1900 <= v <= 2099: continue
        if round(v, 6) not in ok:
            issues.append(("ERROR", path, nums.count("\n", 0, m.start()) + 1, "number-not-in-ledger", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bib"); ap.add_argument("--log"); ap.add_argument("--ledger")
    a = ap.parse_args()
    issues, cites, allnums = [], [], []
    for f in a.files:
        allnums.append((f, lint(f, issues, cites)))
    if a.bib or a.log: citation_audit(a.files[0], cites, a.bib, a.log, issues)
    if a.ledger:
        for f, n in allnums: number_audit(f, n, a.ledger, issues)
    shown = {}
    for lvl, f, line, tag, sn in sorted(issues, key=lambda x: ({"ERROR": 0, "WARN": 1}.get(x[0], 2), x[3], x[1], x[2])):
        shown[tag] = shown.get(tag, 0) + 1
        if shown[tag] <= 8:
            print(f"{lvl:5s} {f}:{line}  [{tag}] {sn}")
        elif shown[tag] == 9:
            print(f"      ... more [{tag}] findings hidden")
    e = sum(i[0] == "ERROR" for i in issues); w = sum(i[0] == "WARN" for i in issues); nn = len(issues) - e - w
    print(f"\n{e} errors, {w} warnings, {nn} notes")
    sys.exit(1 if e else 0)


if __name__ == "__main__":
    main()
