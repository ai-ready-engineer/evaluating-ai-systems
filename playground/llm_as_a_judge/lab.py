"""Helper functions for the L2 "LLM as a Judge" lab.

Source of uncertainty in focus: **variance from the scoring instrument itself** — the same
frozen output, scored twice, can get two answers. The wobble does not come from picking
different cases (that was L1's sampling noise); it comes from *which judge you happened to
pick and how its rubric is worded*.

The lab scores one fixed set of ITSM (IT-support) agent replies with several judges:

  * programmatic judges  — exact match / normalized match
  * reference-based judge — token-overlap vs the golden reference
  * LLM-as-judge          — strict / lenient binary rubrics and a 1-5 scale rubric

Each judge is a stub: with ``LIVE != true`` it falls back to a deterministic mock (mirroring
L1's ``call_llm`` pattern), so the whole lab rebuilds offline with no API key and no network.

It also carries two human annotators per reply, so we can ask the L2 "judge of judges"
question: how often does a judge agree with a human, and what is Cohen's kappa between them?

The data is **ITSM-flavored**: frozen agent replies to support tickets, each with a senior
engineer's reference answer and two human quality labels. ``examples.csv`` ships a small
hand-written QA set used by the original Demo A/B; ``build_judgments.py`` synthesizes the
larger ITSM dataset deterministically and writes it under ``judgments/``.
"""
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import jinja2

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
JUDGMENTS_DIR = HERE / "judgments"


# --- Dataset loaders ---

def load_examples():
    """The tiny hand-written QA triples (question, ai_answer, reference). Used by the
    classic strict/lenient/scale demo and as a quick exact-match-trap illustration."""
    return pd.read_csv(HERE / "examples.csv")


def load_judgments(dataset="itsm_replies"):
    """Load the synthesized ITSM dataset written by build_judgments.py.

    Returns a DataFrame with columns: ticket, reference, reply, quality (intended ground
    truth 0/1), human_1, human_2, variant, length. Run build_judgments.py first."""
    path = JUDGMENTS_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python build_judgments.py` to synthesize it first.")
    obj = json.loads(path.read_text())
    return pd.DataFrame(obj["items"])


# --- Text normalization shared by the programmatic judges ---

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s):
    """Lower-case, strip, collapse whitespace, drop trailing punctuation."""
    return re.sub(r"\s+", " ", str(s).strip().rstrip(".,!?;:").lower())


def _tokens(s):
    return _WORD_RE.findall(str(s).lower())


# --- Programmatic judges (no reference comparison beyond string equality) ---

def exact_match(reply, reference):
    """Strict programmatic judge: 1 iff the reply string equals the reference string
    byte-for-byte (after a light strip). The harshest possible comparator — any paraphrase,
    added courtesy, or extra detail scores 0."""
    return int(str(reply).strip() == str(reference).strip())


def normalized_match(reply, reference):
    """Slightly kinder programmatic judge: 1 iff the *normalized* reply equals the
    normalized reference (case-folded, whitespace-collapsed, trailing punctuation dropped).
    Catches casing/spacing differences but still fails on any real paraphrase."""
    return int(_norm(reply) == _norm(reference))


# --- Reference-based judge ---

def token_overlap_score(reply, reference):
    """Reference-based soft judge: token-level F1 between reply and reference (a BoW
    overlap, the spirit of ROUGE/token-F1). Returns a float in [0, 1].

    Rewards a reply that *contains the reference's words*, so paraphrases that reuse the
    key terms score high — but it has no notion of meaning, so a verbose reply that pads
    in the reference's words can score well even when it buries or contradicts them.
    """
    a, b = _tokens(reply), _tokens(reference)
    if not a or not b:
        return 0.0
    ca, cb = {}, {}
    for t in a:
        ca[t] = ca.get(t, 0) + 1
    for t in b:
        cb[t] = cb.get(t, 0) + 1
    overlap = sum(min(ca[t], cb.get(t, 0)) for t in ca)
    if overlap == 0:
        return 0.0
    prec = overlap / len(a)
    rec = overlap / len(b)
    return 2 * prec * rec / (prec + rec)


def token_overlap_match(reply, reference, threshold=0.5):
    """Binarize token_overlap_score at a threshold so it sits next to the binary judges."""
    return int(token_overlap_score(reply, reference) >= threshold)


# --- LLM-as-judge: a stub with a deterministic mock fallback (mirrors L1.call_llm) ---

JUDGE_TO_TEMPLATE = {
    "strict":     "judge_strict.j2",
    "lenient":    "judge_lenient.j2",
    "scale_1_5":  "judge_scale_1_5.j2",
}
JUDGE_TO_KIND = {
    "strict":     "binary",
    "lenient":    "binary",
    "scale_1_5":  "scale_1_5",
}


def render_prompt(template_name, **kwargs):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    return env.get_template(template_name).render(**kwargs)


def call_llm(prompt, temperature=0.0):
    """Call an LLM (OpenRouter / OpenAI / Anthropic). Returns model text, or None if
    LIVE != true. Same shape as L1's call_llm so the offline path is identical: a None
    return tells the caller to use its deterministic mock."""
    if os.getenv("LIVE", "false").lower() != "true":
        raise NotImplementedError(
            "Live LLM judging is disabled (LIVE != true). The deterministic mock judge is "
            "used instead so the lab runs offline. Set LIVE=true and an API key to call a "
            "real model.")
    if os.getenv("OPENROUTER_API_KEY"):
        from openai import OpenAI  # OpenRouter speaks the OpenAI API
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=90.0,
            max_retries=5,
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(timeout=90.0, max_retries=5)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=10,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    raise NotImplementedError(
        "LIVE=true but no API key found (OPENROUTER_API_KEY / OPENAI_API_KEY / "
        "ANTHROPIC_API_KEY). Add one to .env or set LIVE=false to use the mock.")


def _parse_score(response, kind):
    if response is None:
        return None
    s = response.strip().splitlines()[0].strip().rstrip(".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    v = float(m.group(0))
    if kind == "binary" and v not in (0, 1):
        return None
    if kind == "scale_1_5" and not (1 <= v <= 5):
        return None
    return v


def _mock_judge(judge, ticket, reference, reply):
    """Deterministic mock LLM judge used when LIVE != true.

    It mimics two real behaviours of an LLM-as-judge so the demos land:

    1. **Meaning over surface.** It keys off token overlap with the reference rather than
       exact string equality, so a correct paraphrase passes where exact_match fails.
    2. **Verbosity bias.** A longer reply nudges the judge toward a higher score even when
       its content overlap is unchanged — the classic "the wordy answer looks more
       thorough" failure. The lenient judge is the most length-susceptible; strict is the
       least. This is what verbosity_bias_curve() makes visible.

    The mock is a pure function of (judge, ticket, reference, reply) — no randomness — so
    rebuilds are byte-identical.
    """
    overlap = token_overlap_score(reply, reference)
    n_tok = len(_tokens(reply))
    # A small, saturating verbosity bonus in "overlap units". Longer => looks more thorough.
    verb = min(0.20, 0.012 * n_tok)
    # Per-judge temperament: lenient is generous and very length-susceptible; strict is
    # harsh and barely moved by length; the scale judge sits in between.
    if judge == "strict":
        eff = overlap + 0.25 * verb
        return 1 if eff >= 0.75 else 0
    if judge == "lenient":
        eff = overlap + 1.0 * verb
        return 1 if eff >= 0.45 else 0
    if judge == "scale_1_5":
        eff = overlap + 0.6 * verb
        if eff >= 0.85:  return 5
        if eff >= 0.65:  return 4
        if eff >= 0.45:  return 3
        if eff >= 0.20:  return 2
        return 1
    raise ValueError(f"unknown judge: {judge}")


def llm_judge(judge, ticket, reference, reply, temperature=0.0):
    """Score one (ticket, reference, reply) with an LLM rubric judge.

    Renders the judge's Jinja2 prompt and calls the model; on the offline path (LIVE != true,
    so call_llm raises NotImplementedError) it falls back to the deterministic mock. Returns
    a numeric score (1/0 for binary judges, 1-5 for the scale judge), or None on parse fail.
    """
    prompt = render_prompt(
        JUDGE_TO_TEMPLATE[judge], ticket=ticket, reference=reference, reply=reply)
    try:
        response = call_llm(prompt, temperature=temperature)
    except NotImplementedError:
        return _mock_judge(judge, ticket, reference, reply)
    parsed = _parse_score(response, JUDGE_TO_KIND[judge])
    if parsed is None:                       # tolerate a malformed live response
        return _mock_judge(judge, ticket, reference, reply)
    return parsed


def llm_judge_all(df, judge, temperature=0.0):
    """Run an LLM rubric judge over every row of a judgments DataFrame. Returns a 1-D
    numpy array of scores aligned to df's rows."""
    return np.array([
        llm_judge(judge, r["ticket"], r["reference"], r["reply"], temperature)
        for _, r in df.iterrows()
    ], dtype=float)


# --- Agreement / calibration (the "judge of judges") ---

def agreement(a, b):
    """Raw agreement: fraction of items where two label vectors match. Both must be the
    same length; entries are compared with ==, so use it on binary (or already-discretized)
    labels."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    return float(np.mean(a == b))


def cohen_kappa(a, b):
    """Cohen's kappa between two binary raters — agreement corrected for chance.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is the
    agreement expected if both raters labelled independently at their own base rates.
    kappa = 1 perfect, 0 chance-level, < 0 worse than chance. Hand-rolled (no sklearn
    dependency) and restricted to binary labels, which is all the lab needs.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    p_o = np.mean(a == b)
    pa1, pb1 = np.mean(a == 1), np.mean(b == 1)
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if p_e == 1.0:                           # both raters constant and identical
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1 - p_e))


# --- Verbosity bias ---

def verbosity_bias_curve(df, judge, scores=None, n_bins=4):
    """Does the judge reward length independent of correctness?

    Bins replies by length (token count) and, *within the subset of replies whose intended
    quality is the same* (so correctness is held roughly fixed), reports the judge's mean
    score per length bin. If the mean climbs with length, the judge is paying for words.

    Returns a DataFrame: length_bin, n, mean_len, mean_score (and mean_quality for context).
    Pass ``scores`` to reuse an already-computed score vector; otherwise it scores df here.
    """
    d = df.copy().reset_index(drop=True)
    if scores is None:
        scores = llm_judge_all(d, judge)
    d["_score"] = np.asarray(scores, dtype=float)
    if "length" not in d.columns:
        d["length"] = d["reply"].apply(lambda s: len(_tokens(s)))
    # Hold correctness fixed: study only the replies meant to be CORRECT, where any score
    # lift across length bins is bias, not content. (quality == 1 are the "good" replies.)
    good = d[d["quality"] == 1].copy()
    if len(good) < n_bins:
        good = d.copy()
    good["length_bin"] = pd.qcut(good["length"], q=min(n_bins, good["length"].nunique()),
                                 labels=False, duplicates="drop")
    out = good.groupby("length_bin").agg(
        n=("_score", "size"),
        mean_len=("length", "mean"),
        mean_score=("_score", "mean"),
        mean_quality=("quality", "mean"),
    ).reset_index()
    return out


# --- QA-pairs judge API (the question/ai_answer/reference triples in examples.csv) ---
# The "LLM as a Judge" demo path used by notebook.ipynb: the same three rubrics
# (strict / lenient / scale_1_5) applied to short QA answers, where the headline number
# swings with the judge. It sits beside the ITSM judgments API above; the two share the
# judge names and JUDGE_TO_KIND but use different prompts — the QA prompts take
# question/ai_answer/reference, the ITSM prompts take ticket/reference/reply.

QA_JUDGE_TO_TEMPLATE = {
    "strict":    "strict.j2",
    "lenient":   "lenient.j2",
    "scale_1_5": "scale_1_5.j2",
}


def _mock_qa_score(question, ai_answer, reference, judge):
    """Deterministic mock judge for QA triples, used on the offline path (LIVE != true).

    strict    — 1 iff the normalized answer equals the reference (exact-ish);
    lenient   — 1 iff the reference text is contained in the answer (accepts paraphrase/extra);
    scale_1_5 — graded by how much of the reference the answer captures.
    A pure function of its inputs, so rebuilds are byte-identical.
    """
    a, r = _norm(ai_answer), _norm(reference)
    if judge == "strict":
        return 1 if a == r else 0
    if judge == "lenient":
        return 1 if r in a else 0
    if judge == "scale_1_5":
        if a == r:                            return 5
        if r in a:                            return 4
        if set(a.split()) & set(r.split()):   return 3
        if any(c.isalpha() for c in a):       return 2
        return 1
    raise ValueError(f"unknown judge: {judge}")


def score_one(judge, question, ai_answer, reference, temperature=0.0):
    """Run one rubric judge on one QA triple. The live path renders the QA prompt and calls
    the LLM; offline (call_llm raises NotImplementedError) or on a parse failure it falls back
    to the deterministic mock — so the demo always returns a number."""
    try:
        response = call_llm(
            render_prompt(QA_JUDGE_TO_TEMPLATE[judge],
                          question=question, ai_answer=ai_answer, reference=reference),
            temperature=temperature)
    except NotImplementedError:
        return _mock_qa_score(question, ai_answer, reference, judge)
    parsed = _parse_score(response, JUDGE_TO_KIND[judge])
    if parsed is None:                        # tolerate a malformed live response
        return _mock_qa_score(question, ai_answer, reference, judge)
    return parsed


def score_all(df, judge, temperature=0.0, n_reruns=1):
    """Score every QA row (question, ai_answer, reference) with the given rubric judge.

    Returns a long DataFrame with columns example_id, rerun, judge, question, ai_answer,
    score. With n_reruns > 1 each example is scored several times (to expose a live judge's
    re-run variance at temperature > 0); offline the mock is deterministic so re-runs match.
    """
    rows = []
    for r in range(n_reruns):
        for i, ex in df.iterrows():
            rows.append({
                "example_id": i, "rerun": r, "judge": judge,
                "question": ex["question"], "ai_answer": ex["ai_answer"],
                "score": score_one(judge, ex["question"], ex["ai_answer"],
                                   ex["reference"], temperature),
            })
    return pd.DataFrame(rows)
