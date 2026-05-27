"""Helper functions for the L2 'Pick your Judge' lab.

Three judge configurations (strict, lenient, 1-5 scale), each implemented as an LLM call
with a Jinja2 prompt. Mock fallback when LIVE != true so the lab runs offline.
"""
import os
from pathlib import Path
import pandas as pd
import jinja2

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"


def load_examples():
    return pd.read_csv(HERE / "examples.csv")


def render_prompt(template_name, **kwargs):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    return env.get_template(template_name).render(**kwargs)


def call_llm(prompt, temperature=0.0):
    """Call an LLM. Returns model text, or None if LIVE != true."""
    if os.getenv("LIVE", "false").lower() != "true":
        return None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
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
    return None


def _parse_score(response, kind):
    if response is None:
        return None
    s = response.strip().splitlines()[0].strip().rstrip(".")
    try:
        v = float(s)
    except Exception:
        return None
    if kind == "binary" and v not in (0, 1):
        return None
    if kind == "scale_1_5" and not (1 <= v <= 5):
        return None
    return v


def _norm(s):
    return s.strip().rstrip(".,!?").lower()


def _mock_score(question, ai_answer, reference, judge):
    """Deterministic mock scoring used when LIVE != true."""
    a, r = _norm(ai_answer), _norm(reference)
    if judge == "strict":
        return 1 if a == r else 0
    if judge == "lenient":
        return 1 if r in a else 0
    if judge == "scale_1_5":
        if a == r:                                   return 5
        if r in a:                                   return 4
        if set(a.split()) & set(r.split()):          return 3
        if any(c.isalpha() for c in a):              return 2
        return 1
    raise ValueError(f"unknown judge: {judge}")


JUDGE_TO_TEMPLATE = {
    "strict":    "strict.j2",
    "lenient":   "lenient.j2",
    "scale_1_5": "scale_1_5.j2",
}
JUDGE_TO_KIND = {
    "strict":    "binary",
    "lenient":   "binary",
    "scale_1_5": "scale_1_5",
}


def score_one(judge, question, ai_answer, reference, temperature=0.0):
    """Run one judge on one example. Returns a numeric score (None on parse fail)."""
    prompt = render_prompt(
        JUDGE_TO_TEMPLATE[judge],
        question=question, ai_answer=ai_answer, reference=reference,
    )
    response = call_llm(prompt, temperature=temperature)
    if response is None:
        return _mock_score(question, ai_answer, reference, judge)
    return _parse_score(response, JUDGE_TO_KIND[judge])


def score_all(df, judge, temperature=0.0, n_reruns=1):
    """Score every row of df with the given judge. With n_reruns>1, runs the judge multiple
    times per example (useful to expose re-run variance when temperature > 0)."""
    rows = []
    for r in range(n_reruns):
        for i, ex in df.iterrows():
            s = score_one(judge, ex["question"], ex["ai_answer"], ex["reference"], temperature)
            rows.append({
                "example_id": i, "rerun": r, "judge": judge,
                "question": ex["question"], "ai_answer": ex["ai_answer"], "score": s,
            })
    return pd.DataFrame(rows)
