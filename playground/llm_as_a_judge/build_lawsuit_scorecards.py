"""L2 'LLM as a judge — scorecard' dataset: summarize legal cases, then score each summary.

Reads the cached Multi-LexSum subset (../../datasets/multi_lexsum/cases.jsonl, built by that
folder's build_cache.py), and for each case:

  1. SUMMARIZE the full source with an LLM (no truncation — cases are < 20k words).
  2. JUDGE the summary with an LLM scorecard, reference-based against the expert short
     summary: Accuracy / Completeness / Fluency (1-5) + a holistic Overall (1-5, halves).

Writes lawsuits/scorecards.json — the committed artifact the lab/HTML browses. Mirrors slide
11 of L2_main.pptx (the lawsuit-summarizer scorecard). Live-only (needs LIVE=true + a key);
there is no mock — this builds a real dataset.

    LIVE=true uv run python build_lawsuit_scorecards.py --n 15        # dry run, eyeball quality
    LIVE=true uv run python build_lawsuit_scorecards.py --n 200       # full build
    uv run python build_lawsuit_scorecards.py --model openai/gpt-5.5  # pick the model
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import jinja2

HERE = Path(__file__).resolve().parent
CASES = HERE.parent.parent / "datasets" / "multi_lexsum" / "cases.jsonl"
OUTDIR = HERE / "lawsuits"
PROMPTS = jinja2.Environment(loader=jinja2.FileSystemLoader(str(HERE / "prompts")))

_USAGE = {"prompt": 0, "completion": 0, "calls": 0}

# Overall is COMPUTED from the three judged dimensions (not asked of the judge), so it is a
# fixed, transparent rule. Substance dominates; fluency barely counts (it's near-constant for
# a capable model). These weights reproduce slide 11's example: 0.45*3 + 0.45*4 + 0.10*2 = 3.35
# -> 3.5 when rounded to the nearest half.
W_ACCURACY, W_COMPLETENESS, W_FLUENCY = 0.45, 0.45, 0.10


def compute_overall(sc):
    raw = (W_ACCURACY * sc["accuracy"] + W_COMPLETENESS * sc["completeness"]
           + W_FLUENCY * sc["fluency"])
    return max(1.0, min(5.0, round(raw * 2) / 2))   # nearest 0.5, clamped to [1, 5]


def complete(prompt):
    """One OpenRouter chat completion. No temperature override (gpt-5.x rejects non-default).
    Fails loud — this script builds a real dataset, so there is no silent mock."""
    if os.getenv("LIVE", "false").lower() != "true":
        sys.exit("LIVE != true — set LIVE=true and an OPENROUTER_API_KEY to build the dataset.")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("no OPENROUTER_API_KEY in .env")
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    api_key=key, timeout=180.0, max_retries=5)
    resp = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL"),
        messages=[{"role": "user", "content": prompt}],
    )
    u = getattr(resp, "usage", None)
    if u:
        _USAGE["prompt"] += u.prompt_tokens or 0
        _USAGE["completion"] += u.completion_tokens or 0
    _USAGE["calls"] += 1
    return resp.choices[0].message.content


def parse_scorecard(text):
    """Pull the JSON object out of the judge's reply; validate the four numeric fields."""
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in ("accuracy", "completeness", "fluency"):
        v = d.get(k)
        if not isinstance(v, (int, float)) or not (1 <= v <= 5):
            return None
        out[k] = float(v)
    out["overall"] = compute_overall(out)            # computed, not judged
    out["rationale"] = str(d.get("rationale", ""))[:300]
    return out


def excerpt(text, words=120):
    return " ".join(text.split()[:words])


def recompute_overall():
    """Recompute every case's Overall from its stored A/C/F (no LLM). Used after changing
    the weights, or to fix a build made when the judge still emitted its own Overall."""
    out = OUTDIR / "scorecards.json"
    data = json.loads(out.read_text())
    for c in data["cases"]:
        c["scorecard"]["overall"] = compute_overall(c["scorecard"])
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"recomputed Overall for {len(data['cases'])} cases "
          f"(weights A={W_ACCURACY} C={W_COMPLETENESS} F={W_FLUENCY}, nearest 0.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="openai/gpt-5.5")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute Overall from existing scorecards.json, no LLM calls")
    args = ap.parse_args()
    if args.recompute:
        recompute_overall()
        return
    os.environ["OPENROUTER_MODEL"] = args.model    # overrides .env default (gpt-5.2)
    os.environ.setdefault("LIVE", "true")

    if not CASES.exists():
        sys.exit(f"missing {CASES} — run datasets/multi_lexsum/build_cache.py first")
    cases = [json.loads(l) for l in CASES.open()][: args.n]

    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / "scorecards.json"
    # Resume: keep scorecards already built, skip those ids, and write after every new case
    # so an interruption (e.g. running out of API credits) never loses progress.
    results = json.loads(out.read_text())["cases"] if out.exists() else []
    done = {r["id"] for r in results}
    todo = [c for c in cases if c["id"] not in done]
    print(f"model {args.model} · {len(cases)} cases · {len(done)} already done · {len(todo)} to do\n")

    def save():
        out.write_text(json.dumps({
            "model": args.model, "n": len(results),
            "dimensions": ["accuracy", "completeness", "fluency"],
            "cases": results,
        }, ensure_ascii=False, indent=1))

    sum_tpl = PROMPTS.get_template("summarize_lawsuit.j2")
    judge_tpl = PROMPTS.get_template("judge_scorecard.j2")

    failed = 0
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        try:
            summary = (complete(sum_tpl.render(source=c["source"])) or "").strip()
            sc = parse_scorecard(complete(judge_tpl.render(reference=c["summary_short"], summary=summary)))
        except Exception as e:                       # one bad case must not kill the run
            failed += 1
            print(f"[{i:>3}] {(c['case_name'] or c['id']):<45.45}  ERROR: {type(e).__name__}: {str(e)[:80]}")
            continue
        if sc is None:
            failed += 1
            print(f"[{i:>3}] {c['case_name'] or c['id']:<45.45}  scorecard PARSE FAILED")
            continue
        results.append({
            "id": c["id"], "case_name": c["case_name"], "src_words": c["src_words"],
            "reference": c["summary_short"], "summary": summary, "scorecard": sc,
            "source_excerpt": excerpt(c["source"]),
        })
        save()                                       # incremental — crash-proof
        print(f"[{i:>3}] {(c['case_name'] or c['id']):<45.45}  "
              f"A{sc['accuracy']:.0f} C{sc['completeness']:.0f} F{sc['fluency']:.0f} "
              f"O{sc['overall']:.1f}  ({c['src_words']}w)")

    dt = time.time() - t0
    print(f"\nwrote {len(results)} scorecards to {out.relative_to(HERE.parent.parent)} "
          f"({failed} failed) in {dt:.0f}s")
    print(f"tokens: {_USAGE['prompt']:,} prompt + {_USAGE['completion']:,} completion "
          f"over {_USAGE['calls']} calls")
    if results:
        import statistics as st
        for d in ("accuracy", "completeness", "fluency", "overall"):
            xs = [r["scorecard"][d] for r in results]
            print(f"  mean {d}: {st.mean(xs):.2f}")


if __name__ == "__main__":
    main()
