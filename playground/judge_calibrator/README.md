# judge_calibrator — "Pick your Judge"

L2 lab. In L1 the judge was perfect; here it isn't. The student turns a **bias** dial and a
**noise** dial over a frozen pool of answers with a *known* true score, watches the measured
number shift and wobble off the truth, and then **bootstraps** the result to see what each
defect does to the estimate.

**Source of uncertainty in focus:** bias and noise around a score — the judge is part of the
measurement instrument, not an objective oracle. Distinct from L1's sampling noise (picking
different cases); here the offset and the wobble come from the judge itself.

## The one idea: bias ≠ noise

- **Bias is systematic.** A consistently generous or harsh judge shifts every score the same
  direction. It moves the *centre* of your measurement off the truth, and **more data does not
  fix it** — it just tightens a confident interval around the wrong value.
- **Noise is random.** An inconsistent judge (temperature, an ambiguous rubric, a different
  annotator) scores the same answer differently on re-run. It **widens** the spread, and it
  *does* average out with more judgments.

The payoff is the **bootstrap** — the general-purpose successor to L1's Beta interval (it works
for any score: binary, ordinal, continuous). Resample the scored test set with replacement,
recompute the mean thousands of times, read off the 95% interval. The sharp beat: **noise widens
the interval; bias slides it off the truth — and the bootstrap can't see the bias.** A biased
judge hands you a tight, confident interval centred on the wrong answer.

## Why a graded score (not pass/fail)

The judge gives a graded quality score (0–100), not a binary stamp. This is both truer to gen-AI
answers (which sit on a spectrum) and necessary statistically: a binary judge's pass-rate
variance is pinned by its mean, so judge noise gets absorbed and does *not* widen a single run's
interval. A graded score lets noise add genuine variance (`Var ≈ σ²/n`), so the "noise widens the
interval" beat is both visible and honest.

## The HTML simulator (`index.html`)

No-code, fully offline, no dependencies. A frozen pool of 40 (question, answer) pairs, each with
a known true quality; the pool's true mean is **70/100** by construction. Controls:

- **Bias** ∈ [−25, +25] pts — harsher ↔ more generous. A pure centre-shift.
- **Noise** (σ) ∈ [0, 30] pts — perfectly consistent ↔ very inconsistent. A pure spread.
- **Test-set size** ∈ [20, 500] — watch the asymmetry vs. `n`: more data narrows the interval
  around the noise-blurred mean, but the bias offset stays (and the interval excludes the truth
  more decisively).
- **Re-judge** — re-roll the noise and redraw the test set; the number jumps under noise, stays
  put under bias.

A per-row table shows each answer's true quality next to the judge's rating; a bootstrap panel
draws the distribution of the mean with the 95% interval, the true-70 line, and a coverage
readout.

## The notebook (`notebook.ipynb`) — real LLM judges

The HTML is an abstract, fully controllable simulator; the notebook grounds the same three beats
on real judges over real answers:

1. **The exact-match trap → bias.** Programmatic exact-match vs. a lenient LLM judge on free-text
   QA answers; exact-match's systematic under-counting *is* a negative bias, and the gap is the
   bias made visible.
2. **Re-run noise.** One LLM judge, one answer, K calls at `temperature>0`; the per-example score
   spread is the judge's own noise.
3. **Bootstrap the estimate.** From one scored pass, bootstrap the answers for a 95% interval on
   the mean score (the code behind the HTML panel); optionally compare a generous vs. a harsh
   rubric to watch the interval *shift*, not just widen.

## What's inside

- `index.html` — the two-dial simulator + bootstrap (no-code path).
- `notebook.ipynb` — real LLM judges + bootstrap on real answers (code path).
- `lab.py` — judge runners (programmatic + LLM), loaders, and a bootstrap helper, with a
  deterministic mock fallback when `LIVE != true`.
- `prompts/` — judge rubrics (strict / lenient / scale 1–5, plus a no-ground-truth quality
  rubric) and the reply-generation prompt, as Jinja2 templates.
- `answers/` — pre-computed AI answers/replies with true-quality labels, committed so the lab
  runs offline.
- `.env.example` — copy to `.env` and add an API key to run the LLM live (`LIVE=true`).

## Datasets

From `../../datasets/`:
- **QA set** — small factual short-answer data: question, gold answer, free-text AI answer
  (TriviaQA slice; swappable for SQuAD / NQ-open). Paraphrases and added units are what make
  exact match under-count.
- **`bitext_customer_support`** — reused from L1 for the no-ground-truth variant: the AI drafts a
  reply with no reference, so only a rubric judge can score it and inter-judge disagreement is the
  only signal.

## Running

```
pip install pandas python-dotenv jinja2 matplotlib
# optional, for live LLM calls / re-generating answers: pip install openai anthropic
cp .env.example .env
jupyter notebook notebook.ipynb
```

The HTML needs nothing — open `index.html` in a browser. Without API keys the notebook uses the
committed answers and a deterministic mock judge; re-run noise and re-generating answers need a
live LLM with `LIVE=true`.
