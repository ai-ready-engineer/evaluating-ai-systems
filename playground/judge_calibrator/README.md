# judge_calibrator — "Pick your Judge"

L2 lab. The same frozen AI answers, scored by different judges over different data — and
the headline accuracy moves. The point is to *feel* how **sensitive** a reported number is
to how the judge is worded and which data you happened to score.

**Source of uncertainty in focus:** variance around a score — the judge is part of the
measurement instrument, not an objective oracle. Distinct from L1's sampling noise (picking
different cases); here the wobble comes from the judge prompt and the slice.

## The idea

You need a judge even when you *have* ground truth: exact string match marks plainly-correct
free-text answers wrong ("Paris, the City of Light", "approximately 300,000 km/s"). Once you
reach for a judge, two situations appear:

- **Case 1 — ground truth exists, but it's hard to compare.** The judge is a smarter
  comparator against a known reference (the QA set).
- **Case 2 — no ground truth at all.** The judge *is* the definition of "good"; nothing to
  compare against (drafted customer-support replies). All you can observe is judges
  disagreeing with each other.

## What's inside

- `lab.py` — judge runners (programmatic + LLM) and dataset/answer loaders, with a
  deterministic mock fallback when `LIVE != true`.
- `notebook.ipynb` — the lab: the exact-match trap, the case-1 sensitivity grid, case-2
  no-ground-truth disagreement, and an optional re-run-noise view.
- `index.html` — no-code path: turn the dials (judge × slice) over the pre-computed scores.
- `prompts/` — judge rubrics and the reply-generation prompt, as Jinja2 templates.
- `answers/` — pre-computed AI answers/replies, committed so the lab runs offline.
- `.env.example` — copy to `.env` and add an API key to run the LLM live (`LIVE=true`).

## Datasets

From `../../datasets/`:
- **QA set (case 1)** — small factual short-answer data: question, gold answer, free-text AI
  answer. Default a slice of TriviaQA (swappable for SQuAD / NQ-open). The paraphrases and
  added units are what make exact match fail.
- **`bitext_customer_support` (case 2)** — reused from L1; the AI drafts a reply to each
  message and there is no reference reply, so only a rubric judge can score it.

## Judges

- **Programmatic baseline** — exact match / regex / schema check. Works on the easy case-1
  rows, fails on paraphrase, and *cannot run at all* on case 2.
- **LLM judges** — `strict` (binary, exact), `lenient` (binary, accepts paraphrase),
  `scale_1_5` (ordinal), plus reworded-but-equivalent rubric variants for case 1 and a
  quality rubric for case 2. Same task, different temperament and wording.

## What it shows

1. **The exact-match trap** — ground truth is present, yet the naive comparator already lies.
2. **The sensitivity grid (case 1)** — score the same answers across judge prompts × data
   slices; the accuracy spreads across both axes. Same system, different number.
3. **No ground truth (case 2)** — rubric judges land far apart on the same reply, and nothing
   in the data can adjudicate between them.
4. **Re-run noise (optional)** — one judge at `temperature>0` wobbles with prompt and data
   fixed. Secondary; the prompt × data sensitivity is the headline.

## Running

```
pip install pandas python-dotenv jinja2 matplotlib
# optional, for live LLM calls / re-generating answers: pip install openai anthropic
cp .env.example .env
jupyter notebook notebook.ipynb
```

Without API keys the lab uses the committed pre-computed answers and a deterministic mock
judge, so the exact-match trap and the sensitivity grid still run. Re-run noise (Demo 4) and
re-generating answers need a live LLM with `LIVE=true`.
