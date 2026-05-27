# Playground for lesson 2

Same shape as L1, different axis. L1 froze the classifier and resampled the **data** to
watch the score wobble. Here we freeze the AI's answers and vary the **judge prompt** and
the **data slice** — and watch the *accuracy* move. The thing the student should walk away
feeling: a reported accuracy is not a fact about the system; it is **sensitive** to how the
judge is worded and to which data you happened to score.

## Assets

### Datasets

Kept deliberately small — **one new dataset** plus a set we already cached for L1.

**Dataset 1 — QA short-answer (case 1: ground truth exists, but hard to compare).** A
factual question-answering set: `question`, `gold` short answer, and a pre-generated
free-text `ai_answer`. Default a slice of **TriviaQA** (swappable for SQuAD / NQ-open),
pulled from HuggingFace and cached. The free-text answers are the point — paraphrases
("Paris, the City of Light"), added units ("approximately 300,000 km/s"), and extra context
("It stands for Central Processing Unit") are all correct yet none match the gold string.

**Dataset 2 — Bitext Customer Support (case 2: no ground truth at all).** Reused from L1.
For each customer message the AI drafts a **support reply**. There is no reference reply, so
exact match is not even an option — a rubric judge is the only scorer there is.

### Judges — programmatic baseline + LLM, side by side

Mirrors L1's "regular vs LLM" pairing, but now the thing under test is the **judge**:

- **Programmatic baseline** — exact match / regex / schema check. Works on the easy case-1
  rows, fails on paraphrase, and *cannot run at all* on case 2.
- **LLM judges** — a small family of rubrics, each a Jinja2 template in `prompts/`:
  - `strict` — binary 0/1, exact-meaning match only
  - `lenient` — binary 0/1, accepts paraphrase / extra context
  - `lenient_v2` — a **reworded-but-equivalent** lenient rubric (same intent, different
    wording) — included specifically to show that *cosmetic* rewording moves the number
  - `scale_1_5` — ordinal 1–5, binarised at a chosen cut (≥4 vs ≥3 are two different judges)
  - `reply_quality` — the case-2 rubric (helpful / correct / on-policy), 1–5

### Preparation

The "system under test" is frozen, exactly as L1 froze its classifiers. We **pre-generate
the AI answers once** and commit them — the lab never calls the model at runtime, it just
reads the stored answers and re-scores them.

- **Case 1:** hold out a fixed pool of **~300 QA items**; generate one free-text AI answer
  per item. Some are right, some wrong, many right-but-unmatchable.
- **Case 2:** take **~200 bitext messages**; generate one AI reply each. No reference exists.

For each (answer × judge prompt) we **pre-compute the judge's score** and store it, so the
HTML can flip judge and slice instantly. The offline default uses a deterministic mock
judge; a `LIVE=true` path re-generates answers and re-runs the LLM judges for those who want
to. Everything lives under `answers/` and is committed.

### Scores are per judge, per slice

A **score** is what one judge assigns one answer; the headline number is a judge's
aggregate over a slice:

- **Case 1** — each LLM judge's **pass rate** on a slice is an *estimate of the system's
  accuracy*. Strict, lenient, and the two scale thresholds give **different estimates of the
  same quantity**. (Optionally, a small hand-labelled "is this answer actually correct?"
  column lets us ask which judge is closest — the first taste of "judge of judges".)
- **Case 2** — there is no accuracy. We report each judge's **mean score** and the
  **disagreement between judges** on the same reply; that disagreement is the entire signal.

Score types appear naturally: **binary** (strict / lenient), **ordinal** (1–5), and the
binarised-ordinal that shows a threshold is itself a judge choice.

## Content flow for the html

The page opens with the hook, then splits into the two cases as tabs.

**0. The exact-match trap (intro, above the tabs).** Show a handful of QA rows with the gold
answer and the AI's free-text answer, and run the **programmatic exact-match** check live.
It marks plainly-correct answers wrong. Punchline: ground truth is present, yet the naive
comparator is already lying — so we need a judge, and a judge is a *choice*.

### Tab A — Case 1: ground truth, hard to compare

**1. Describe the dataset** — source, what "gold" looks like, why free-text answers break
exact match.

**2. The sensitivity grid** — the centrepiece. Rows = judge prompts (strict / lenient /
lenient_v2 / scale≥4 / scale≥3), columns = data slices (all / numeric answers / names /
dates / a couple of random subsamples). Each cell is that judge's pass rate (estimated
accuracy) on that slice, shown as a heatmap. Read it two ways:
- **down a column** — same data, change the judge wording → the number moves (and
  `lenient` vs `lenient_v2` shows even *cosmetic* rewording moves it);
- **across a row** — same judge, change the slice → the number moves again.

**3. "Same answers, different number" callout** — the frozen answers never changed; only the
judge and the slice did. Accuracy is a joint property of *(system × judge × data)*, not the
system alone.

**4. (Optional) Calibrate against humans** — flip on the small hand-labelled correctness
column and show which judge's pass rate tracks the human truth best. A first "judge of
judges", kept light.

### Tab B — Case 2: no ground truth

**1. Describe the task** — drafted support replies, no reference reply, so exact match and
reference metrics are off the table.

**2. Judges disagree** — pick two or three rubric judges; for a chosen reply show each
judge's 1–5 score side by side, and across the pool show the **inter-judge spread**. There
is no gold column to adjudicate — the disagreement is irreducible from the data alone.

**3. Callout** — when the judge *is* the definition of "good", your reported quality is only
as stable as your rubric. Nothing in the dataset can rescue a shaky rubric.

### (Optional) Re-run noise panel

Re-run one judge at `temperature>0` with prompt and data held fixed; the score still
wobbles. The judge is a noisy instrument too. Kept secondary — the prompt × data sensitivity
is the headline of L2.

## What we run inside the lab

1. **The exact-match trap** — run the programmatic check over a QA subsample; watch it mark
   correct answers wrong. Motivation made concrete.
2. **The sensitivity grid (case 1)** — score the same frozen answers across judge prompts ×
   data slices; tabulate / plot the accuracy spread across both axes.
3. **No ground truth (case 2)** — score drafted replies with several rubric judges; plot how
   far apart they land and that nothing in the data can break the tie.
4. **(Optional) Re-run noise** — one judge, `temperature>0`, repeated; plot the per-example
   score spread.

## Two paths, one lab

- **HTML** (`index.html`) — turn the dials (judge × slice) over the pre-computed scores and
  watch the grid light up. No Python required.
- **Jupyter notebook** (`notebook.ipynb`) — the code that produces those scores; students
  re-run it, reword a rubric, add a slice, or point it at their own answers.

## Folder layout

```
playground/judge_calibrator/
  notebook.ipynb           # code path
  index.html               # no-code path
  lab.py                   # judge runners (programmatic + LLM) + loaders
  prompts/
    strict.j2  lenient.j2  lenient_v2.j2  scale_1_5.j2   # case-1 rubrics
    reply_quality.j2                                     # case-2 rubric
    generate_reply.j2                                    # drafts the bitext replies
  answers/
    qa_answers.json        # pre-generated free-text answers + per-judge scores (case 1)
    bitext_replies.json    # pre-generated support replies + per-judge scores (case 2)
  .env.example
```
