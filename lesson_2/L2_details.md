## Playground for lesson 2

Same shape as L1, different axis. L1 froze the model and resampled the **data** to watch
the score wobble (sampling noise). Here we freeze the AI's answers and vary the **judge
prompt** and the **data** — and watch the *accuracy* move. The thing the student should
walk away feeling: a reported accuracy is not a fact about the system; it is **sensitive**
to how the judge is worded and to which data you happened to score, choices that look
incidental but aren't.

### The motivating concept — you need a judge even when you have ground truth

Open before any talk of rubrics. Take a dataset where every question has a gold answer and
the AI answers in free text, and run the obvious programmatic check: exact string match
against the gold. It fails constantly — "Paris, the City of Light", "approximately
300,000 km/s", "It stands for Central Processing Unit" are all correct yet none match the
reference. So even *with* ground truth, naive comparison badly undercounts. That is the
hook: you need a **judge** — a comparator that decides "does this answer *mean* the same as
the reference?" — and the moment you introduce one, you've introduced something with its
own behaviour to worry about.

### Case 1 vs case 2 — two reasons you reach for a judge

- **Case 1 — ground truth exists, but it's hard to compare.** The judge is a *smarter
  comparator* against a known reference. It approximates an answer we actually have.
- **Case 2 — no ground truth at all.** The judge *is* the definition of "good"; there's
  nothing to compare against. Reuse L1's already-cached `bitext_customer_support` messages
  and have the AI *draft a support reply*. There is no gold reply, so exact match isn't even
  an option — a rubric judge is the only scorer there is.

The split matters for what variability *means*: in case 1 we can ask "how often does the
judge agree with the gold?"; in case 2 there is no gold, so all we can observe is judges
disagreeing with each other — and that disagreement is the entire signal.

Kept deliberately small: **one new dataset** (the QA set) plus the bitext set already
cached for L1. No more.

### Datasets

- **QA set (case 1)** — a small factual short-answer dataset: question, gold answer, and a
  pre-generated free-text AI answer. Default a slice of TriviaQA (swappable for SQuAD /
  NQ-open); cached from HuggingFace into `datasets/`. The free-text answers are the point —
  paraphrases, added units, and extra context are what make exact match fail.
- **`bitext_customer_support` (case 2)** — reused from L1. The AI drafts a reply to each
  customer message; the replies are pre-generated and committed. No reference reply exists.

### What the student runs

1. **The exact-match trap.** Run the programmatic exact-match check over a QA subsample and
   watch it mark plainly-correct answers wrong. This is the motivation made concrete —
   ground truth is present, yet the naive comparator is already lying to you.
2. **Case 1 — the sensitivity grid.** Score the *same frozen answers* with several judge
   prompts (strict / lenient / a reworded-but-equivalent rubric / a 1–5 scale thresholded at
   different cut points) across several data slices. Lay it out as a grid —
   **judge prompt × data slice → accuracy** — and read it across both axes: change the
   wording and the number moves; change the slice and it moves again. Same system, same
   answers, a spread of headline accuracies.
3. **Case 2 — no ground truth.** Score the drafted bitext replies with a couple of rubric
   judges. There's no accuracy to report, only the judges' scores — show how far apart they
   land on the same reply, and that nothing in the data can adjudicate between them.
4. **(Optional) Re-run noise.** Re-run one judge at `temperature>0` with prompt and data
   fixed; the score still wobbles. The judge is a noisy instrument too — kept secondary, as
   the prompt × data sensitivity is the headline.

### Components

- **Programmatic baseline judge** — exact match / regex / schema check. Works on the easy
  case-1 rows, fails on paraphrase, and *cannot run at all* on case 2. Mirrors L1's non-AI
  baseline sitting beside the LLM.
- **LLM judges** — the rubric prompts (strict / lenient / scale, plus reworded variants for
  case 1, and a quality rubric for case 2), as Jinja2 templates in `prompts/`.

### Two paths, one lab

- **HTML** (`index.html`) — turn the dials over the pre-computed scores: pick a judge, pick
  a slice, watch the grid light up. No Python required.
- **Jupyter notebook** (`notebook.ipynb`) — the code that produces those scores; students
  re-run it, reword a rubric, or point it at their own answers.

### Conventions

- New QA dataset under `datasets/`, cached from HuggingFace and registered in
  `cache_datasets.py` + the datasets README; `bitext_customer_support` reused as-is.
- AI answers/replies **pre-computed and committed** (an `answers/` dir, like L1's
  `predictions/`) so the lab runs fully offline; `LIVE=true` re-generates them live.
- Judge prompts are Jinja2 templates in `prompts/`; model + API key from `.env`;
  deterministic mock fallback when `LIVE != true`.

### Folder layout

```
playground/judge_calibrator/
  notebook.ipynb           # code path
  index.html               # no-code path
  lab.py                   # judge runners + loaders
  prompts/
    strict.j2  lenient.j2  scale_1_5.j2   # case-1 rubrics (+ reworded variants)
    reply_quality.j2                      # case-2 rubric
    generate_reply.j2                     # drafts the bitext replies
  answers/
    qa_answers.json        # pre-generated free-text answers (case 1)
    bitext_replies.json    # pre-generated support replies (case 2)
  .env.example
```

### Skills exercised

- Choose a scoring approach that fits the task — programmatic vs reference-based vs LLM-judge
- Write a rubric an LLM-judge can apply consistently — and see how fragile small wording
  changes make the number
- Spot when variability is coming from the judge/prompt, not from the data
- Pick the right score type (binary / ordinal / continuous) for the output
