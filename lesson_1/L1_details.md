# Lesson 1 — The Evolution of "Eval" - from traditional software to ML to AI Agents

Eval in ML is running an *experiment*, not running a *check*.


## Concepts

- Progress means different things in SW 1.0 vs ML — a visible state change vs. a metric you must *establish* moved
- How the meaning of "eval" changes from software 1.0 to ML to AI agents. One word is used to mean many different things
- An eval in ML and AI is an *experiment* to *estimate* a quantity and to identify where we need to improve
- Core concepts: metrics; test sets; the sealed test set
- A score is an *unbiased estimate* of the truth when the test set is fair; sample size shapes trust
- Why "regression testing" doesn't transfer cleanly to ML — shown by example, not yet explained
- List (just list) the sources of uncertainty and error in experiment design — a map for the rest of the course
- Begin the SW 1.0 ↔ ML similarities/differences comparison
- We will begin to list the sources of uncertainty and errors
- we will begin to see calculator to estimate noise and error. we will later understand its principles

[what we leave for session 2: the notion]

> The class will end with a list of reasons why we may be uncertain about our experiment, and a list of traps we need to avoid.

## Patterns

- Read a score as an *estimate*, not a fact — and quote the sample size next to it
- Keep a sealed test set, used only to measure
- Re-run before believing a change

## Anti-patterns — and how they materialize

- **Reading a re-run drop as a regression.** Re-run the same eval; a few checks "fail" anyway — it's noise, not a break. (Don't formalize multiple comparisons yet — just show it.)
- **Quoting a single number as ground truth.** A 78% from 100 cases gets reported as "78%" with no hint it could have been 73% or 83%.

## Understand · report · reduce

The uncertainty this lesson handles: **sampling noise** — a score is an estimate that could have come out differently.

- **Understand it** — a score on a test set is one draw; sample size drives how much it wobbles between runs.
- **Report it** — quote the sample size, and give the score as a plausible range, not a bare point.
- **Reduce it** — use a larger, fairer test set; re-run before believing a change.

## Tools

> Each tool is dual-use — a *playground* to feel the concept, and a *production instrument* you point at your own eval numbers.

- Uncertainty calculator — `accuracy_estimator.html` (counts in → plausible range out; never call it "Beta")
- `sampling_noise.html`
- Multi-label classifier playground — per-label confusion, thresholds `[NEW]`
- Ask the Playground `[NEW]`

## Content notes

- Staircase starts here: SW 1.0 (check) vs ML (experiment). Begin the running comparison table.
- "Regression" = regression *testing*: in SW a regression suite reliably catches a break; in ML a dropped score may be noise. Show by example (re-run, some checks "fail" anyway); don't formalize multiple hypothesis testing yet.
- Sources of uncertainty/error — just list, as a course map: sampling noise, sampling bias, multiple comparisons, variance across domains, choice of metric, drift, model evolution, judge noise, overfitting.




## Playground for lesson 1

Three text-classification datasets, three flavors of "real classification work" — students always work on **subsamples**, never the full set.

**Dataset 1 — Bitext Customer Support** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace) — real customer-support utterances with 27 intent labels (billing, refund, cancellation, delivery, account, …). Multi-class. Pulled via `datasets.load_dataset(...)` in one line.

**Dataset 2 — GoEmotions** (`go_emotions` on HuggingFace) — 58K Reddit comments with 27 emotions + neutral. Truly multi-label, naturally heavy-tailed (joy ~30%, grief ~0.2%). Demonstrates per-label variance and rare-label instability.

**Dataset 3 — Synthetic ITSM tickets** (generated for the course, committed to the repo) — ~1K LLM-generated tickets tagged with `[billing, urgent, refund, account, complaint, hardware, software, network, security, login]`. Multi-label with **controllable skew** — lets us dial the class imbalance for the majority-tag demo.

### Classifiers — two per dataset, side by side

- **Regular**: TF-IDF (or sentence-embeddings) + LogisticRegression in a one-vs-rest multi-label setup.
- **LLM (few-shot)**: 2–3 labeled examples per call, prompt as a Jinja2 template in `prompts/`, model & API key from `.env`. **Predictions pre-computed and committed** so the lab browses without live API calls; a `--live` code path re-runs for those who want to.

### What we run inside the lab

1. Pick a dataset → run both classifiers on a **subsample** → show per-label precision/recall/F1 + micro/macro averages. The headline number hides the rare-label story.
2. **Resample many times** → plot the spread of each metric across subsamples. Rare labels wobble much more than common ones — the L1 sampling-noise punchline.
3. **Skew slider** (synthetic dataset only) → dial label imbalance up; watch a trivial "predict the common tags" baseline catch up on subset-accuracy but collapse on rare-label F1.

### Two paths, one lab

- **HTML** (`index.html`) — interactive widgets over the pre-computed predictions. No Python required.
- **Jupyter notebook** (`notebook.ipynb`) — the code that produces the HTML; students can re-run, plug in their own dataset, or swap classifiers.

### Folder layout

```
playground/lesson_1/
  notebook.ipynb         # code-savvy path
  index.html             # no-code path
  datasets/
    bitext.csv           # cached pull
    goemotions.csv       # cached pull
    synthetic_itsm.csv   # generated & committed
  prompts/
    multilabel_classify.j2
  predictions/
    tfidf_logreg_<dataset>.json
    llm_fewshot_<dataset>.json
  .env.example
```




## To discuss

- Name the false-alarm phenomenon casually ("lucky drops") or leave it nameless until Lesson 4?


# Notes

Scope: classification — including multi-label, where a single instance can carry multiple tags. Customer-support tagging as the running example.

