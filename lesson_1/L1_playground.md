# Playground for lesson 1

> **Scope — single-label only.** L1 stays with **single-label** classification — one answer per
> item, so accuracy is cleanly binary per example. We use both the multi-class case and the binary
> case (binary is just two classes). We *call out* two extensions we deliberately leave for later
> lessons — **multi-label** (many tags per item) and **non-classification** tasks (generation,
> extraction, agents) — but we don't touch them here.

## Assets
### Datasets
Two text-classification datasets — students always work on **subsamples**, never the full set.

**Bitext Customer Support** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace) — real customer-support utterances. The dataset has 27 fine-grained intents grouped into 11 coarse **categories**; we restrict to **5 categories** (account, order, refund, payment, delivery) to keep the confusion matrix readable and the task non-trivial. **Multi-class**: exactly one category per ticket. Pulled via `datasets.load_dataset(...)` (the `category` column).

**Rotten Tomatoes** (`rotten_tomatoes` on HuggingFace) — movie-review sentences labelled POSITIVE or NEGATIVE. **Binary** — the textbook two-class case; one label per review. Binary is just multi-class with two classes, so the same machinery and right/wrong accuracy apply (here a 2×2 confusion matrix).

### Classifiers — two, side by side

- **Regular**: TF-IDF + LogisticRegression. One-vs-rest under the hood, but `predict` commits to a single **argmax** class, so it never abstains and accuracy reconciles with the confusion matrix.
- **LLM (few-shot)**: 2–3 labeled examples per call, prompt as a Jinja2 template in `prompts/`, model & API key from `.env`. The prompt instructs the model to return **exactly one** class from the allowed list. **Predictions pre-computed and committed** so the lab browses without live API calls; a `--live` code path re-runs for those who want to.

In summary, we have two fixed classifiers: the regular one is trained; for the LLM we fix the prompt and constrain it to return one valid class.

### Preparation
We hold out a fixed pool of **1000 test points**. The classifiers are **trained on a disjoint
split** (so accuracy is honest — the model never saw the test points): the 2000 cached Bitext
rows give ~1000 train / 1000 test. The trained model is fit on only a **small capped slice of
the train split** (`TRAIN_CAP`, ~100 examples) on purpose — a fully-fit model scores
near-perfectly and hides the uncertainty we want to feel; a model trained on limited labels is
honestly imperfect (~85%).

For each test point we precompute, with **each** of the two classifiers, the predicted class
plus the ground truth. For the LLM we **batch ≥20 classifications per call** (the texts are
short) so the whole pool is a few dozen calls, not a thousand — and we constrain the model to
return only valid classes.

This is precomputed and stored under `predictions/`, so we don't re-run the classifiers at
runtime — the page just "pretends" to. A `--live` path re-runs for those who want to.

### Metrics
The task has one property — the **category** — taking 5 values, so we get one **5×5 confusion
matrix** and one **multi-class accuracy**. A class selector lets you zoom into any single
category for its own **precision and recall** (treating that class as "this vs. the rest").

"Accuracy" is the fraction of tickets put in the right class (a binomial proportion), which is
exactly what the Beta uncertainty tool below consumes.


## Content flow for the html
This playground shows

1. How we approach evaluation of classification tasks given a classifier: which metrics we use, how do we know when a classifier is "good" or "good enough"
2. Understand the uncertainty around our experiment results, where does it come from, and how to report it

We have one dataset, and our goal is to test two classifiers: a "traditional" classifier and a gen-AI model.

**1. Describe the dataset** — source, what's being classified (one of 5 categories, multi-class).

**2. Key metrics** (both classifiers side by side; a class selector chooses which category to inspect):
- overall accuracy
- the 5×5 confusion matrix
- per-class precision and recall for the selected class of interest

**3. "Spin the wheel" / try your luck** — a single field sets the **test-set size (default
50)**, framed as *"pretend you've crafted a test set of size 50."* Each spin draws a fresh
random sample of that size from the precomputed pool and recomputes the metrics. Spin a few
times → the numbers move. That wobble is the L1 point: a score is one draw.

**4. The one real test set** — in reality you have only *one* test set: you constructed it,
labeled it, and it is of that size. Spin once and treat that draw as your real test set.

**5. Sample uncertainty estimation (Beta tool)** — the proper way to read that one accuracy.
From k correct out of n on the drawn test set, we put a uniform prior and show the posterior
**Beta(k+1, n−k+1)**. The tool reports the **95% credible interval** and **P(accuracy > X)**
for any X the user enters. We don't explain the mechanism here (that's a deep dive) — we just
offer it as a tool.

**6. The jaggedness (the real Beat-3 picture)** — every test case is embedded, projected to 2D,
and colored by whether the selected classifier got it right (per the selected property). Right
and wrong patches **interlock — no clean boundary**, and that jaggedness doesn't vanish with
more data. Two toggles: **feature space** (semantic LSA vs. lexical TF-IDF) and **which
classifier** to color by. The current spin's draw is ringed on top, so you see that one draw
lands on one mix of right/wrong dots → one score, and the next draw lands elsewhere → another.
This is the mechanism *behind* the wobble in (3): coordinates are precomputed offline by
`build_embeddings.py` (numpy-only TF-IDF + LSA + t-SNE) and stored in each item of
`predictions/<dataset>.json`; the notebook reproduces it live with scikit-learn (and a real
sentence-transformer + UMAP if installed).

**7. The competence surface and the dark space** — a canvas below the scatter paints the
classifier's right/wrong as a *continuous surface* over the 2D map (a local kNN of the real
predictions): a green field of competence with red islands of failure. A **Reveal** toggle then
switches the surface off — **"Reality (dark)"** drops to a dark space where only the drawn test
points are lit. The teaching beat: we never actually see the surface; we label only our test set
and infer the rest from a handful of lit spots. Spin to light a different set; raise the test-set
size to light more of the room. Surface is computed in-browser and cached per
view/classifier/property; the deck mirrors this as a two-slide reveal (`07_surface_painted.png`,
`08_surface_dark.png`).






### What we run inside the lab (notebook demos)

1. **Demo A — accuracy hides the per-class story.** Run both classifiers on the fixed pool → overall accuracy + a per-class precision/recall table. The headline number hides where the model actually fails.
2. **Demo B — resampling spread.** Resample the fixed pool many times → plot the spread of accuracy and per-class recall. Rare classes wobble much more than overall accuracy — the L1 sampling-noise punchline.
3. **Demo C — class imbalance (multi-class).** A **skew slider** dials class imbalance up by over-weighting the majority **category** in the draw; a trivial "predict-the-majority-class" baseline climbs on **accuracy** while its **macro-recall** stays pinned near 1/5. Aggregate accuracy flatters it; macro-recall exposes it. A *preview of Lesson 4* — not formalized here.
4. **Demo D — the jaggedness.** Embed the test set, project to 2D, color by right/wrong: right and wrong patches interlock with no clean boundary — the mechanism behind the wobble.

### Two paths, one lab

- **HTML** (`index.html`) — interactive widgets over the pre-computed predictions. No Python required. Includes the spin-the-wheel, the Beta uncertainty tool, the jaggedness scatter/surface, and the class-imbalance demo.
- **Jupyter notebook** (`notebook.ipynb`) — the code that produces the HTML; students can re-run, plug in their own dataset, or swap classifiers.

### Folder layout

```
playground/basic_classification_and_noise/
  notebook.ipynb         # code-savvy path
  index.html             # no-code path
  lab.py                 # shared helpers (loaders, classifiers, multi-class metrics)
  build_predictions.py   # trains + classifies once, caches predictions
  prompts/
    classify.j2          # single-class, one example at a time
    classify_batch.j2    # single-class, batched
  predictions/
    bitext.json          # per-item truth + both classifiers' predicted class
    data.js              # bundle the HTML loads over file://
  .env.example
```


