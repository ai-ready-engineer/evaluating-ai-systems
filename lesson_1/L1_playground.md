# Playground for lesson 1

## Assets 
### Datasets
Three text-classification datasets, three flavors of "real classification work" — students always work on **subsamples**, never the full set.

**Dataset 1 — Bitext Customer Support** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace) — real customer-support utterances. The dataset has 27 fine-grained intents grouped into 11 coarse **categories**; we restrict to **5 categories** (account, order, refund, payment, delivery) to keep the confusion matrix readable and the task non-trivial. Multi-class. Pulled via `datasets.load_dataset(...)` in one line (the `category` column).

**Dataset 2 — GoEmotions** (`go_emotions` on HuggingFace) — 58K Reddit comments with 27 emotions + neutral. Truly multi-label, naturally heavy-tailed (joy ~30%, grief ~0.2%). Demonstrates per-label variance and rare-label instability.

**Dataset 3 — Synthetic ITSM tickets** (generated for the course, committed to the repo) — ~1K LLM-generated tickets tagged with `[billing, urgent, refund, account, complaint, hardware, software, network, security, login]`. Multi-label with **controllable skew** — lets us dial the class imbalance for the majority-tag demo.

### Classifiers — two per dataset, side by side

- **Regular**: TF-IDF (or sentence-embeddings) + LogisticRegression in a one-vs-rest multi-label setup.
- **LLM (few-shot)**: 2–3 labeled examples per call, prompt as a Jinja2 template in `prompts/`, model & API key from `.env`. **Predictions pre-computed and committed** so the lab browses without live API calls; a `--live` code path re-runs for those who want to.


in summary, we have a fixed classifier for each dataset: for regular, they are trained. for LLM, we fix the prompt for each dataset and ensure only proper labels are returned. 

### Preparation
For each dataset we hold out a fixed pool of **1000 test points**. The classifiers are
**trained on a disjoint split** (so accuracy is honest — the model never saw the test
points): Bitext and GoEmotions have 2000 cached rows, so we train on ~1000 and test on the
other 1000; the synthetic ITSM set is regenerated at ~2000 rows (default skew) so it also
leaves 1000 held out. The trained model is fit on only a **small capped slice of the train
split** (`TRAIN_CAP`, ~100 examples) on purpose — a fully-fit model scores near-perfectly and
hides the uncertainty we want to feel; a model trained on limited labels is honestly imperfect.

For each test point we precompute, with **each** of the two classifiers, the predicted
label(s) plus the ground truth. For the LLM we **batch ≥20 classifications per call** (keep
the dataset texts short) so the whole pool is a few dozen calls, not a thousand — and we
constrain the model to return only valid labels.

This is precomputed and stored under `predictions/`, so we don't re-run the classifiers at
runtime — the page just "pretends" to. A `--live` path re-runs for those who want to.

### Metrics are per property
A **property** is a single thing we classify, and every metric is computed **per property,
not per dataset**:

- **Bitext** has one property — the **category** — taking 5 values → one **5×5 confusion
  matrix** and one (multi-class) accuracy.
- **GoEmotions / Synthetic** are multi-label, so **each label is its own yes/no property** →
  each gets its **own 2×2 confusion matrix, accuracy, and precision/recall**. A property
  selector chooses which one to inspect.

"Accuracy" is therefore the per-property fraction correct (a binomial proportion), which is
exactly what the Beta uncertainty tool below consumes.


## Content flow for the html
This playground shows

1. How we approach evaluation of classification tasks given a classifier: which metrics we use, how do we know when a classifier is "good" or "good enough"
2. Understand the uncertainty around our experiment results, where does it come from, and how to report it

We have three datasets, and in our experiments our goal is to test two classifier. one is a "traditional" classifier and the second is a gen AI model

Each dataset is a tab. Inside a tab:

**1. Describe the dataset** — source, what's being classified, single-label vs. multi-label.

**2. Key metrics, per property** (both classifiers side by side; a property selector chooses
which label/intent to inspect):
- accuracy (of the selected property)
- confusion matrix (5×5 for Bitext's category; 2×2 for a multi-label property)
- per-class precision and recall
- if there is a class of interest, a per-class recall

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


