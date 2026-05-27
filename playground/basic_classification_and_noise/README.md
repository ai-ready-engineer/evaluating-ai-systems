# basic_classification_and_noise

L1 lab. Two classifiers — a trained TF-IDF model and a gen-AI few-shot LLM — evaluated on three
datasets, with the focus on **how to read a single score honestly**.

**Source of uncertainty in focus:** sampling noise — a score is one draw.

## Two paths

- **`index.html`** — the no-code path. One tab per dataset: describe it, pick a property
  (label/intent), set a test-set size (default 50), **spin the wheel** to draw a sample, and
  read per-property accuracy + confusion matrix + precision/recall for both classifiers. A
  **Beta-distribution tool** turns that one sample into a 95% credible interval and
  `P(accuracy > X)`. Loads precomputed predictions from `predictions/data.js`, so it works
  opened directly (`file://`) — no server, no API calls.
- **`notebook.ipynb`** — the code path. Trains TF-IDF once, resamples the test set to show the
  metric wobble, and compares micro/macro F1 under skew.

## Files

- `lab.py` — dataset loaders, classifiers (TF-IDF+LogReg, few-shot LLM, batched), evaluators
- `build_predictions.py` — trains each classifier once, runs the LLM in ≥20-item batches over a
  held-out 1000-point pool, writes `predictions/<dataset>.json` + bundles `predictions/data.js`
- `prompts/` — `multilabel_classify.j2` (single) and `multilabel_classify_batch.j2` (batched)
- `.env.example` — copy to a `.env` (here or repo root) for live LLM runs

## Datasets

From `../../datasets/` (cache them once with `python ../../datasets/cache_datasets.py`):

- `bitext_customer_support/` — 5 categories (subset of the 11 coarse groupings of the 27 intents), single-label
- `goemotions/` — 28 emotions, multi-label, heavy-tailed
- `synthetic_itsm/` — 10 tags, multi-label, controllable skew (`generate.py`)

## Regenerating predictions

```
pip install scikit-learn datasets pandas python-dotenv jinja2 matplotlib openai
cp .env.example .env          # set LIVE=true, OPENROUTER_API_KEY, OPENROUTER_MODEL
python build_predictions.py   # all three datasets; writes predictions/ + data.js
python build_predictions.py --bundle   # re-bundle data.js from existing JSON only
```

Set `LIVE=false` (or omit the key) and a deterministic mock classifier runs instead, so the
flow still works offline.
