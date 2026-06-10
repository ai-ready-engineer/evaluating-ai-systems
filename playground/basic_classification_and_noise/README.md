# basic_classification_and_noise

L1 lab. A trained ML classifier (TF-IDF) evaluated on two **single-label** datasets: Bitext
customer-support tickets (**multi-class**, one of five categories) and Rotten Tomatoes review
sentiment (**binary**, POSITIVE/NEGATIVE). The focus is **how to read a single score honestly**.

The **gen-AI (few-shot LLM) classifier moved to the L2 lab** (`../judge_noise_and_bias/`,
*Judge that classifies* tab); old `#…/genai` links redirect there. Its predictions are still
**built here** (`build_predictions.py` writes both `tfidf` and `llm` into `predictions/data.js`)
and the L2 page loads that bundle.

**Source of uncertainty in focus:** sampling noise — a score is one draw.

Scope is deliberately **single-label** (multi-class + binary). Multi-label classification (many
tags per item) and non-classification tasks are introduced in later lessons.

## Two paths

- **`index.html`** — the no-code path. Pick a **dataset** (Bitext / Rotten Tomatoes), then walk
  the stage tabs: **Explore the data** (browse rows + class balance), **ML classifier**
  (accuracy + per-class precision/recall + confusion matrix, spin-the-wheel wobble + Beta
  interval, the jaggedness scatter + competence surface, and browse-its-errors), **Sampling**
  (the class-imbalance / skew demo), and **Uncertainty (Beta)** (the standalone credible-interval
  calculator). Loads precomputed predictions from `predictions/data.js`, so the tabs work opened
  directly (`file://`). Each view is a **shareable URL** via the hash, e.g. `index.html#bitext/ml`,
  `#bitext/sampling`, `#rotten_tomatoes/imbalance`. The ML tab also has a **"Try it — classify
  your own text"** box that calls the local API (see below); the rest of the page works without it.
- **`notebook.ipynb`** — the code path. Trains TF-IDF once, resamples the test set to show the
  accuracy/per-class-recall wobble, compares a majority baseline vs. the trained model under
  class imbalance (accuracy vs. macro-recall), and (Demo D) builds the jaggedness scatter live
  with scikit-learn (+ a sentence-transformer / UMAP if installed).

## Files

- `lab.py` — Bitext loader, classifiers (TF-IDF+LogReg argmax, few-shot LLM, batched), the
  majority baseline, multi-class evaluators, and skew-weighted sampling
- `build_predictions.py` — trains the classifier once, runs the LLM in ≥20-item batches over a
  held-out 1000-point pool, writes `predictions/bitext.json` + bundles `predictions/data.js`
- `build_embeddings.py` — reproduces the same test pool and writes 2D `lex`/`sem` coords into
  each `predictions/bitext.json` item for the jaggedness scatter (numpy-only TF-IDF + LSA +
  t-SNE). Run after `build_predictions.py`: `python build_embeddings.py`.
- `serve.py` — local FastAPI service for live classification (powers the "Try it" box)
- `prompts/` — `classify.j2` (single item) and `classify_batch.j2` (batched); both ask the model
  for exactly one class
- `.env.example` — copy to a `.env` (here or repo root) for live LLM runs

## Live classification API (`serve.py`)

Powers the **"Try it — classify your own text"** box on the ML tab. Start it from this
folder (it fits the TF-IDF model per dataset on first request; the LLM path runs live with your
`.env` keys, else returns a keyword mock):

```
uv add fastapi uvicorn                          # once
uv run uvicorn serve:app --port 8765 --reload
```

Check it: open `http://localhost:8765/health` → `{"ok":true,"datasets":["bitext","rotten_tomatoes"]}`.
The page (`index.html`) calls `http://localhost:8765` — keep this port, or update the `API`
constant in `index.html` if you change it.

Endpoints: `GET /health`, `GET /datasets`, `POST /classify {dataset, model:"tfidf"|"llm", text}`.

**Port already in use** (`[Errno 48] Address already in use`)? A previous run is still holding the
port. Kill it and restart:

```
lsof -ti:8765 | xargs kill -9      # free the port (use 8000 if that's the stuck one)
pkill -f uvicorn                   # or: kill any stray uvicorn workers
```

## Datasets

From `../../datasets/` (cache them once with `python ../../datasets/cache_datasets.py`):

- `bitext_customer_support/` — 5 categories (subset of the 11 coarse groupings of the 27 intents), multi-class (one category per ticket)
- `rotten_tomatoes/` — movie-review sentiment, binary (POSITIVE / NEGATIVE)

## Regenerating predictions

```
pip install scikit-learn datasets pandas python-dotenv jinja2 matplotlib openai
cp .env.example .env          # set LIVE=true, OPENROUTER_API_KEY, OPENROUTER_MODEL
python build_predictions.py   # trains + classifies bitext; writes predictions/ + data.js
python build_predictions.py --bundle   # re-bundle data.js from existing JSON only
```

Set `LIVE=false` (or omit the key) and a deterministic mock classifier runs instead, so the
flow still works offline.
