# basic_classification_and_noise

L1 lab. Multi-label text classification on three datasets with two classifiers, showing how the score wobbles as the test sample changes.

**Source of uncertainty in focus:** sampling noise — a score is one draw.

## What's inside

- `lab.py` — dataset loaders, classifiers, evaluation helpers
- `notebook.ipynb` — the lab: three demos (per-label F1, resampling spread, skew slider)
- `prompts/multilabel_classify.j2` — Jinja2 prompt for the LLM classifier
- `.env.example` — copy to `.env` and fill in API keys to run the LLM live

## Classifiers

- **TF-IDF + LogReg** (one-vs-rest multi-label) — fast, deterministic baseline
- **LLM few-shot** — prompt with 2–3 labelled examples; model/API key from `.env`. If `LIVE` is not set, a deterministic mock predictor runs so the lab still demonstrates the flow end-to-end.

## Datasets used

From `../../datasets/`:
- `bitext_customer_support/` — real customer-support intents
- `goemotions/` — multi-label emotions, naturally heavy-tailed
- `synthetic_itsm/` — multi-tag ITSM tickets with controllable skew (run `python datasets/synthetic_itsm/generate.py` first)

## Running

```
pip install scikit-learn datasets pandas python-dotenv jinja2
# optional: pip install openai anthropic
cp .env.example .env
jupyter notebook notebook.ipynb
```
