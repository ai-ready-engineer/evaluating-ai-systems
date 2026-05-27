# Datasets

Shared datasets used across playgrounds. Each subfolder is one dataset: the data itself (`data.csv` or similar), a `README.md` describing it, and optionally a generation/loading script.

Datasets are referenced from playgrounds via `../datasets/<name>/` — they live here once, not duplicated.

## Index

| Folder | Source | Labels | Used by |
|---|---|---|---|
| `bitext_customer_support/` | HuggingFace `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | Multi-class intent (27 classes — billing, refund, cancellation, delivery, account, …) | `basic_classification_and_noise/` |
| `goemotions/` | HuggingFace `go_emotions` | Multi-label emotion (27 emotions + neutral, heavy-tailed) | `basic_classification_and_noise/` |
| `synthetic_itsm/` | Generated for the course | Multi-label ITSM tags (billing, urgent, refund, account, complaint, hardware, software, network, security, login) with controllable skew | `basic_classification_and_noise/` |

## Conventions

- One folder per dataset.
- Each folder has a `README.md` that names: source / how to load, license, label structure, size, what skew or quirk it shows.
- For datasets pulled from a public source, the local `data.csv` is a cached subset (small enough to commit); the README documents the exact pull/processing.
- For synthetic datasets, the generation script (`generate.py`) is committed so the dataset is reproducible.
