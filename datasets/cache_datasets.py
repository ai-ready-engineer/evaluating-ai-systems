"""Pull subsets of Bitext Customer Support and GoEmotions from HuggingFace and cache
them locally so the L1 lab is fully offline.

Run once on a machine with network access:

    pip install datasets pandas
    cd pub_experiment_design/datasets
    python cache_datasets.py

Produces:
    bitext_customer_support/data.csv
    goemotions/data.csv
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset

HERE = Path(__file__).resolve().parent

# Keep the L1 demo to a handful of distinct categories so the confusion matrix stays
# readable and the task isn't trivially separable. (Full set is 11 categories.)
BITEXT_CATEGORIES = ["ACCOUNT", "ORDER", "REFUND", "PAYMENT", "DELIVERY"]


def cache_bitext(n=2000):
    out = HERE / "bitext_customer_support" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"already cached: {out}")
        return
    print("pulling Bitext Customer Support from HuggingFace...")
    # The HF split is sorted by intent, so shuffle first, then keep only the chosen
    # categories before taking n.
    full = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    ).shuffle(seed=42).filter(lambda r: r["category"] in BITEXT_CATEGORIES)
    ds = full.select(range(min(n, len(full))))
    df = pd.DataFrame({
        "text": ds["instruction"],
        "labels": [c for c in ds["category"]],  # coarse category, restricted to BITEXT_CATEGORIES
    })
    df.to_csv(out, index=False)
    print(f"  wrote {len(df)} rows to {out}")


def cache_goemotions(n=2000):
    out = HERE / "goemotions" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"already cached: {out}")
        return
    print("pulling GoEmotions from HuggingFace...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=f"train[:{n}]")
    names = ds.features["labels"].feature.names
    df = pd.DataFrame({
        "text": ds["text"],
        "labels": [";".join(names[i] for i in lab) for lab in ds["labels"]],
    })
    df.to_csv(out, index=False)
    print(f"  wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    cache_bitext()
    cache_goemotions()
    print()
    print("done — the L1 lab is now offline.")
