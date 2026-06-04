"""Helper functions for the L1 lab.

Scope: **multi-class** text classification — each item is assigned exactly one of N
classes. Two classifiers, side by side: TF-IDF + LogReg (argmax over classes) and an
LLM few-shot classifier. One dataset: Bitext customer-support categories.

(Multi-label classification and non-classification tasks are introduced in later lessons;
L1 stays multi-class so "accuracy" is cleanly binary per example.)
"""
import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jinja2

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

random.seed(42)
np.random.seed(42)

# Restrict Bitext to a few distinct categories (full set is 11) — keeps the confusion
# matrix readable and the task non-trivial. Mirror of cache_datasets.BITEXT_CATEGORIES.
BITEXT_CATEGORIES = ["ACCOUNT", "ORDER", "REFUND", "PAYMENT", "DELIVERY"]


# --- Dataset loader ---

def _parse_labels(col):
    """Parse a labels column that might be a list, a stringified list, or a delimited string.

    Multi-class data still rides in a one-element list (e.g. ``["ORDER"]``) so the shape
    matches the rest of the lab; ``label_of`` pulls the single class back out.
    """
    if isinstance(col, list):
        return col
    if isinstance(col, str):
        s = col.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s.replace("'", '"'))
            except Exception:
                pass
        if ";" in s:
            return [t.strip() for t in s.split(";") if t.strip()]
        return [s]
    return []


def load_bitext(n=2000):
    csv_path = DATASETS_DIR / "bitext_customer_support" / "data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        # The HF split is sorted by intent, so train[:n] would only cover the first
        # few intents. Shuffle first so the subset spans all categories.
        full = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split="train",
        ).shuffle(seed=42).filter(lambda r: r["category"] in BITEXT_CATEGORIES)
        ds = full.select(range(min(n, len(full))))
        df = pd.DataFrame({
            "text": ds["instruction"],
            "labels": [c for c in ds["category"]],  # coarse category, restricted to BITEXT_CATEGORIES
        })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    df["labels"] = df["labels"].apply(_parse_labels)
    return df


def load_rotten_tomatoes(n=2000):
    """Binary sentiment: movie-review sentences labelled POSITIVE / NEGATIVE.

    Binary is just the two-class case of multi-class — one label per review, so the same
    machinery and right/wrong accuracy apply.
    """
    csv_path = DATASETS_DIR / "rotten_tomatoes" / "data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        names = {0: "NEGATIVE", 1: "POSITIVE"}
        ds = load_dataset("rotten_tomatoes", split="train").shuffle(seed=42)
        ds = ds.select(range(min(n, len(ds))))
        df = pd.DataFrame({
            "text": ds["text"],
            "labels": [names[l] for l in ds["label"]],
        })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    df["labels"] = df["labels"].apply(_parse_labels)
    return df


def label_set_of(df):
    s = set()
    for labs in df["labels"]:
        s.update(labs)
    return s


def label_of(label_list):
    """The single class from a one-element label list (``['ORDER'] -> 'ORDER'``)."""
    return label_list[0] if label_list else "∅"


# --- Classifiers ---

class TfidfLogRegClassifier:
    """TF-IDF + LogisticRegression, multi-class (argmax).

    Returns exactly one class per item (the highest-scoring class), so it never abstains
    and accuracy reconciles with the confusion matrix. Implemented one-vs-rest under the
    hood, but ``predict`` always commits to a single argmax label.
    """
    def __init__(self, max_features=20000, ngram_range=(1, 2), min_df=2, C=2.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.mlb = MultiLabelBinarizer()
        self.clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=C, class_weight="balanced")
        )

    def fit(self, texts, label_lists):
        X = self.vec.fit_transform(texts)
        Y = self.mlb.fit_transform(label_lists)
        self.clf.fit(X, Y)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        scores = np.asarray(self.clf.decision_function(X))
        classes = self.mlb.classes_
        return [[classes[i]] for i in scores.argmax(axis=1)]


def render_prompt(text, labels, examples):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    tpl = env.get_template("classify.j2")
    return tpl.render(text=text, labels=labels, examples=examples)


def render_batch_prompt(texts, labels, examples):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    tpl = env.get_template("classify_batch.j2")
    return tpl.render(texts=texts, labels=labels, examples=examples)


def _parse_batch_response(text, n, valid):
    """Parse a JSON array of class names into n single-label lists, filtered to valid labels."""
    s = text.strip()
    if s.startswith("```"):                       # strip ``` / ```json fences
        s = s.strip("`")
        s = s[s.find("["):]
    try:
        arr = json.loads(s)
    except Exception:                              # last resort: grab outermost [...]
        start, end = s.find("["), s.rfind("]")
        arr = json.loads(s[start:end + 1]) if 0 <= start < end else []
    out = []
    for i in range(n):
        item = arr[i] if i < len(arr) else None
        if isinstance(item, list):                 # tolerate [["ORDER"], ...]
            item = item[0] if item else None
        out.append([item] if item in valid else [])
    return out


def _call_llm_retry(prompt, attempts=4):
    """call_llm with manual retries on transient errors. Returns None in mock mode or if all
    attempts fail (caller then uses the mock fallback so a network blip can't abort a long run)."""
    for i in range(attempts):
        try:
            return call_llm(prompt)
        except Exception as e:
            if i == attempts - 1:
                print(f"  LLM call failed after {attempts} tries ({e}); mock fallback for this batch")
                return None
            time.sleep(3 * (i + 1))


def classify_llm_batched(texts, labels, examples, batch_size=20):
    """Classify many texts with the LLM, one class each, ≥batch_size per call.

    Mock fallback (deterministic keyword match, else random class) if LIVE != true, so the
    lab can be rebuilt offline.
    """
    labels = sorted(labels)
    valid = set(labels)
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = _call_llm_retry(render_batch_prompt(chunk, labels, examples))
        if resp is None:                           # deterministic mock, per item
            for t in chunk:
                low = t.lower()
                hit = [l for l in labels if l.replace("_", " ").lower() in low]
                preds.append([hit[0] if hit else random.choice(labels)])
        else:
            preds.extend(_parse_batch_response(resp, len(chunk), valid))
    return preds


def call_llm(prompt):
    """Call an LLM (OpenAI or Anthropic). Returns model text, or None if LIVE != true."""
    if os.getenv("LIVE", "false").lower() != "true":
        return None
    if os.getenv("OPENROUTER_API_KEY"):
        from openai import OpenAI  # OpenRouter speaks the OpenAI API
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=90.0,
            max_retries=5,
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(timeout=90.0, max_retries=5)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    return None


class LlmFewShotClassifier:
    """Few-shot LLM multi-class classifier — one class per item. Falls back to a
    deterministic mock when LIVE != true."""
    def __init__(self, label_set, examples):
        self.labels = sorted(label_set)
        self.examples = examples

    def predict(self, texts):
        preds = []
        for t in texts:
            response = call_llm(render_prompt(t, self.labels, self.examples))
            if response is None:
                low = t.lower()                     # mock: keyword match against class names
                hit = [l for l in self.labels if l.replace("_", " ").lower() in low]
                preds.append([hit[0] if hit else random.choice(self.labels)])
            else:
                pred = response.strip().strip('"').strip("'")
                preds.append([pred] if pred in self.labels else [])
        return preds


class MajorityBaseline:
    """Trivial classifier: always predict the single most common class in the training set.

    Used in the imbalance demo — its aggregate accuracy climbs as one class dominates,
    while its macro (per-class average) recall stays near 1/num_classes.
    """
    def __init__(self):
        self.top = None

    def fit(self, _texts, label_lists):
        from collections import Counter
        c = Counter(label_of(labs) for labs in label_lists)
        self.top = c.most_common(1)[0][0] if c else "∅"
        return self

    def predict(self, texts):
        return [[self.top] for _ in texts]


# --- Evaluation (multi-class) ---

def evaluate(y_true, y_pred, label_set):
    """Multi-class metrics: overall accuracy + per-class precision/recall/F1/support,
    plus macro averages. y_true / y_pred are one-element label lists (``['ORDER']``)."""
    classes = sorted(label_set)
    yt = [label_of(x) for x in y_true]
    yp = [label_of(x) for x in y_pred]
    n = len(yt)
    correct = sum(t == p for t, p in zip(yt, yp))
    per_class = {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(yt, yp))
        fp = sum(t != c and p == c for t, p in zip(yt, yp))
        fn = sum(t == c and p != c for t, p in zip(yt, yp))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    def _macro(metric):
        vals = [per_class[c][metric] for c in classes if not np.isnan(per_class[c][metric])]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "accuracy": correct / n if n else float("nan"),
        "n": n,
        "correct": correct,
        "macro_precision": _macro("precision"),
        "macro_recall": _macro("recall"),
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in classes])) if classes else 0.0,
        "per_class": per_class,
    }


# The lab studies *testing*, not training. We fit each classifier ONCE on a fixed
# train split, pre-compute its predictions on a held-out test pool, then resample
# that pool to watch the score wobble. No model is retrained inside any loop.

def make_split(df, train_frac=0.7, seed=0):
    """Shuffle once into a fixed train split and a held-out test pool."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(train_frac * len(shuffled))
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    test = shuffled.iloc[n_train:].reset_index(drop=True)
    return train, test


def prepare(df, train_frac=0.7, seed=0, llm_examples=3):
    """Train both classifiers ONCE and pre-compute their predictions on the test pool.

    Returns the fixed test pool's ground truth and each classifier's predictions.
    Everything downstream just re-scores subsamples of these stored predictions.
    """
    labels = label_set_of(df)
    train, test = make_split(df, train_frac=train_frac, seed=seed)

    tfidf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["labels"].tolist())

    examples = list(zip(
        train["text"].iloc[:llm_examples].tolist(),
        [label_of(l) for l in train["labels"].iloc[:llm_examples].tolist()],
    ))
    llm = LlmFewShotClassifier(labels, examples)

    return {
        "labels": labels,
        "test": test,
        "y_true": test["labels"].tolist(),
        "tfidf_pred": tfidf.predict(test["text"].tolist()),
        "llm_pred": llm.predict(test["text"].tolist()),
    }


def evaluate_resample(y_true, y_pred, label_set, n=None, seed=0, replace=False):
    """Draw one test subsample from pre-computed predictions and score it.

    The L1 mechanic: the model is fixed, so the only thing that changes between draws is
    *which test cases you happened to sample*.
    """
    rng = np.random.RandomState(seed)
    m = len(y_true)
    n = m if n is None else n
    if not replace:
        n = min(n, m)
    idx = rng.choice(m, size=n, replace=replace)
    yt = [y_true[i] for i in idx]
    yp = [y_pred[i] for i in idx]
    return evaluate(yt, yp, label_set)


def skewed_indices(y_true, n, majority, skew=0.0, seed=0, replace=True):
    """Draw a class-imbalanced subsample: items of the ``majority`` class are up-weighted.

    skew=0 → sample roughly as the pool is; skew→1 → the draw is dominated by the majority
    class. Used by the imbalance demo to show aggregate accuracy inflating while macro
    recall (every class equal) collapses. Returns sampled row indices.
    """
    rng = np.random.RandomState(seed)
    yt = [label_of(x) for x in y_true]
    # minority classes down-weighted as skew rises, so the majority share grows smoothly
    min_w = max(1e-3, 1.0 - skew)
    w = np.array([1.0 if c == majority else min_w for c in yt], dtype=float)
    w /= w.sum()
    m = len(yt)
    if not replace:
        n = min(n, m)
    return rng.choice(m, size=n, replace=replace, p=w)
