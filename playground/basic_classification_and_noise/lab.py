"""Helper functions for the L1 lab.

Two classifiers: TF-IDF + LogReg, and an LLM few-shot classifier.
Loaders for Bitext, GoEmotions, and synthetic ITSM datasets.
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


# --- Dataset loaders ---

def _parse_labels(col):
    """Parse a labels column that might be a list, a stringified list, or a delimited string."""
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
        # few intents. Shuffle first so the subset spans all 27 intents.
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


def load_goemotions(n=2000):
    csv_path = DATASETS_DIR / "goemotions" / "data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=f"train[:{n}]")
        names = ds.features["labels"].feature.names
        df = pd.DataFrame({
            "text": ds["text"],
            "labels": [";".join(names[i] for i in lab) for lab in ds["labels"]],
        })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    df["labels"] = df["labels"].apply(_parse_labels)
    return df


def load_synthetic_itsm():
    csv_path = DATASETS_DIR / "synthetic_itsm" / "data.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["labels"] = df["labels"].apply(_parse_labels)
    return df


def label_set_of(df):
    s = set()
    for labs in df["labels"]:
        s.update(labs)
    return s


# --- Classifiers ---

class TfidfLogRegClassifier:
    """TF-IDF + one-vs-rest LogisticRegression.

    single_label=True → multi-class mode: always return exactly one label, the highest-scoring
    class (argmax). Never abstains, so accuracy and the confusion matrix reconcile.
    single_label=False → multi-label mode: threshold per label (may return zero or several).
    """
    def __init__(self, max_features=20000, ngram_range=(1, 2), min_df=2, C=2.0, single_label=False):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        self.single_label = single_label
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
        if self.single_label:
            scores = np.asarray(self.clf.decision_function(X))
            classes = self.mlb.classes_
            return [[classes[i]] for i in scores.argmax(axis=1)]
        Y = self.clf.predict(X)
        return [list(p) for p in self.mlb.inverse_transform(Y)]


def render_prompt(text, labels, examples):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    tpl = env.get_template("multilabel_classify.j2")
    return tpl.render(text=text, labels=labels, examples=examples)


def render_batch_prompt(texts, labels, examples, multilabel):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    tpl = env.get_template("multilabel_classify_batch.j2")
    return tpl.render(texts=texts, labels=labels, examples=examples, multilabel=multilabel)


def _parse_batch_response(text, n, valid):
    """Parse a JSON array-of-arrays into n label-lists, filtered to valid labels."""
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
        item = arr[i] if i < len(arr) else []
        if isinstance(item, str):
            item = [item]
        if not isinstance(item, list):
            item = []
        out.append([l for l in item if l in valid])
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


def classify_llm_batched(texts, labels, examples, multilabel=True, batch_size=20):
    """Classify many texts with the LLM, ≥batch_size per call. Mock fallback if LIVE != true."""
    labels = sorted(labels)
    valid = set(labels)
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = _call_llm_retry(render_batch_prompt(chunk, labels, examples, multilabel))
        if resp is None:                           # deterministic mock, per item
            chunk_preds = []
            for t in chunk:
                low = t.lower()
                p = [l for l in labels if l.replace("_", " ").lower() in low]
                if not p:
                    p = [random.choice(labels)]
                chunk_preds.append(p if multilabel else p[:1])
            preds.extend(chunk_preds)
        else:
            parsed = _parse_batch_response(resp, len(chunk), valid)
            if not multilabel:
                parsed = [(p[:1] if p else []) for p in parsed]
            preds.extend(parsed)
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
    """Few-shot LLM multi-label classifier. Falls back to a deterministic mock when LIVE != true."""
    def __init__(self, label_set, examples):
        self.labels = sorted(label_set)
        self.examples = examples

    def predict(self, texts):
        preds = []
        for t in texts:
            prompt = render_prompt(t, self.labels, self.examples)
            response = call_llm(prompt)
            if response is None:
                # Mock fallback: keyword match against label names
                low = t.lower()
                pred = [l for l in self.labels if l.replace("_", " ").lower() in low]
                if not pred:
                    pred = [random.choice(self.labels)]
                preds.append(pred)
            else:
                try:
                    pred = json.loads(response.strip())
                    if not isinstance(pred, list):
                        pred = []
                except Exception:
                    pred = []
                preds.append([p for p in pred if p in self.labels])
        return preds


# --- Evaluation ---

def evaluate(y_true, y_pred, label_set):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score
    classes = sorted(label_set)
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    Y_true = mlb.transform(y_true)
    Y_pred = mlb.transform(y_pred)
    per_label = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "per_label": dict(zip(classes, [float(x) for x in per_label])),
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
    Demos A and B both reuse this — one training, one prediction pass; everything
    downstream just re-scores subsamples of these stored predictions.
    """
    labels = label_set_of(df)
    train, test = make_split(df, train_frac=train_frac, seed=seed)

    tfidf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["labels"].tolist())

    examples = list(zip(
        train["text"].iloc[:llm_examples].tolist(),
        train["labels"].iloc[:llm_examples].tolist(),
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

    The L1 mechanic: the model is fixed, so the only thing that changes between
    draws is *which test cases you happened to sample*. Defaults to a subsample
    without replacement; set replace=True for a bootstrap draw.
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
