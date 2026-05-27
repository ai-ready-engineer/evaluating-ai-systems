"""Helper functions for the L1 lab.

Two classifiers: TF-IDF + LogReg, and an LLM few-shot classifier.
Loaders for Bitext, GoEmotions, and synthetic ITSM datasets.
"""
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import jinja2

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

random.seed(42)
np.random.seed(42)


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
        ds = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split=f"train[:{n}]",
        )
        df = pd.DataFrame({
            "text": ds["instruction"],
            "labels": [";".join([i]) for i in ds["intent"]],
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
        ds = load_dataset("go_emotions", "simplified", split=f"train[:{n}]")
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
    """One-vs-rest multi-label classifier: TF-IDF + LogisticRegression."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        self.mlb = MultiLabelBinarizer()
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=2.0))

    def fit(self, texts, label_lists):
        X = self.vec.fit_transform(texts)
        Y = self.mlb.fit_transform(label_lists)
        self.clf.fit(X, Y)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        Y = self.clf.predict(X)
        return [list(p) for p in self.mlb.inverse_transform(Y)]


def render_prompt(text, labels, examples):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)))
    tpl = env.get_template("multilabel_classify.j2")
    return tpl.render(text=text, labels=labels, examples=examples)


def call_llm(prompt):
    """Call an LLM (OpenAI or Anthropic). Returns model text, or None if LIVE != true."""
    if os.getenv("LIVE", "false").lower() != "true":
        return None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
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


def run_one(df, n_sample=300, train_frac=0.7, seed=0, llm_examples=3):
    """Sample, train TF-IDF, run both classifiers, return both classifiers' metrics."""
    sub = df.sample(n=min(n_sample, len(df)), random_state=seed).reset_index(drop=True)
    n_train = int(train_frac * len(sub))
    train = sub.iloc[:n_train]
    test = sub.iloc[n_train:]
    labels = label_set_of(df)

    # TF-IDF + LogReg
    tfidf = TfidfLogRegClassifier().fit(train["text"].tolist(), train["labels"].tolist())
    tfidf_pred = tfidf.predict(test["text"].tolist())

    # LLM few-shot
    examples = list(zip(
        train["text"].iloc[:llm_examples].tolist(),
        train["labels"].iloc[:llm_examples].tolist(),
    ))
    llm = LlmFewShotClassifier(labels, examples)
    llm_pred = llm.predict(test["text"].tolist())

    y_true = test["labels"].tolist()
    return {
        "tfidf": evaluate(y_true, tfidf_pred, labels),
        "llm":   evaluate(y_true, llm_pred,   labels),
        "n_test": len(test),
    }
