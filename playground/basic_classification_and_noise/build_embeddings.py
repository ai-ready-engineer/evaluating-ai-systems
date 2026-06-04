"""Precompute 2D coordinates for the L1 "jaggedness" scatter — offline, numpy-only.

For each dataset we reproduce the *exact* held-out test pool that build_predictions.py
scored (same loader, same `df.sample(frac=1, random_state=0)`, same last-`POOL` slice), so the
coordinates line up one-to-one with predictions/<dataset>.json items by index.

We then lay each test point out in 2D twice, in two genuinely different feature spaces:

  * "lex"  — TF-IDF bag-of-words. The lexical space the trained TF-IDF+LogReg classifier
             actually decides in. Surface form: same words → close.
  * "sem"  — LSA embeddings (truncated SVD of the TF-IDF matrix). A classic *semantic*
             embedding: synonyms and co-occurring terms collapse onto shared latent
             dimensions, so meaning (not just wording) drives nearness.

Both high-dim spaces are flattened to 2D with a small from-scratch t-SNE (numpy only — the
sandbox can't pip-install scikit-learn). The result is written back into each item of
predictions/<dataset>.json as `lex: [x, y]` and `sem: [x, y]`, then data.js is re-bundled.

    python build_embeddings.py                # bitext
    python build_embeddings.py bitext         # explicit

The notebook does the same thing with the "real" stack (scikit-learn TF-IDF/SVD, optional
sentence-transformers + UMAP) — this script is the offline path that keeps index.html
dependency-free.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions"

POOL = 1000          # mirror build_predictions.POOL
MIN_TRAIN = 200      # mirror build_predictions.MIN_TRAIN
SEED = 0

DATASETS = {
    "bitext": lambda: lab.load_bitext(n=2000),
    "rotten_tomatoes": lambda: lab.load_rotten_tomatoes(n=2000),
}

_TOKEN = re.compile(r"[a-z0-9]+")


def test_pool(loader):
    """Reproduce build_predictions.py's held-out test pool, in order."""
    df = loader()
    if df is None:
        return None
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pool = min(POOL, len(df) - MIN_TRAIN)
    return df.iloc[-pool:].reset_index(drop=True)


# --- TF-IDF (numpy) ---

def tokenize(text):
    # mask the dataset's {{Placeholder}} spans so they don't dominate the vocabulary
    text = re.sub(r"\{\{.*?\}\}", " ", str(text).lower())
    return _TOKEN.findall(text)


def tfidf_matrix(texts, max_features=3000, min_df=2):
    docs = [tokenize(t) for t in texts]
    df_count = {}
    for toks in docs:
        for w in set(toks):
            df_count[w] = df_count.get(w, 0) + 1
    vocab = [w for w, c in df_count.items() if c >= min_df]
    vocab.sort(key=lambda w: (-df_count[w], w))
    vocab = vocab[:max_features]
    idx = {w: i for i, w in enumerate(vocab)}
    n, v = len(docs), len(vocab)
    X = np.zeros((n, v), dtype=np.float32)
    for r, toks in enumerate(docs):
        for w in toks:
            j = idx.get(w)
            if j is not None:
                X[r, j] += 1.0
    # tf-idf with smoothed idf, then L2-normalize rows
    idf = np.log((1.0 + n) / (1.0 + np.array([df_count[w] for w in vocab]))) + 1.0
    X *= idf[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def lsa_embeddings(X, k=64):
    """Latent Semantic Analysis: top-k right singular space of the TF-IDF matrix."""
    k = min(k, min(X.shape) - 1)
    # economy SVD; X is (n, v). U S Vt. Document embeddings = U * S (= X @ V).
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    emb = U[:, :k] * S[:k][None, :]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


# --- from-scratch t-SNE (numpy) ---

def _pairwise_sq_dists(X):
    sq = np.sum(X * X, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return D


def _p_joint(D, perplexity=30.0, tol=1e-5, steps=50):
    n = D.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    target = np.log(perplexity)
    for i in range(n):
        betamin, betamax, beta = -np.inf, np.inf, 1.0
        Di = np.delete(D[i], i)
        for _ in range(steps):
            Pi = np.exp(-Di * beta)
            sumPi = Pi.sum()
            if sumPi == 0:
                sumPi = 1e-12
            H = np.log(sumPi) + beta * np.sum(Di * Pi) / sumPi
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
        Pi = np.exp(-Di * beta)
        Pi /= max(Pi.sum(), 1e-12)
        P[i] = np.insert(Pi, i, 0.0)
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def tsne(X, perplexity=30.0, n_iter=500, seed=0):
    """Compact Barnes-Hut-free t-SNE. Fine for ~1k points."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    P = _p_joint(_pairwise_sq_dists(X), perplexity)
    P *= 4.0  # early exaggeration
    Y = rng.normal(0, 1e-4, size=(n, 2))
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for it in range(n_iter):
        D = _pairwise_sq_dists(Y)
        num = 1.0 / (1.0 + D)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        # vectorized gradient: dY_i = 4 * [ (sum_j L_ij) Y_i - sum_j L_ij Y_j ], L = (P-Q)*num
        L = (P - Q) * num
        dY = 4.0 * (L.sum(axis=1)[:, None] * Y - L @ Y)
        momentum = 0.5 if it < 250 else 0.8
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01
        iY = momentum * iY - 200.0 * gains * dY
        Y += iY
        Y -= Y.mean(axis=0)
        if it == 100:
            P /= 4.0  # stop early exaggeration
    return Y


def to_unit_box(Y):
    """Scale coords into [0,1]^2 with a small margin, rounded for compact JSON."""
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    U = (Y - lo) / span
    U = 0.04 + 0.92 * U
    return np.round(U, 4)


def build(name, loader):
    test = test_pool(loader)
    if test is None:
        print(f"{name}: dataset missing — skipping")
        return
    pred_path = OUT / f"{name}.json"
    data = json.loads(pred_path.read_text())
    items = data["items"]
    assert len(items) == len(test), f"{name}: {len(items)} preds vs {len(test)} test rows"
    mism = sum(1 for i in range(len(items)) if items[i]["text"] != str(test["text"].iloc[i]))
    assert mism == 0, f"{name}: {mism} text mismatches — split drifted"

    texts = test["text"].tolist()
    print(f"{name}: TF-IDF over {len(texts)} texts ...")
    X = tfidf_matrix(texts)
    print(f"{name}: lexical t-SNE ({X.shape[1]}-dim TF-IDF) ...")
    lex = to_unit_box(tsne(X, seed=SEED))
    print(f"{name}: LSA + semantic t-SNE ...")
    sem = to_unit_box(tsne(lsa_embeddings(X), seed=SEED))

    for i, it in enumerate(items):
        it["lex"] = [float(lex[i, 0]), float(lex[i, 1])]
        it["sem"] = [float(sem[i, 0]), float(sem[i, 1])]
    data["has_coords"] = True
    pred_path.write_text(json.dumps(data))
    print(f"{name}: wrote coords into {pred_path.relative_to(HERE)}")


def bundle():
    data = {}
    for name in DATASETS:
        p = OUT / f"{name}.json"
        if p.exists():
            data[name] = json.loads(p.read_text())
    (OUT / "data.js").write_text("window.L1_PREDICTIONS = " + json.dumps(data) + ";\n")
    print(f"bundled {list(data)} -> {(OUT / 'data.js').relative_to(HERE)}")


def main(argv):
    names = [a for a in argv if not a.startswith("-")] or list(DATASETS)
    for name in names:
        build(name, DATASETS[name])
    bundle()


if __name__ == "__main__":
    main(sys.argv[1:])
