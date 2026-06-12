"""Export the trained TF-IDF models to predictions/models.js for in-browser use.

Trains exactly like serve.py (same loaders, shuffle seed, train cap, and per-dataset
config) and dumps, per dataset:

  clf:   classes, vocabulary, idf, per-class logistic-regression coef + intercept,
         ngram_max — everything tryit.js needs to reproduce decision_function()
  place: vocabulary + idf of the full-vocab placement vectorizer fit on the pool
         texts (the same items shown in the scatter), for nearest-example placement

so the "Try it — classify your own text" box runs with no backend. serve.py remains
only as an optional local convenience.

    uv run python build_model_js.py
"""
import json
from pathlib import Path

import numpy as np

import lab

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions" / "models.js"

# Must match serve.py / build_predictions.py so the in-browser model behaves like
# the precomputed predictions.
DATASET_LOADERS = {
    "bitext": lambda: lab.load_bitext(n=2000),
    "rotten_tomatoes": lambda: lab.load_rotten_tomatoes(n=2000),
}
TFIDF_CFG = {
    "bitext": dict(max_features=50, ngram_range=(1, 1), min_df=1, C=0.6),
    "rotten_tomatoes": dict(max_features=1500, ngram_range=(1, 2), min_df=2, C=1.0),
}
TRAIN_CAP = {"bitext": 100, "rotten_tomatoes": 1000}
POOL = 1000

R = 6  # float rounding for a compact file


def _round(a):
    return [round(float(v), R) for v in np.asarray(a).ravel()]


def export_clf(tfidf, cfg):
    vec, mlb, ovr = tfidf.vec, tfidf.mlb, tfidf.clf
    coef = [_round(est.coef_) for est in ovr.estimators_]
    intercept = [round(float(est.intercept_[0]), R) for est in ovr.estimators_]
    n_feat = len(vec.vocabulary_)
    for c in coef:
        assert len(c) == n_feat, "coef width != vocabulary size"
    return {
        "classes": [str(c) for c in mlb.classes_],
        "ngram_max": cfg["ngram_range"][1],
        "vocab": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": _round(vec.idf_),
        "coef": coef,
        "intercept": intercept,
    }


def export_place(name):
    pj = HERE / "predictions" / f"{name}.json"
    items = json.loads(pj.read_text())["items"]
    pool_texts = [it["text"] for it in items if it.get("lex") and it.get("sem")]
    assert pool_texts, f"{name}: no placeable pool items"
    from sklearn.feature_extraction.text import TfidfVectorizer
    place_vec = TfidfVectorizer(min_df=2).fit(pool_texts)
    return {
        "ngram_max": 1,
        "vocab": {t: int(i) for t, i in place_vec.vocabulary_.items()},
        "idf": _round(place_vec.idf_),
    }


def main():
    out = {}
    for name, loader in DATASET_LOADERS.items():
        df = loader().sample(frac=1.0, random_state=0).reset_index(drop=True)
        train = df.iloc[:-POOL] if len(df) > POOL else df
        cap = TRAIN_CAP.get(name)
        if cap:
            train = train.iloc[:cap]
        cfg = TFIDF_CFG[name]
        tfidf = lab.TfidfLogRegClassifier(**cfg).fit(
            train["text"].tolist(), train["labels"].tolist())
        out[name] = {"clf": export_clf(tfidf, cfg), "place": export_place(name)}
        print(f"{name}: clf vocab={len(out[name]['clf']['vocab'])} "
              f"classes={out[name]['clf']['classes']} "
              f"place vocab={len(out[name]['place']['vocab'])}")

    OUT.write_text("window.L1_MODELS = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
