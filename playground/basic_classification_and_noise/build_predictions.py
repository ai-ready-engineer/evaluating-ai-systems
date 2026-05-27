"""Precompute classifier predictions for the L1 no-code (HTML) path.

For each dataset: train TF-IDF once on a disjoint split, run the LLM in batches over a
held-out 1000-point test pool, and write predictions/<dataset>.json (per item: text,
ground truth, both classifiers' predictions). Run once — live — and commit the JSON;
the HTML then browses it offline and only "pretends" to classify.

    python build_predictions.py                 # all datasets
    python build_predictions.py bitext          # one dataset
    python build_predictions.py --limit 40      # tiny pool, for a quick live sanity check

Reads .env (LIVE, OPENROUTER_*) the same way the notebook does.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import lab

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions"
POOL = 1000          # held-out test points per dataset
MIN_TRAIN = 200      # sanity floor for the training split

# Per-dataset classifier config. Bitext is deliberately weakened (few labelled examples + a
# tiny, low-capacity vectorizer) so the trained model is honestly imperfect (~75%) and its
# score carries real uncertainty. GoEmotions / Synthetic stay full-strength.
DATASETS = {
    "bitext":     dict(loader=lambda: lab.load_bitext(n=2000), multilabel=False,
                       train_cap=100,
                       tfidf=dict(max_features=50, ngram_range=(1, 1), min_df=1, C=0.6)),
    "goemotions": dict(loader=lambda: lab.load_goemotions(n=2000), multilabel=True),
    "synthetic":  dict(loader=lab.load_synthetic_itsm,          multilabel=True),
}


def build(name, cfg, pool=POOL, reuse_llm=False):
    df = cfg["loader"]()
    if df is None:
        print(f"{name}: dataset missing — skipping (generate it first)")
        return
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    pool = min(pool, len(df) - MIN_TRAIN)
    test = df.iloc[-pool:].reset_index(drop=True)
    train = df.iloc[:-pool].reset_index(drop=True)
    if cfg.get("train_cap"):
        train = train.iloc[:cfg["train_cap"]].reset_index(drop=True)
    labels = sorted(lab.label_set_of(df))
    multilabel = cfg["multilabel"]

    # Traditional classifier: trained once on the disjoint train split.
    # Single-label datasets use argmax (always one class, never abstains).
    tfidf = lab.TfidfLogRegClassifier(single_label=not multilabel, **cfg.get("tfidf", {})).fit(
        train["text"].tolist(), train["labels"].tolist())
    tfidf_pred = tfidf.predict(test["text"].tolist())

    # LLM few-shot: 3 labelled examples from train, then batched over the test pool.
    examples = list(zip(train["text"].iloc[:3].tolist(), train["labels"].iloc[:3].tolist()))
    if reuse_llm:
        prev = {it["text"]: it["llm"] for it in json.loads((OUT / f"{name}.json").read_text())["items"]}
        llm_pred = [prev.get(t, []) for t in test["text"].tolist()]
        missing = sum(1 for t in test["text"].tolist() if t not in prev)
        print(f"{name}: reusing cached LLM predictions ({missing} missing)")
    else:
        print(f"{name}: training on {len(train)}, classifying {len(test)} with LLM "
              f"(LIVE={lab.os.getenv('LIVE')})...")
        llm_pred = lab.classify_llm_batched(test["text"].tolist(), labels, examples,
                                            multilabel=multilabel, batch_size=20)

    items = [{
        "text":  test["text"].iloc[i],
        "truth": list(test["labels"].iloc[i]),
        "tfidf": list(tfidf_pred[i]),
        "llm":   list(llm_pred[i]),
    } for i in range(len(test))]

    out = {
        "dataset": name,
        "multilabel": multilabel,
        "labels": labels,
        "train_size": len(train),
        "n": len(items),
        "items": items,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(out))
    print(f"{name}: wrote {len(items)} items, {len(labels)} labels, "
          f"multilabel={multilabel} -> {path.relative_to(HERE)}")


def bundle():
    """Bundle the per-dataset JSON into a single data.js the HTML can load over file://."""
    data = {}
    for name in DATASETS:
        p = OUT / f"{name}.json"
        if p.exists():
            data[name] = json.loads(p.read_text())
    (OUT / "data.js").write_text("window.L1_PREDICTIONS = " + json.dumps(data) + ";\n")
    print(f"bundled {list(data)} -> {(OUT / 'data.js').relative_to(HERE)}")


def main(argv):
    if "--bundle" in argv:          # just re-bundle existing JSON, no classification
        bundle()
        return
    limit = POOL
    reuse_llm = "--reuse-llm" in argv      # recompute TF-IDF only, reuse cached LLM predictions
    names = []
    i = 0
    while i < len(argv):
        if argv[i] == "--limit":
            limit = int(argv[i + 1]); i += 2
        elif argv[i] == "--reuse-llm":
            i += 1
        else:
            names.append(argv[i]); i += 1
    names = names or list(DATASETS)
    for name in names:
        build(name, DATASETS[name], pool=limit, reuse_llm=reuse_llm)
    bundle()


if __name__ == "__main__":
    main(sys.argv[1:])
