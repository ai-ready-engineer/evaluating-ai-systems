"""Local FastAPI service for live classification in the L1 playground.

    uv add fastapi uvicorn                          # once
    uv run uvicorn serve:app --port 8765 --reload   # from this folder

Reuses lab.py: a TF-IDF + Logistic Regression model is fit per dataset on first use, and the
few-shot LLM runs live (needs LIVE=true and an API key in .env; otherwise it falls back to the
deterministic keyword mock). Powers the "type -> Classify" box in index.html.

Endpoints:
    GET  /health
    GET  /datasets
    POST /classify   {dataset, model: "tfidf"|"llm", text}  ->  {label, scores}
"""
import json
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import lab

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # pub_experiment_design/ (so ../../style.css resolves)
LAB_URL = "/" + str(HERE.relative_to(REPO)) + "/"

DATASET_LOADERS = {
    "bitext": lambda: lab.load_bitext(n=2000),
    "rotten_tomatoes": lambda: lab.load_rotten_tomatoes(n=2000),
}
# Match build_predictions.py so the live model behaves like the precomputed one.
TFIDF_CFG = {
    "bitext": dict(max_features=50, ngram_range=(1, 1), min_df=1, C=0.6),
    "rotten_tomatoes": dict(max_features=1500, ngram_range=(1, 2), min_df=2, C=1.0),
}
TRAIN_CAP = {"bitext": 100, "rotten_tomatoes": 1000}
POOL = 1000

_MODELS = {}


def _prepare(name):
    df = DATASET_LOADERS[name]()
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    train = df.iloc[:-POOL] if len(df) > POOL else df
    cap = TRAIN_CAP.get(name)
    if cap:
        train = train.iloc[:cap]
    labels = sorted(lab.label_set_of(df))
    tfidf = lab.TfidfLogRegClassifier(**TFIDF_CFG.get(name, {})).fit(
        train["text"].tolist(), train["labels"].tolist())
    examples = list(zip(train["text"].iloc[:3].tolist(),
                        [lab.label_of(l) for l in train["labels"].iloc[:3].tolist()]))

    # Pool used for 2D placement: the precomputed items (same points shown in the scatter),
    # with their lex/sem coords. We place new text at the centroid of its nearest neighbours
    # in TF-IDF space (t-SNE can't embed a brand-new point directly).
    pj = HERE / "predictions" / f"{name}.json"
    pool_texts, pool_lex, pool_sem, place_vec, place_mat = [], None, None, None, None
    if pj.exists():
        items = json.loads(pj.read_text()).get("items", [])
        items = [it for it in items if it.get("lex") and it.get("sem")]
        if items:
            from sklearn.feature_extraction.text import TfidfVectorizer
            pool_texts = [it["text"] for it in items]
            pool_lex = np.array([it["lex"] for it in items], dtype=float)
            pool_sem = np.array([it["sem"] for it in items], dtype=float)
            # A FULL-vocabulary TF-IDF for placement (the classifier's 50-word vocab is too
            # coarse — most typed text wouldn't overlap it). L2-normalized → dot == cosine.
            place_vec = TfidfVectorizer(min_df=2).fit(pool_texts)
            place_mat = place_vec.transform(pool_texts)

    return {"tfidf": tfidf, "labels": labels, "examples": examples,
            "pool_texts": pool_texts, "pool_lex": pool_lex, "pool_sem": pool_sem,
            "place_vec": place_vec, "place_mat": place_mat}


def _position(m, text):
    """Place new text at its single most-similar example (no averaging -> no center bias)."""
    if m["place_mat"] is None:
        return None
    v = m["place_vec"].transform([text])
    sims = np.asarray((v @ m["place_mat"].T).todense())[0]
    order = np.argsort(sims)[::-1]
    top = int(order[0])
    if sims[top] <= 1e-9:                              # no shared words -> can't place honestly
        return None
    return {"lex": m["pool_lex"][top].tolist(), "sem": m["pool_sem"][top].tolist(),
            "sim": float(sims[top]), "nearest": m["pool_texts"][top],
            "neighbors": [m["pool_texts"][int(i)] for i in order[:5]]}


def get_model(name):
    if name not in DATASET_LOADERS:
        return None
    if name not in _MODELS:
        _MODELS[name] = _prepare(name)
    return _MODELS[name]


app = FastAPI(title="L1 classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Always revalidate static assets so regenerated predictions/data.js + page edits show up on
# reload (the page loads data.js via a plain <script src>, which the browser would otherwise cache).
@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


class ClassifyRequest(BaseModel):
    dataset: str
    text: str
    model: str = "tfidf"


@app.get("/health")
def health():
    return {"ok": True, "datasets": list(DATASET_LOADERS)}


@app.get("/datasets")
def datasets():
    return {name: get_model(name)["labels"] for name in DATASET_LOADERS}


@app.post("/classify")
def classify(req: ClassifyRequest):
    m = get_model(req.dataset)
    if m is None:
        return {"error": f"unknown dataset '{req.dataset}'"}
    text = (req.text or "").strip()
    if not text:
        return {"error": "empty text"}

    pos = _position(m, text)

    if req.model == "tfidf":
        clf = m["tfidf"]
        X = clf.vec.transform([text])
        scores = np.asarray(clf.clf.decision_function(X))[0]
        classes = list(clf.mlb.classes_)
        label = classes[int(np.argmax(scores))]
        return {"model": "tfidf", "label": label, "pos": pos,
                "scores": {classes[i]: float(scores[i]) for i in range(len(classes))}}

    if req.model == "llm":
        llm = lab.LlmFewShotClassifier(set(m["labels"]), m["examples"])
        pred = llm.predict([text])[0]
        return {"model": "llm", "label": (pred[0] if pred else "∅"), "pos": pos,
                "live": lab.os.getenv("LIVE", "false").lower() == "true", "scores": {}}

    return {"error": f"unknown model '{req.model}'"}


# Serve the playground page so everything is one URL. We mount the repo ROOT (not just this
# folder) so the page's ../../style.css and ../style.css resolve; "/" redirects to the lab.
#   http://localhost:8765/            -> redirect to the lab page
#   http://localhost:8765/#bitext/ml  -> deep link to a tab (after the redirect)
@app.get("/")
def root():
    return RedirectResponse(url=LAB_URL)

app.mount("/", StaticFiles(directory=REPO, html=True), name="static")
