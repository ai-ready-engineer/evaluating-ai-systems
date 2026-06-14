"""Cache a small Multi-LexSum subset for the L2 'LLM as a judge — scorecard' lab.

Multi-LexSum (Shen et al., 2022; allenai/multi_lexsum) is civil-rights litigation from the
Civil Rights Litigation Clearinghouse: long legal source documents, each with EXPERT-written
summaries at three lengths (long / short / tiny). We use it for the L2 scorecard demo —
summarize a case, then have an LLM judge score the summary (Accuracy / Completeness / Fluency).

`datasets` 4.x dropped script-based loaders, so we read the auto-converted parquet branch
(refs/convert/parquet) directly. We keep only cases whose concatenated source is **under
20,000 words** (so the summarizer sees the WHOLE document — no truncation) and that have an
expert short summary to serve as a reference. Deterministic: sorted by id, first N.

    uv run python build_cache.py            # writes cases.jsonl (default 200 cases)
    uv run python build_cache.py --n 50

License note: source documents are public U.S. court records; the Clearinghouse summaries are
distributed with Multi-LexSum for research use (CC BY-NC 4.0). This subset is cached for
teaching only. See https://huggingface.co/datasets/allenai/multi_lexsum.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases.jsonl"

REPO = "allenai/multi_lexsum"
CONFIG = "v20230518"
WORD_CAP = 20_000
DOC_SEP = "\n\n===== DOCUMENT BREAK =====\n\n"
# parquet shards on refs/convert/parquet (datasets 4.x dropped the script loader)
SHARDS = ["train/0000.parquet", "train/0001.parquet",
          "validation/0000.parquet", "test/0000.parquet"]


def word_count(sources):
    if sources is None:
        return 0
    if isinstance(sources, str):
        sources = [sources]
    return sum(len(s.split()) for s in sources if s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of cases to cache")
    args = ap.parse_args()

    parts = []
    for shard in SHARDS:
        path = hf_hub_download(REPO, f"{CONFIG}/{shard}", repo_type="dataset",
                               revision="refs/convert/parquet")
        parts.append(pd.read_parquet(path))
    df = pd.concat(parts, ignore_index=True)
    df["src_words"] = df["sources"].apply(word_count)

    keep = df[(df["src_words"] > 0) & (df["src_words"] < WORD_CAP) & df["summary/short"].notna()]
    keep = keep.sort_values("id").head(args.n)
    assert len(keep) >= args.n, f"only {len(keep)} qualifying cases (<{WORD_CAP} words + short summary); wanted {args.n}"

    rows = []
    for _, r in keep.iterrows():
        srcs = r["sources"]
        srcs = [srcs] if isinstance(srcs, str) else list(srcs)
        meta = r.get("case_metadata")
        if hasattr(meta, "item"):           # numpy scalar / 0-d
            meta = meta.item()
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except Exception: meta = {"raw": meta}
        elif not isinstance(meta, dict):
            meta = {}
        rows.append({
            "id": str(r["id"]),
            "src_words": int(r["src_words"]),
            "n_docs": len(srcs),
            "case_name": meta.get("case_name") or meta.get("name") or "",
            "court": meta.get("court", ""),
            "filed_date": meta.get("filing_date") or meta.get("filed_date") or "",
            "source": DOC_SEP.join(srcs),
            "summary_long": r["summary/long"],
            "summary_short": r["summary/short"],
            "summary_tiny": r["summary/tiny"] if pd.notna(r["summary/tiny"]) else None,
        })

    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mb = OUT.stat().st_size / 1e6
    ws = [r["src_words"] for r in rows]
    print(f"wrote {len(rows)} cases to {OUT.name} ({mb:.1f} MB)")
    print(f"source words: min {min(ws)}, median {sorted(ws)[len(ws)//2]}, max {max(ws)}")
    print(f"sample: {rows[0]['case_name'] or rows[0]['id']} — {rows[0]['src_words']} words, "
          f"short summary {len(rows[0]['summary_short'].split())} words")


if __name__ == "__main__":
    main()
