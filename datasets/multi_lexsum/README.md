# multi_lexsum — legal-case summaries for the L2 scorecard lab

A small subset of **Multi-LexSum** (Shen et al., 2022) used by the L2 *LLM as a Judge* lab to
demonstrate a **scorecard judge** (slide 11 of `L2_main.pptx`: the "LLM-powered Lawsuit
Summarizer" scored on Accuracy / Completeness / Fluency).

Multi-LexSum is civil-rights litigation from the **Civil Rights Litigation Clearinghouse**:
long legal source documents (filings, opinions, dockets), each with **expert-written summaries**
at three lengths (long / short / tiny). Source: `allenai/multi_lexsum` on HuggingFace.

## What we cache

`build_cache.py` reads the auto-converted parquet branch (`datasets` 4.x dropped the script
loader) across all splits, keeps only cases whose **whole source is under 20,000 words** — so
the summarizer sees the entire document, no truncation — and that have an expert **short**
summary to serve as a reference. Deterministic: sorted by `id`, first N (default 200).

```
uv run python build_cache.py --n 200     # -> cases.jsonl  (gitignored, ~15 MB)
```

`cases.jsonl` (one case per line) holds: `id`, `src_words`, `n_docs`, `case_name`, `court`,
`filed_date`, `source` (concatenated docs), and `summary_long` / `summary_short` / `summary_tiny`.
It is **gitignored** — large and reproducible from HuggingFace. The committed artifact is the
lab's `playground/llm_as_a_judge/lawsuits/scorecards.json` (generated summaries + judge
scorecards), built by `playground/llm_as_a_judge/build_lawsuit_scorecards.py`.

## License / use

Source documents are public U.S. court records; the Clearinghouse summaries are distributed
with Multi-LexSum for research use (CC BY-NC 4.0). Cached here for teaching only. See
https://huggingface.co/datasets/allenai/multi_lexsum and the Multi-LexSum paper (NeurIPS 2022).
