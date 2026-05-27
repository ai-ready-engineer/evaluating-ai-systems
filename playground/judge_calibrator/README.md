# judge_calibrator — "Pick your Judge"

L2 lab. Show how the same AI answer gets scored differently depending on **which judge** you use (strict vs lenient vs 1–5 scale), the **rubric** you write, and even **which re-run** you happen to look at.

**Source of uncertainty in focus:** variance around a score — the judge is an instrument with its own noise.

## What's inside

- `examples.csv` — a small Q&A dataset with the question, an AI-generated answer, and a reference answer. Mix of exact matches, paraphrases, partial answers, and wrong answers.
- `prompts/strict.j2`, `prompts/lenient.j2`, `prompts/scale_1_5.j2` — three judge rubrics. Same task, different temperaments.
- `lab.py` — judge runner with a deterministic mock fallback when `LIVE != true`.
- `notebook.ipynb` — two demos: cross-judge disagreement, and re-run variance for the same judge.

## Running

```
pip install pandas python-dotenv jinja2 matplotlib
# optional, for live LLM calls: pip install openai anthropic
cp .env.example .env
jupyter notebook notebook.ipynb
```

Without API keys the lab uses a deterministic mock scoring function so Demo A still illustrates the cross-judge gap. Demo B (re-run variance) requires a live LLM with `temperature>0`.
