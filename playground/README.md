# Playgrounds

Each subfolder is a self-contained playground — a hands-on tool that demonstrates one source of uncertainty or one practical move from the course. Each is named for what it shows, not for which lesson references it, so the same playground can be linked from multiple lessons.

A shared `style.css` lives in this folder for common look-and-feel; individual playgrounds may add their own CSS/JS.

## Index

| Folder | What it shows | Referenced by |
|---|---|---|
| `accuracy_estimator/` | Counts in → plausible range out (the "uncertainty calculator") | L1, L3 |
| `sampling_noise/` | How a score wobbles as the sample changes | L1 |
| `basic_classification_and_noise/` | Multi-label classification lab — real & synthetic datasets, TF-IDF+LogReg vs LLM few-shot, subsampling spread, skew slider | L1 |
| `distribution_explorer/` | The shape of a random variable — mean, median, variance | L3 |
| `bootstrap_lab/` | Bootstrap to estimate any metric's spread | (deep dive) |
| `convolution_of_independent_random_variables/` | Averages settle as samples grow (CLT, felt) | (deep dive) |
| `multiple_hypothesis_testing/` | Track many metrics or re-test often → some look good by chance ("lucky-winner finder") | L4 |
| `sampling_bias/` | Unrepresentative test sets and the gap they hide | L6 |
| `domain_variance/` | Variability across customers/domains/time | L3 |
| `metric_choice/` | The same data, different metrics, different verdicts | L4 |
| `judges_and_compounding/` | Judge noise + how uncertainty sources stack ("compounding visualizer") | L2, L6, L7 |
| `temporal_drift/` | Evals age — yesterday's number stops meaning what it did | L7 |
| `llm_evolution/` | How model upgrades shift eval results | L7 |

## Conventions

- Each playground folder contains at least an `index.html`. Notebook-style playgrounds also include a `notebook.ipynb`.
- Datasets are referenced from `../datasets/<name>/` — never duplicated inside a playground.
- Prompts (for LLM-driven playgrounds) live in a local `prompts/` subfolder as Jinja2 templates.
- LLM config (model name, API key) comes from `.env`; predictions are pre-computed and committed where practical, so browsing the playground doesn't require live API calls.
