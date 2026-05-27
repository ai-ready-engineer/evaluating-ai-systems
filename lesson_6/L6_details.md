# Lesson 5 — Measuring the wrong thing (bias) *(provisional)*

**Thesis:** some error is systematic — more data won't fix it.

## Concepts

- Biased vs unbiased estimates — callback to L1; noise shrinks with data, bias does not
- Sampling bias — an unrepresentative test set
- Aggregation traps (Simpson's Paradox) — an aggregate that reverses the per-domain truth (variance across domains itself introduced in L2)
- Choice of metric — measuring something that isn't what you care about

## Patterns

- Audit the test set for representativeness against real traffic
- Break results down by domain before trusting the aggregate
- Choose the metric *before* seeing results, tied to what you actually care about

## Anti-patterns — and how they materialize

- **Unrepresentative test set.** Eval cases all from one source score 90%; production traffic, broader, runs 65%.
- **Aggregation trap.** System B beats A overall, yet A wins in every single domain — Simpson's Paradox in a dashboard.
- **Metric picked to flatter.** The metric is chosen after the fact because it's the one that looks good.

## Understand · report · reduce

The uncertainty this lesson handles: **bias** — systematic error that comes from the test set or the metric, not the sample.

- **Understand it** — bias is built into how the test set was assembled or the metric was chosen; more data won't remove it.
- **Report it** — report results broken down by domain, and state how the test set was built and the metric chosen.
- **Reduce it** — audit the test set against real traffic; choose the metric before seeing results.

## Tools

> Dual-use — a playground to see bias appear, and a production instrument for auditing your own test set and breakdowns.

- `sampling_bias.html`
- `metric_choice.html`

## Content notes

- Provisional — to refine once Lessons 1–2 are locked.
