# Lesson 7 — Additional sources of errors and uncertainty — and why they compound *(provisional)*

**Thesis:** an eval isn't run once — it has to keep meaning something as the system and the world move.

## Concepts

- Drift & model evolution — yesterday's eval doesn't carry forward; the target moves
- LLM-as-judge when there's no ground truth — judge noise and how uncertainty compounds
- **Uncertainties compound** — the sources from L1-L5 don't act one at a time; sampling noise, bias, overfitting, drift and judge noise stack, and the real uncertainty in a number is their *combination*, not the largest single one
- Design → run → report an eval honestly; separate noise from bias; the eval-driven improvement loop

## Patterns

- Re-validate evals on a schedule; re-baseline after a model change
- Calibrate the judge against human labels before trusting it
- Report design and method honestly, not just the headline number

## Anti-patterns — and how they materialize

- **Trusting a stale eval.** Last quarter's number is quoted after a model update; the target moved and nobody re-ran it.
- **Uncalibrated judge.** An LLM judge scores the same output 3 then 5 on re-run; the noise compounds into the final metric.
- **Headline-only reporting.** A single number ships with no sample size, no range, no method — and downstream nobody can question it.

## Understand · report · reduce

The uncertainty this lesson handles: **drift & compounding** — evals age, and the separate sources from L1-L5 stack.

- **Understand it** — as the model and the world move, yesterday's eval stops meaning what it did; and the earlier sources combine rather than acting alone.
- **Report it** — timestamp and re-baseline evals; report the *combined* uncertainty, not just the headline number.
- **Reduce it** — re-validate on a schedule; re-baseline after model changes; calibrate judges against humans.

## Tools

> Dual-use — playgrounds to feel drift and judge noise, and production instruments for monitoring and reporting your own evals.

- `temporal_drift.html`
- `judges_and_compounding.html`
- Experiment-Report Evaluator `[NEW]`
- Recap of earlier playgrounds

## Content notes

- Compounding is the closing message of the course: each earlier lesson isolated one source so you could feel it; L6 puts them back together. A number that looks fine on any single axis can still be badly off once the sources stack.
- Provisional — to refine once Lessons 1–2 are locked.
