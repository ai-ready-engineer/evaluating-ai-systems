# Lesson 8 — Designing Reliable Experiments

**Number:** L8
**Tagline:** Run the whole eval: from product question to ship / do-not-ship decision.
**Source of uncertainty:** Combined uncertainty across the full experiment design

## What we cover

- Start with the decision: what would change if the experiment came back positive?
- Turn a product claim into an experiment plan: baseline, candidate, unit of analysis, population, metrics, and guardrails
- Build the evaluation set: representative strata, dev / validation / sealed test split, and a small human-labeled calibration set
- Choose scoring instruments: programmatic checks where possible; calibrated LLM judge where judgment is required
- Run the comparison: point estimate, range, per-domain breakdowns, judge reliability, and multiple-metric discipline
- Write the decision report: what we know, what we do not know, what can ship, what needs another iteration, and when to re-baseline

## Skills you unlock

- Translate a vague quality goal into a measurable experiment without losing the product decision
- Design an eval plan that includes data, metrics, judges, holdouts, acceptance criteria, and reporting before anyone sees results
- Read an end-to-end result as evidence, not as a verdict handed down by one number
- Spot the hidden blocker in a positive aggregate result: weak strata, judge disagreement, safety guardrails, drift, or overfitting
- Produce a final eval report that a PM, engineer, and executive can all act on consistently

## Uncertainty dimension discussed

**Total experiment risk.** This is not a new source of uncertainty. It is the realistic case where all earlier sources are present at once: sampling noise, representativeness, metric choice, judge noise, subjectivity, multiple testing, overfitting, drift, and model evolution. The lesson is how to keep the decision honest when none of them can be ignored completely.

## Tools and Playgrounds

Experiment-Report Evaluator (`experiment_report.html`) — submit the capstone write-up and check whether the evidence supports the decision.

Supporting tools: Uncertainty calculator, Judge auditor, Holdout simulator, Compounding visualizer, and Temporal Drift Lab.
