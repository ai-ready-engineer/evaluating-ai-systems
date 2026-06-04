# Lesson 1 — The Evolution of "Eval" - from traditional software to ML to AI Agents

Eval in ML is running an *experiment*, not running a *check*.


## Concepts

- Progress means different things in SW 1.0 vs ML — a visible state change vs. a metric you must *establish* moved
- How the meaning of "eval" changes from software 1.0 to ML to AI agents. One word is used to mean many different things
- An eval in ML and AI is an *experiment* to *estimate* a quantity and to identify where we need to improve
- Core concepts: metrics; test sets; the sealed test set
- A score is an *unbiased estimate* of the truth when the test set is fair; sample size shapes trust
- Why "regression testing" doesn't transfer cleanly to ML — shown by example, not yet explained
- List (just list) the sources of uncertainty and error in experiment design — a map for the rest of the course
- Begin the SW 1.0 ↔ ML similarities/differences comparison
- We will begin to list the sources of uncertainty and errors
- we will begin to see calculator to estimate noise and error. we will later understand its principles

[what we leave for session 2: the notion]

> The class will end with a list of reasons why we may be uncertain about our experiment, and a list of traps we need to avoid.

## Patterns

- Read a score as an *estimate*, not a fact — and quote the sample size next to it
- Keep a sealed test set, used only to measure
- Re-run before believing a change

## Anti-patterns — and how they materialize

- **Reading a re-run drop as a regression.** Re-run the same eval; a few checks "fail" anyway — it's noise, not a break. (Don't formalize multiple comparisons yet — just show it.)
- **Quoting a single number as ground truth.** A 78% from 100 cases gets reported as "78%" with no hint it could have been 73% or 83%.

## Understand · report · reduce

The uncertainty this lesson handles: **sampling noise** — a score is an estimate that could have come out differently.

- **Understand it** — a score on a test set is one draw; sample size drives how much it wobbles between runs.
- **Report it** — quote the sample size, and give the score as a plausible range, not a bare point.
- **Reduce it** — use a larger, fairer test set; re-run before believing a change.

## Tools

> Each tool is dual-use — a *playground* to feel the concept, and a *production instrument* you point at your own eval numbers.

- Uncertainty calculator — `accuracy_estimator.html` (counts in → plausible range out; never call it "Beta")
- `sampling_noise.html`
- Multi-class classifier playground — 5×5 confusion, per-class precision/recall, class-imbalance demo `[NEW]`
- Ask the Playground `[NEW]`

## Content notes

- Staircase starts here: SW 1.0 (check) vs ML (experiment). Begin the running comparison table.
- "Regression" = regression *testing*: in SW a regression suite reliably catches a break; in ML a dropped score may be noise. Show by example (re-run, some checks "fail" anyway); don't formalize multiple hypothesis testing yet.
- Sources of uncertainty/error — just list, as a course map: sampling noise, sampling bias, multiple comparisons, variance across domains, choice of metric, drift, model evolution, judge noise, overfitting.






## To discuss

- Name the false-alarm phenomenon casually ("lucky drops") or leave it nameless until Lesson 4?


# Notes

Scope: classification — multi-class, where each instance is assigned exactly one of N classes. Customer-support ticket routing as the running example. We keep accuracy cleanly binary per example.

We explicitly **call out, but do not cover, two extensions** here: **multi-label** classification (a single instance can carry several tags, each its own yes/no property) and **non-classification** tasks (generation, extraction, agent trajectories). Both arrive in later lessons; L1 stays multi-class so the sampling-noise story is uncluttered.

