# Lesson 2 — Assessing one "run" of an Agent · Part I

**Number:** L2
**Tagline:** How we programmatically assess whether an AI answer is correct — programmatic checks, judges, and numeric scores. Ground Truth, AI Judges, and Judges of Judges 
**Source of uncertainty:** Variance around a score

## What we cover

- Programmatic checks for the "easy" cases — exact match, format and schema checks, JSON validation
- When there is no clean ground truth — reference-based metrics, and LLM-as-judge
- Score types: binary (0/1), multiclass, ordinal (1–5), continuous
- **Variance around a score** — the same AI answer can be scored differently on re-run, even when the case has a clear answer
- A first "judge of judges" — calibrating a judge against humans, sanity-checking the rubric
- (Subjectivity and "the judge has opinions" is Part II, in L6)

## Skills you unlock

- Choose a scoring approach that fits the task — programmatic, reference-based, or LLM-as-judge
- Write a clear rubric an LLM-as-judge can apply consistently
- Spot when variance is coming from the judge, not from the data
- Calibrate a judge against human labels before trusting it
- Pick the right score type (binary, multiclass, ordinal, continuous) for the output

## Uncertainty dimension discussed

**Variance around a score** — the wobble in the score for a given case, even when the case has a clear answer. The judge — programmatic, LLM, or human — is an instrument with its own noise. Distinct from L1's sampling noise, which comes from picking different cases.

## Tools and Playgrounds

Pick your Judge (`judge_calibrator.html`): see how an AI judge can work. See the effect of minor judge changes on the final result
