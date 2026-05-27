# Lesson 4 — Closed-Book Overfitting: Why "regression testing" generates biased results

**Number:** L4
**Tagline:** Run an eval many times or watch many metrics and some will look good by pure chance. The sealed test set is the defense.
**Source of uncertainty:** Bias from Multiple Hypothesis Testing (MHT)

## What we cover

- "Regression testing" in the AI sense — why the same eval re-run shows different numbers
- Multiple comparisons: with many metrics — and many re-tests — some will move by pure chance
- The lucky-winner problem: best-of-N picked across many tries
- **Validation datasets** — the train / validation / test split as the structural defense
- Sealed test set: touched only to report, never to tune
- Simpson's paradox in comparisons — an aggregate disagreeing with every per-domain story

## Skills you unlock

- Recognize a winner picked from too many tries — tell honest comparison from fishing
- Decide which metrics matter up front, so the results land with weight
- Set up a train / validation / test split that does its job
- Keep a sealed test set — and notice when one is being touched during iteration
- Spot Simpson's paradox in a comparison and ask for the per-domain breakdown

## Uncertainty dimension discussed

**MHT — bias from multiple hypothesis testing.** Track many metrics or re-run many times and one will jump by pure chance. The sealed test set is the structural defense; statistical corrections are the numerical one.

## Tools and Playgrounds

Lucky-winner finder (`multiple_hypothesis_testing.html`) · Holdout simulator
