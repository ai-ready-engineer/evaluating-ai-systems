# Deep Dive — Conditional Probability & Bayes

**Code:** DD_cond_and_bayes
**Tagline:** What a probability *means* once you condition on evidence — and why "92% confident" is a promise the model may not keep.
**Status:** Optional, take-home. A gentle follow-on to L3 — not required to follow the lessons.

## What we cover

- Conditional probability — "given what?", and the trap of reading an unconditional number as if it were conditional
- Bayes' rule — flipping the conditional you can measure into the one you care about
- Base rates and the prosecutor's fallacy — why a 90%-accurate judge's flag can still be wrong most of the time
- Calibration — `P(correct | model says 0.9) ≈ 0.9`, the reliability diagram, and ECE
- Calibration ≠ accuracy; why modern LLMs skew overconfident; temperature scaling
- The operational payoff — confidence-based routing and abstention only work if the model is calibrated

## Who it's for

The curious who want the "why" behind the L3 trio and behind trusting (or distrusting) a confidence number. No prerequisites.

## Where it connects

- L1 — class imbalance / base rates, seen from the belief side
- L2 — judge accuracy vs. `P(truly bad | flagged)`
- L3 — proportion / probability / belief, applied to the model's own self-report
- Deep Dive · Gaussian & CLT — the companion deep dive
