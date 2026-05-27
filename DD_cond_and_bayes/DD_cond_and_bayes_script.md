# Script

> Take-home deep dive — not lectured live. This is a reading-walkthrough rather than a talk script: the order to read in and the one idea to leave with.

## Reading path

1. **"Given what?"** Almost every eval probability is conditional. The first move is always to name the condition out loud.
2. **Bayes flips it.** You can measure `P(flagged | bad)` but you act on `P(bad | flagged)`. Bayes converts one to the other — and the base rate is the hinge.
3. **The base-rate trap.** Walk the 1000-case arithmetic: a great-looking detector, a rare failure, and most flags turn out false. Same instrument, different base rate.
4. **Calibration as a conditional.** "92% confident" only means something if `P(correct | says ~0.92) ≈ 0.92`. Reliability diagram, then ECE as the one-number gap.
5. **Two traps + the payoff.** Calibration ≠ accuracy; LLMs skew overconfident (temperature scaling); routing/abstention only works if calibrated.

## The one takeaway

A probability is a promise about a *conditioned* world — "given this evidence, this often." Calibration is whether the model keeps that promise; base rates decide whether a flag is worth believing. A confidence number you can't condition on is decoration.
