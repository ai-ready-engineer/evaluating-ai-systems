# Deep Dive — Conditional Probability & Bayes: what a probability *means*

*Optional, take-home. A gentle follow-on to L3 for the curious — not required to follow the lessons.*

**Thesis:** almost every probability in an eval is *conditional* — the honest question is always "given what?". Bayes' rule is how you update a belief when evidence arrives, base rates are the thing everyone forgets, and **calibration** is the special case where the evidence you condition on is the model's own confidence.

## Conditional probability — "given what?"

- A bare probability is rarely the claim you want. The useful quantities are conditional: `P(spam | the model flagged it)`, `P(correct | the case is from a new domain)`, `P(failure | the input is long)`.
- The trap is reading an *unconditional* number as if it were conditional, or swapping the condition. "The judge is 90% accurate" is `P(judge correct)` — it is **not** `P(answer truly bad | judge flagged it)`. Those can be wildly different numbers.

## Bayes' rule — flipping the conditional

- Bayes lets you turn the conditional you can measure into the one you actually care about, using the **base rate**:
  `P(truly bad | flagged) = P(flagged | truly bad) · P(truly bad) / P(flagged)`.
- **The base-rate trap (prosecutor's fallacy), made concrete.** A judge that catches 90% of real failures and false-flags only 5% of good answers sounds excellent. But if real failures are rare — say 2% of traffic — then most flags are false alarms: of every 1000 cases, ~18 true failures are caught and ~49 good answers are wrongly flagged, so a flag means "truly bad" barely a third of the time. The instrument didn't get worse; the *base rate* did the damage.
- This is the same lesson as L1's class imbalance, seen from the belief side: when one class is rare, a good-looking detector can still be mostly wrong about the cases it flags.

## Calibration — conditioning on the model's own confidence

When a model says "92% confident," that 0.92 is a probability claim about itself. It only means something if it cashes out as a conditional proportion: `P(correct | model says ~0.92) ≈ 0.92`. That property is **calibration**, and a confidence number you can't trust is just decoration.

### Seeing calibrated vs. not — the reliability diagram

- Bin predictions by stated confidence (0–10%, …, 90–100%).
- For each bin, plot **mean confidence (x) vs. observed accuracy (y)**.
- **Diagonal = perfectly calibrated.** **Below** the diagonal = **overconfident** (says 90%, right 70%). **Above** = underconfident.
- One-number summary: **ECE** (expected calibration error) — the average distance from the diagonal.
- The intuition without math: *"Take the 100 cases where the model said 90%. Calibrated → ~90 right. Overconfident → maybe 65 right, so 'I'm 90% sure' lies 25% more often than it admits."*

### Two things people get wrong

- **Calibration ≠ accuracy.** They're orthogonal. A model can be very accurate but wildly overconfident (its numbers are meaningless even when its answers are good), or honestly calibrated but not very accurate. Showing both side by side — same predictions — is the "aha."
- **Modern nets and LLMs skew overconfident.** Softmax probabilities run high; verbalized "I'm 95% sure" is worse still. The classic one-knob fix is **temperature scaling** — a natural "turn the dial" demo.

### Why it matters operationally

**Confidence-based routing and abstention only work if the model is calibrated.** "Send everything below 0.7 to a human" is only safe if 0.7 means what it says — otherwise you're auto-approving garbage. The confidence number is load-bearing exactly when you let it make decisions.

## Where this connects

- L1 — class imbalance / base rates (the same fact, seen from the belief side)
- L2 — a judge's accuracy vs. `P(truly bad | judge flagged)`; why base rate decides whether a flag is trustworthy
- L3 — the proportion / probability / belief trio (calibration is the trio applied to the model's own self-report)
- Deep Dive · Gaussian & CLT — the companion on why averages settle
- Possible playground — a reliability diagram with a single distortion/temperature slider that bows the curve off the diagonal while ECE updates live; a lab that runs an LLM classifier on a cached dataset (`goemotions` / `bitext_customer_support`) and reveals its overconfidence on real data. (Name it distinctly from the existing `llm_as_a_judge`, which calibrates a *judge's thresholds* — a different sense of the word.)
