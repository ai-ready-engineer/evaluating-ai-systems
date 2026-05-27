# Lesson 4 — Overfitting & validation datasets *(provisional)*

**Thesis:** an eval number is optimistic when the system has seen — directly or indirectly — what you're testing it on.

## Concepts

- Overfitting as the gap between the score you measured and the behaviour you'll get in the wild
- The different *types* of overfitting:
  - ML — the model memorizes the training set, or a leaked test set
  - Gen AI — the prompt / system is tuned to the eval set
- Validation datasets — the train / validation / test split, why each set exists, and the sealed test set as the defense (callback to L1)

(Multiple hypothesis testing — the lucky-winner problem — is covered in L3, where metrics multiply.)

## Patterns

- Keep a sealed test set — touched only to report, never to tune
- Use a separate validation set for iteration; the test set stays unseen
- Report the number from data the system has never been optimized against
- Freeze and version the test set; watch for benchmark contamination

## Anti-patterns — and how they materialize

- **Tuning against the test set.** Tweak the prompt, re-run the same 50-item eval, score climbs 72 → 88 over ten iterations — then a fresh 50 items scores 73. The 16 points were overfitting, not improvement.
- **Leaked / contaminated test data.** The model saw the benchmark in training; the score measures memorization, not capability.

## Understand · report · reduce

The uncertainty this lesson handles: **overfitting** — the score is optimistic when the system has seen, or been tuned on, the eval data.

- **Understand it** — the gap between the measured score and behaviour in the wild grows every time you tune against the eval.
- **Report it** — report the validation-vs-test gap, and be explicit about what the system was tuned against.
- **Reduce it** — keep a sealed test set; iterate on a separate validation set; freeze and version the test set.

## Tools

> Dual-use — a playground for *how easily we overfit*, and a production instrument for sizing your own train-vs-test gap.

- Validation-dataset / holdout simulator — tune freely against one set, watch the train-vs-test gap widen

## Content notes

- "Different types of overfitting" is the spine: model overfitting and prompt overfitting are the same phenomenon at two levels — the system has seen, or been tuned on, the eval data.
- Multiple hypothesis testing moved to L3 (many metrics = many comparisons).
- Provisional — to refine once Lessons 1–2 are locked.
