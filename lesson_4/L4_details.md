# Lesson 3 — Beyond binary: multiple classes & regression *(provisional)*

**Thesis:** the eval-as-experiment idea carries past the spam filter — to systems that pick one of many classes, or predict a number.

## Concepts

- Multiple classes: how you measure error when there are more than two outcomes (per-class accuracy, where classes get confused)
- Regression: how you measure error when the output is a number — and a 1-5 score as the bridge between "a class" and "a number"
- How confident to be in an error number — read it as noise around a mean
- Multiple comparisons (multiple hypothesis testing): once you track many metrics — and re-test often — some will look good by pure chance
- (Uncertainty *around* the variance itself is deferred — that's a later lesson, and the bootstrap is the tool for it)

## Patterns

- Match the metric to the output type — class, number, or ordinal score
- Report a per-class / per-error breakdown, not just one headline number
- Show the noise around a mean, not the mean alone
- Decide which metrics matter up front — don't fish across all of them for the flattering one

## Anti-patterns — and how they materialize

- **One number hides a weak class.** Overall accuracy reads 90%, but one class sits at 20% — the average buried it.
- **A mean error with no spread.** "Average error 0.4" reported as solid when it swings 0.1–0.9 case to case.
- **Binary metric on a graded task.** A 1-5 quality score collapsed to pass/fail throws away most of the signal.
- **Many metrics, pick the winner.** Track ten metrics across a few re-runs; one shows a nice jump — by chance alone — and it gets reported as a real gain.

## Understand · report · reduce

The uncertainty this lesson handles: **noise around a mean & multiple comparisons** — error has a spread, and with many metrics some look good by chance.

- **Understand it** — an error number is a mean with a spread; track many metrics or re-test often and one will jump by pure chance.
- **Report it** — give a per-class / per-error breakdown, show the noise around the mean, and say how many metrics you looked at.
- **Reduce it** — decide which metric(s) matter up front; more data; don't keep re-testing until something flatters you.

## Tools

> Dual-use — a playground to feel the spread, and a production instrument for reading your own error numbers.

- A tool that shows noise around a mean and its variance — `distribution_explorer.html`
- `multiple_hypothesis_testing.html` — many metrics / many tests, some good by chance

## Content notes

- No bootstrap here — it's advanced. Bootstrap belongs to the later lesson on uncertainty around variance. Keep L3 to: measure error, gauge confidence via noise-around-the-mean.
- Multiple hypothesis testing enters here because L3 expands the *set of metrics* — many metrics is itself a many-comparisons problem. (It was foreshadowed in L1 as "lucky drops".)
- Provisional — to refine once Lessons 1–2 are locked.
