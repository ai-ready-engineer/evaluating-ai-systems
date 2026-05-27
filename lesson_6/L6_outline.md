# Lesson 6 — Subjectivity and Noise in Judges and Ground Truth

**Number:** L6
**Tagline:** When there is no clean ground truth — when the "right answer" is contested, and the judge has its own opinions.
**Source of uncertainty:** Subjectivity & judge errors

## What we cover

- When ground truth itself is contested — humans disagree on the right answer
- Annotator / rater disagreement, and inter-rater agreement
- LLM-as-judge has biases too: length bias, position bias, sycophancy, format preferences
- Multiple raters and pooling judgments
- Subjectivity is not bias to be "removed" — it's the reality of the task
- A deeper "judge of judges" — auditing a judge for systematic errors

## Skills you unlock

- Tell when "no ground truth" is the actual situation, not a temporary gap
- Set up multiple raters and read inter-rater agreement
- Spot common LLM-judge biases — length, position, sycophancy
- Push back on a single judge's score for a subjective task
- Decide whether to fix the judge or to live with — and report — the subjectivity

## Uncertainty dimension discussed

**Subjectivity & judge errors.** When the question has no single right answer, the score isn't just noisy — it's contested. Adding raters helps; pretending the judge is objective doesn't.

## Tools and Playgrounds

Judge auditor (`judge_audit.html`) · Inter-rater agreement viewer
