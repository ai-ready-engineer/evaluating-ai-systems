# Lesson 8 — Designing Reliable Experiments

This lesson is the worked example the course has been building toward. We do not introduce a new uncertainty source. Instead, we run one realistic AI eval end to end and make every earlier lesson do work.

## Capstone case

The product team wants to ship a new version of a customer-support assistant. The assistant reads an incoming ticket, proposes a customer-facing answer, chooses whether to escalate, and tags the issue category.

The claim on the table:

> Candidate V2 resolves more tickets than V1 without making safety, policy, or customer-experience failures worse.

The decision:

> Should V2 ship to production, ship only to a limited slice, or go back for another iteration?

## Experiment plan

### 1. State the decision before measuring

- **Decision owner:** product and engineering lead
- **Baseline:** current production assistant, V1
- **Candidate:** new prompt and routing policy, V2
- **Unit of analysis:** one support ticket and the assistant's proposed handling
- **Population:** production tickets from the last quarter, excluding incidents already used for prompt development
- **Primary decision:** full ship, limited canary, or no ship

### 2. Pre-register metrics and guardrails

Primary metric:

- **Acceptable resolution rate** — percent of tickets where the answer solves the customer need without unnecessary escalation, scored by a calibrated judge rubric.

Guardrails:

- **Policy safety:** no increase in unsafe refund, legal, privacy, or account-access advice.
- **Escalation appropriateness:** urgent or regulated cases must be escalated.
- **Per-domain performance:** no major degradation in any high-volume intent or language stratum.
- **Cost / latency:** candidate must stay within the product budget.

Acceptance criteria:

- V2 must improve acceptable resolution rate by at least 4 percentage points on the sealed test set.
- The plausible range for the improvement should not include zero.
- No pre-registered guardrail may regress beyond its tolerance.
- Judge calibration must be good enough to trust the score, or the result is inconclusive.

### 3. Build the data

Use 1,200 historical tickets:

- 300 dev cases for prompt and rubric development
- 300 validation cases for iteration decisions
- 600 sealed test cases for the final report

The sealed test set is stratified by:

- intent: billing, refund, cancellation, account access, delivery, technical support
- language / locale
- difficulty: routine, ambiguous, policy-sensitive
- customer segment, when relevant

A separate 120-case calibration set is human-labeled by two reviewers. This is used to audit the LLM judge before trusting it on the full test set.

### 4. Choose scoring instruments

Use programmatic checks where the answer has a hard requirement:

- valid JSON / schema compliance
- required escalation flag for regulated intents
- forbidden policy phrases
- missing citation or missing ticket ID

Use a calibrated LLM judge where quality is contextual:

- correctness of the answer
- completeness
- tone
- whether escalation was necessary

The judge rubric is tested against human labels. Disagreements are not treated as noise to hide; they are evidence about where the task is subjective or where the rubric is unclear.

### 5. Run the sealed comparison

Example result on the sealed test set:

| Measure | V1 | V2 | Readout |
|---|---:|---:|---|
| Acceptable resolution | 78% | 84% | +6 pp, plausible range +2 to +10 |
| Routine tickets | 83% | 91% | strong improvement |
| Refund / policy-sensitive tickets | 80% | 73% | regression |
| Unsafe policy advice | 2.0% | 4.5% | guardrail failure |
| Judge-human agreement | 86% | — | acceptable overall, weaker on refunds |
| Median latency | 2.1 s | 2.5 s | within budget |

The aggregate says "ship." The experiment says "not yet."

The primary metric improved enough to matter, but the improvement is concentrated in routine cases. V2 creates a policy-sensitive regression exactly where the product risk is highest. The correct decision is not full rollout.

## Decision report

Final recommendation:

> Do not ship V2 broadly. Canary V2 only for routine, low-risk intents if the product benefit is urgent. Block V2 for refund and policy-sensitive flows until the regression is fixed and re-tested on the sealed or next sealed set.

What we know:

- V2 probably improves routine support quality.
- The overall improvement is not just sampling noise.
- V2 regresses on policy-sensitive refund cases.
- The judge is good enough for aggregate readout but needs additional calibration on refund judgments.

What we do not know:

- Whether the refund regression is caused by the new prompt, the routing policy, or judge confusion.
- Whether the result will hold after the next model version change.
- Whether production traffic has shifted since the historical sample was drawn.

Next experiment:

- Create a focused validation set for refund and policy-sensitive cases.
- Add stricter programmatic policy checks.
- Recalibrate the judge on disagreement cases.
- Re-run on validation; touch the sealed set only for the final candidate.
- Schedule re-baselining after launch and after every model upgrade.

## Patterns

- The product decision is written before the result exists.
- Metrics are separated into primary metric and guardrails.
- The sealed test set is used once for the final comparison.
- The report gives a recommendation, not just a dashboard.
- A positive aggregate can still produce a no-ship decision.

## Anti-patterns — and how they materialize

- **Headline-only shipping.** "V2 is up six points" hides the refund regression and creates customer harm.
- **Metric shopping.** The team tries several score definitions after seeing the result until one makes V2 look clean.
- **Judge laundering.** The LLM judge score is treated as objective even though humans disagree most on the riskiest cases.
- **Sealed-set erosion.** The team fixes the refund prompt by repeatedly checking the sealed test set, turning the final report into another validation run.
- **Stale success.** The eval is treated as permanent even though the model provider or ticket distribution can change next month.

## Understand · report · reduce

The uncertainty this lesson handles: **total experiment risk** — the realistic combination of the course's uncertainty sources.

- **Understand it** — identify which uncertainty sources matter for the decision and which are secondary.
- **Report it** — state the decision, method, sample, ranges, breakdowns, judge calibration, guardrails, and limitations.
- **Reduce it** — improve the data design, add checks, calibrate the judge, isolate risky strata, preserve holdouts, and re-baseline over time.

## Playground for lesson 8

The capstone lab uses the Experiment-Report Evaluator. Students receive a short draft report for the V1 vs V2 support-assistant comparison. The first draft intentionally contains common flaws:

- only the aggregate score is reported
- sample size is present but no range is given
- the refund regression is buried in an appendix
- judge calibration is described vaguely
- no decision rule is stated
- no re-baselining plan is included

Students revise the report until it supports a concrete decision. The evaluator returns structured feedback on:

- claim clarity
- metric and guardrail separation
- uncertainty reporting
- representativeness
- judge reliability
- holdout discipline
- decision quality
- follow-up experiment design

The lesson ends when the class can explain why a higher aggregate score still does not justify a full launch.
