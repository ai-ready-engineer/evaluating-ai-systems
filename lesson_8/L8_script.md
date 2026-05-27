# Script

> Talk outline / script. Beats to hit, transitions, what to show on screen, what to say at each turn.

## Open

Put the decision on screen before any numbers:

> We have a new version of the support assistant. It looks better in demos. Should we ship it?

Ask the room what evidence they would need before approving the rollout. Capture the first answers, then sort them into the course vocabulary: metric, sample, judge, guardrail, baseline, uncertainty, decision rule.

## Beats

1. **The claim is not the experiment.**  
   The product claim is "V2 resolves more tickets." The experiment needs a baseline, candidate, population, unit of analysis, primary metric, guardrails, and acceptance criteria. Make the class write the decision rule before seeing any result.

2. **Build the measurement system.**  
   Walk through the data split: dev, validation, sealed test, and judge calibration set. Show why programmatic checks and LLM judges both belong in the same eval, but for different kinds of evidence.

3. **Run the result and resist the headline.**  
   Reveal the aggregate first: V1 78%, V2 84%, delta +6 pp. Let the room react. Then reveal the range, per-domain breakdown, safety guardrail, and judge calibration notes.

4. **Turn evidence into a decision.**  
   The result is not "V2 wins." It is "V2 improves routine cases but regresses on refund / policy-sensitive cases." Write the recommendation: no broad ship; optional limited canary for low-risk intents; focused next experiment for policy-sensitive flows.

5. **Plan the next loop.**  
   Close the experiment by deciding what happens after the report: fix the policy regression, preserve the sealed set, recalibrate the judge, and schedule re-baselining after launch or model change.

## Demo / playground walkthrough

1. Open the Experiment-Report Evaluator with the flawed draft report.
2. Show the evaluator flags: missing decision rule, aggregate-only result, buried guardrail failure, vague judge calibration, no re-baselining plan.
3. Revise the report live:
   - add the product decision and acceptance criteria
   - separate primary metric from guardrails
   - add plausible ranges and sample sizes
   - add per-domain breakdowns
   - add judge-human calibration readout
   - state the recommendation and next experiment
4. Re-run the evaluator and compare the feedback.
5. Optional: open the Compounding visualizer to show why the final uncertainty is wider than the sampling interval alone.

## Close

The single takeaway:

> A good experiment does not answer "which number is bigger?" It answers "what decision does the evidence support, and what risk remains?"
