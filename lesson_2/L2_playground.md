# Playground for lesson 2

Same shape as L1, different axis. L1 froze the system and resampled the **data** to watch the
score wobble — its only source of uncertainty was *which cases* landed in the test set. L1 also
made a second, quieter assumption: that the **comparison** of an AI answer to the truth was
itself perfect. L2 removes that assumption. The thing that scores an answer — the **judge** — is
an instrument, and instruments are imperfect in two distinct ways: **bias** (systematic) and
**noise** (random). The thing the student should walk away feeling: those two defects do
*different* things to your number, and both must be folded into how uncertain you are about it.

## The one idea: bias ≠ noise

- **Bias is systematic.** A consistently generous or harsh judge shifts every score the same
  direction. It moves the *centre* of your measurement off the truth, and **more data does not
  fix it** — it just shrinks your interval around the wrong value.
- **Noise is random.** An inconsistent judge (temperature, an ambiguous rubric, a different
  annotator) scores the same answer differently on re-run. It **widens** the spread of your
  measurement, and it *does* average out with more judgments.

The payoff ties back to L1's uncertainty tool, but with a more general instrument: instead of a
closed-form Beta interval (which only fits a binomial proportion), we use the **bootstrap** —
resample the scored test set with replacement, recompute the metric thousands of times, read off
the spread. Bootstrap works for any score (binary, ordinal 1–5, continuous) and shows the two
defects cleanly: **noise widens the bootstrap interval; bias slides it off the truth — and the
bootstrap cannot see the bias.** That last point is the sharpest beat of the lesson: a biased
judge hands you a *tight, confident* interval centred on the wrong answer.

## Why a graded score (not pass/fail)

Gen-AI answers are rarely cleanly right or wrong — "Paris", "Paris, the City of Light", "a city
in France" sit on a spectrum — so the judge gives a **graded quality score** (0–100), not a
binary stamp. This also matters statistically: a *binary* judge's pass-rate variance is pinned by
its mean (a Bernoulli is fully determined by its mean), so judge noise gets absorbed and does not
widen a single run's interval. A **graded** score lets judge noise add genuine variance
(`Var ≈ σ²/n`), so the "noise widens the interval" beat is both visible and honest. This also
sets up the course's score-types thread (binary / ordinal / continuous) and the syllabus line that
gen-AI progress is *better/worse*, not *correct/incorrect*.

## Assets

### The frozen pool

A small, committed pool of **40 (question, AI answer)** pairs, each carrying a **known true
quality** in [0, 1] — the thing we pretend to be able to peek at but never can in real life. By
construction the pool's **true mean score is exactly 70 / 100**. Four answer flavours:

- **great** (truly correct, exact phrasing) — quality ≈ .85
- **good** (truly correct, paraphrased / extra context) — quality ≈ .78 (a *harsh* judge
  under-rates these — the exact-match trap as a bias, not a hard failure)
- **weak** (wrong but plausible) — quality ≈ .45 (a *generous* judge over-rates these)
- **bad** (clearly wrong) — quality ≈ .30

Qualities are kept off the 0/1 ceiling on purpose, so that judge noise — not just the spread of
true quality across the pool — is what visibly drives the bootstrap width.

### The judge model

The judge rates each answer `rating = clamp( quality + bias + noise·z )`, with `z ~ N(0,1)`.
Two dials expose the two defects independently:

- **Bias** ∈ [−25, +25] points — harsher (←) to more generous (→). A pure centre-shift.
- **Noise** (σ) ∈ [0, 30] points — perfectly consistent to very inconsistent. A pure spread.

A third **test-set size** dial (20–500) lets the student watch the bias-vs-noise asymmetry play
out against `n`: more data narrows the interval around the noise-blurred mean, but the
bias-induced offset stays put (and the interval excludes the truth *more* decisively).

## Content flow for the HTML

1. **Where L1 left off** — name the two dropped assumptions (perfect ground truth, perfect
   comparison) and introduce the judge as an instrument with two defects. Two concept cards
   contrast bias (systematic, shifts the centre, unfixed by data) and noise (random, widens the
   spread, averages out).

2. **The simulator** — the frozen pool with its known true mean of 70. Three readouts: **true
   mean score** (fixed 70), **measured** (this run's mean rating), and **gap = measured − true**
   ("the bias you'd never see"). Guided experiment in the takeaway: *noise 0, slide bias* → the
   number slides off 70 and stays there on every Re-judge; *bias 0, raise noise, Re-judge a few
   times* → the number jitters around 70 but keeps returning. Bias moves the centre; noise moves
   each draw.

3. **What the judge did to each answer** — a per-row table: each answer's **true quality** (grey
   bar) beside the **judge's rating** (amber bar + number). Bias slides every amber bar the same
   way; noise jitters each one independently. Makes the aggregate number concrete.

4. **Bootstrap — how sure are you?** — resample the scored test set with replacement (2000×),
   re-score, and draw the distribution of the mean as a histogram with the **95% interval**
   shaded, plus the **true-70 line** (green) and **this-run measurement** (amber). A coverage
   readout states whether the interval actually contains 70. The takeaway names the asymmetry:
   **raise noise → wider interval** (honest, reportable, shrinkable); **slide bias → interval
   shifts off the truth** and the bootstrap can't see it; **raise n with bias on → narrower
   interval that excludes the truth more decisively.** More data cures noise, not bias.

5. **L2 takeaway** — (a) bias and noise are different defects; (b) both widen real uncertainty,
   differently — noise as measurable width, bias as an invisible offset; (c) different fixes —
   noise: average repeats, lower temperature, sharpen the rubric; bias: audit the judge against
   trusted labels (a "judge of judges", picked up in L6).

## What we run inside the lab (notebook)

The HTML is an abstract, fully controllable simulator. The notebook grounds the same three beats
in **real LLM judges over real answers**, so students see the phenomenon is not an artefact of the
toy model:

1. **The exact-match trap → bias.** Score free-text QA answers (paraphrases, added units, extra
   context) with a programmatic exact-match check and with a lenient LLM judge. Exact-match's
   systematic under-counting *is* a negative bias; the gap between the two scorers is the bias made
   visible. *(QA answers committed under `answers/`.)*
2. **Re-run noise.** Hold one LLM judge and one answer fixed; call it K times at `temperature>0`.
   The per-example score spread is the judge's own noise. Aggregate it to show how a single-run
   mean is one draw.
3. **Bootstrap the estimate.** From one scored pass, bootstrap-resample the answers to get the
   95% interval on the mean score — the code behind the HTML's bootstrap panel — and (optionally)
   compare a generous vs. a harsh rubric to see the interval *shift*, not just widen.

Datasets stay deliberately small: the QA answers above, plus `bitext_customer_support` reused
from L1 for a no-ground-truth variant (drafted support replies scored by a quality rubric, where
inter-judge disagreement is the only signal).

## Two paths, one lab

- **HTML** (`index.html`) — two judge-type tabs. *Judge that classifies*: the gen-AI classifier
  demo moved from the L1 lab (real precomputed LLM predictions on the L1 datasets — spin-the-wheel
  draws, accuracy, per-class precision/recall, confusion matrix whose persistent lean is the bias),
  a no-ground-truth batch with a re-judge button, and the ML-vs-LLM **calibration** contrast. *Judge
  that scores*: turn the **bias** and **noise** dials over the frozen pool and watch the measured
  score shift and wobble, then bootstrap it. No Python required, fully offline (loads the L1
  prediction bundle by relative path).
- **Jupyter notebook** (`notebook.ipynb`) — the code that reproduces the three beats on real LLM
  judges and lets students reword a rubric, change the temperature, or point the bootstrap at
  their own scored answers.

## Folder layout

```
playground/judge_noise_and_bias/
  index.html               # no-code path — the two-dial simulator + bootstrap
  notebook.ipynb           # code path — real LLM judges + bootstrap on real answers
  lab.py                   # judge runners (programmatic + LLM) + loaders + bootstrap helper
  prompts/
    strict.j2  lenient.j2  scale_1_5.j2   # graded / binary rubrics (case 1)
    reply_quality.j2                      # no-ground-truth rubric (case 2)
    generate_reply.j2                     # drafts the bitext replies
  answers/
    qa_answers.json        # pre-generated free-text answers + true-quality labels (committed)
    bitext_replies.json    # pre-generated support replies (committed)
  .env.example
```

## Skills exercised

- Tell **bias apart from noise** in a reported score, and predict what each does to the number.
- Read an uncertainty interval honestly — know that it captures noise but **not** bias.
- Use the **bootstrap** to put an interval on any metric (binary, ordinal, continuous), as the
  general-purpose successor to L1's Beta tool.
- Match the fix to the defect: average / cool / sharpen for noise; **calibrate against trusted
  labels** for bias.
- Pick the right score type (binary / ordinal / continuous) — and see why a graded score is what
  lets judge noise show up at all.
