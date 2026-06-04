# Script — L1 "Eval is now an Experiment"

> Talk outline / script. Beats to hit, transitions, what to show on screen, what to say at each turn.
>
> Total: ~50 min talk + 10 min playground. Adjust by skipping deep dives.

---

## Open (≈5 min)

The Hook is the 78% / green. We describe that this is the new reality and in this course we work on understanding how to produce reliable eval and how to assess them

**Hook.** Open on a green/red CI dashboard — the familiar comfort of software. Every test either passes or fails. We know what "progress" means: the red turns green.

Lets start by showing a matrix that discusses how is progress measured, etc as we have in the slide

Here we also say that tests in 1.0 are written, reviewed and approved.
We are not testing the entire input space.
They are meant to reasonably cover  that space, though.
We think we know how the space looks like.


Then switch to a classifier output: a tag, a score, a "confidence." Re-run it. The output shifts. "Progress" no longer has a single shape.

When we run evaluation for an ML model we don't really know how the model behaves for different points in the space  - and the space is high dimensional.
And, forcing a model to work well on a specified set of points does not mean that it works well for us. our goal is to have software that works well on future cases. We train and test on past cases. sometimes that helps a model or agent to be better in general, and sometimes it doesn't.

It is very easy to both iterate towards a worse agent and also have less reliable estimates, or both.

Gen AI use cases are even more difficult: eg imagine getting a summary of a court case. who is to say if it's "correct".
So the challenges compound. 

The problem is not so much the uncertainty - there are reasonably well established method for designing experiments. the problem is if we are not aware, or if we design experiments that are likely to give inflated and biased results.



**The frame: SW 1.0 / 2.0 / 3.0.**

- **SW 1.0** — humans write the rules. Tests check the rules. Progress = new feature lands green.
- **SW 2.0 / ML** — we don't know the rules, so we let a model learn them. The "code" is a set of weights. Progress = a metric on a test set moved.
- **SW 3.0 / Gen AI** — we don't even fix the model. We rent it, prompt it, point it at tools. Progress = a metric on a test set moved, **even though we didn't change our code**.

Across these three worlds, **the word "eval" gets reused but means three different things**. That's the whole problem.

> Slide on screen: the three layers side by side. Same word, three meanings.

**Why we even use ML in the first place** — quick honest aside: because we don't know how to code the rule, or because writing the rule by hand would be slower than letting the model figure it out. Sometimes both. This is worth saying out loud so students don't feel ML is mystical.

**The thesis (state it once, plainly):**

> An eval in ML and AI is not a *check*. It is an *experiment* — we *estimate* a quantity, and the answer comes with uncertainty.

Everything else in this lesson — and this course — follows from that one sentence.

---

## Beats

### Beat 1 — The check that doesn't transfer (≈5 min)

Show a tiny SW 1.0 test: `assertEqual(add(2,3), 5)`. Pass / fail. Boring. Reliable.

Now show the same shape applied to a classifier: `assertEqual(classify("my order is late"), "delivery")`. Run it. It passes. Run it 10 times on a re-sampled set. **Some fail.**

What broke? Nothing. The classifier didn't change. The *test* changed — or rather, we drew a different slice of the world. The check-frame doesn't transfer.

> Anti-pattern to name out loud: **"reading a re-run drop as a regression."** It feels like a regression. It isn't.

### Beat 2 — The classifier, the test set, the metric (≈8 min)

Concrete running example: **multi-class ticket classification.** Customer-support tickets, each routed to exactly one of a handful of categories like `billing`, `refund`, `login`, `hardware`. One ticket, one label — multi-class is closer to real work than the textbook binary case, and accuracy stays clean: for any single ticket the prediction is either right or wrong.


this is often an "easy" case:
1. we often have a lot of data
2. we often have ground truth for such data
3. when we see the output of the model, it is easy for us to say if it is equal to the ground truth.


> Aside on the slide: "More than two classes, but still one answer per ticket. Accuracy is just: did we pick the right class? Don't be scared of N>2."

At a basic level when we "eval" an ML system we pick: 

- **The "system under eval** In this case, a classifier. Two of them, actually: a traditional one (TF-IDF + logistic regression) and a gen-AI one (few-shot LLM). Same job. Two different ways of doing it.
- **A (set of) metrics. A scorecard** Accuracy to start. We'll add precision, recall, coverage (the classifier that says "not sure") as alternatives.

- **A test dataset.** A representative sample of tickets. Not corner cases — in AI we rarely think in corner cases, we think in distributions.

- **Identification of Ground truth** 

- **An eval workflow**. In our example, we pick  

- **Reporting of results** Including uncertainty and limitations of validity.


Show one ticket. Show what each classifier returns. Show the ground truth. Score it. We have one data point. **One.** That's not an eval — that's an anecdote.



We can then show a space, and show where a model is good or bad, and show that maybe all our data points are in an area our model does well in. Maybe model 2 does a bit worse on those points but better in general.

The fact that a Model M has an accuracy A on a test dataset D really only tells us that model M has an accuracy A on dataset D. The rest is guesswork, and how good our guesswork is depends on all the decision points above.

That sentence — *M has accuracy A on D* — is the narrowest honest claim we can make from this experiment. Anything else we want to say sits *on top* of it: that A is reliable, that A would hold on tomorrow's tickets, that this model is better than the one we shipped last quarter, that A is the best we've seen. Each of those is an extension, and each is somewhere we can be misled. The rest of this course walks through them one at a time — today's lesson picks up the first one, that even the honest A is just one draw from many we could have observed; next time we go after the judge that produced A; later we go after D itself, then comparisons across runs, then drift. Same M-on-D sentence, peeled apart from a different side each lesson.

Surprisingly, our estimate also depends on what we did *before*. If I tried five versions of the model on the same D and reported the best, the "best A" is already inflated — and I haven't done anything wrong yet, just iterated. We come back to this in L4–L5.


### Beat 3 — One number, one draw (≈8 min)

So let's pick up that first crack in the honest claim — the one where even A is one draw — and feel it.

Run on 100 tickets. Accuracy: 78%.

Stop. Look at it. **"78%."** A clean, round number. Easy to put in a slide. Easy to compare against the last release.

But: where did the 100 tickets come from? They were a sample. If we'd drawn a different 100, the number would shift. The 78% is *one draw* from a distribution of possible scores. It is an **estimate**, not a fact.

And here is the part people skip over: this is true *even if the test set is distributed exactly like production*. The luck isn't in whether the sample is biased — it's in which specific tickets we happen to land on.

Show a 2D picture: every dot is a ticket, plotted in some feature space, coloured by whether the classifier got it right or wrong. The real feature space is high-dimensional, so this 2D view is itself a projection down — and it will look jagged. Patches where the classifier is right, patches where it's wrong, interlocking, no clean boundary. That jaggedness doesn't go away when we get more data; it's a property of the model.

Now imagine our test set is drawn fairly — same proportions as production. We still don't choose *which specific* tickets we land on, only the density. One draw lands in one mix of right and wrong patches; another draw lands in a different mix. Same distribution, different luck of the dots, different A. Two engineers can pull a clean random sample of 100 from the same stream and report different numbers. Neither did anything wrong.

That is why the 78% wobbles. Even when our sample is distributed like prod, it's a matter of luck.

> This is the L1 punchline. Say it slowly: **a score is one draw. Quote the sample size next to it. Better yet, quote a range.**

Two anti-patterns to name:

1. **Quoting a single number as ground truth.** "78%" with no range, no n.
2. **Reading a re-run drop as a regression.** Already named in Beat 1 — now it has a cause.

### Beat 4 — Two thought experiments (≈5 min)

**Experiment A:** every release, you make up a *new* test set. Numbers go up. What changed — the model, or the test? You don't know. You can't decompose them.

**Experiment B:** every release, you use the *same* test set. Numbers also move. Now you can compare across releases — but the test set is getting "cozy." The model (or the prompt, or you) starts to game it. We won't formalize this yet — that's lesson 5 (overfitting) — but flag it.

> The structural answer (also not formalized yet, but introduced): **the sealed test set.** A set you touch only to measure, never to tune. We'll keep saying this word across the course.

**The unavoidable truth:** even if you could grab *all* your customer data today, you still couldn't predict tomorrow's. Uncertainty isn't a bug to remove — it's a property of the situation. The job is to *understand it, report it, manage it.*

### Beat 5 — Today's source has a name: sampling noise (≈4 min)

The course map already came out of Beat 2 — every lesson picks up one of the extensions of M-on-D. Here we just give today's extension its working name, *sampling noise*, and say what we're going to do with it. We'll meet the others by their names as we go: judge noise, variance across customers and time, lucky drops, overfitting to the test set or the prompt or the leaderboard, contested ground truth, drift. And they *compound* — by L7 they're all on the table.

For today, one source, one name:

> A score on a test set is one draw. Sample size sets how much it wobbles between draws. Bigger and fairer set → tighter range.

Three moves we'll practise — and we'll keep using these same three across every later lesson:
- **Understand it** — feel the wobble in the playground.
- **Report it** — quote n and a range, not a bare point.
- **Reduce it** — larger, fairer test set; re-run before believing.

---

## Demo / playground walkthrough (≈10 min)

> Goal: **make the wobble visible.** Students should leave physically uncomfortable with the phrase "78% accurate."

1. **Tour the dataset.** Open the Bitext customer-support lab. Show what a row looks like and what the 5 categories mean — one category per ticket (multi-class). ~30 seconds.
2. **Show both classifiers** on the same example. TF-IDF on the left, gen-AI on the right. Same ticket, sometimes same answer, sometimes not.
3. **"Spin the wheel."** Set test-set size to 50. Spin. Note the accuracy. Spin again. Note it again. Spin five times. The numbers move by 4–8 points just from re-sampling. *Nothing else changed.*
4. **"The one real test set."** In real life you spin once. You construct one set, you label it, that's the set you have. Treat one spin as your real eval. What number would you report?
5. **Uncertainty calculator.** Take that one accuracy (say 39/50 correct). Feed it in. The tool shows a 95% credible interval — say [65%, 88%] — and `P(accuracy > 80%)`. **That is what "78%" actually means.**
6. **Raise n.** Same classifier, test set of 200. Interval tightens. Same classifier, test set of 1000. Interval tightens further. Felt, not derived.
7. **(If time) class-imbalance demo.** Drag the skew slider: the drawn test set fills up with the majority category. A trivial "predict-the-majority-class" baseline climbs on aggregate **accuracy** (it's just riding the dominant class) while its **macro-recall** stays pinned near 1/5 — it never gets a rare category right. Accuracy flatters it; macro-recall exposes it. Setup for L4's imbalance / Simpson's paradox — don't name it yet.

> If the room is engaged: invite a student to call out a sample size and a target accuracy. Compute `P(accuracy > target)` live.

---

## Close (≈3 min)

**The one takeaway, said plainly:**

> A score is one draw. Report it with a range, with the sample size, and with what test set it came from. If you can't, you don't yet have an eval — you have an anecdote.

**What we did *not* do today** (set expectations for the course):
- We didn't talk about *judges* — we assumed ground truth was clean. That's L2.
- We didn't formalize *overfitting to the test set* — that's L4–L5.
- We didn't address *drift* — that's L7.

**Tease L2:** today the test set was the source of wobble. Next time, the *scoring instrument itself* is noisy — even on the same example, run twice, a judge can disagree with itself. Same word, different uncertainty.

---

## Optional deep dives

- All the metrics on the Wikipedia accuracy page — precision, recall, F1, AUC, MCC. Why they exist, when each one matters.
- Why we even compute `Beta(k+1, n−k+1)` — the math under the uncertainty calculator. (The full why lives in the Gaussian / CLT deep dive.)

---

## Scratch — raw notes (to fold in or cut)

> Kept from earlier drafts. Mine for phrases; delete when absorbed.

- sw 1.0 vs 2.0 vs 3.0 — what is progress, who drives it, who does what
- first say why we use ML: we tend to use ML when we don't know how to code, or we are just lazy maybe
- Show example of feature and of test case. show green/red dashboards
- Pick now a classifier. be it ML or gen AI. In AI we need many examples. We think less about corner cases (maybe we should), but in general we want a test dataset that is "representative"
- Example: email classifier for high priority vs normal email
- pick a metric / pick a dataset / pick an eval workflow
- Alternative metrics: recall, coverage (classifier that says "not sure")
- For now we assume we have as test set a random sample of our customer or production data.
- Accuracy / Confidence of prediction
- imagine for each new release you pick a new test set, and results improve. what changed? what makes the result go up?
- Now, imagine you don't pick a new test set and use the old one....
- Even if we could take all data from our customers, we can't predict future data. So we always have uncertainty.
- We list (just to flash them) the sources of uncertainty. Here we focus on one.
- we introduce multi-class classification (one of N classes; accuracy stays binary per example). Example is ticket assignment.

## To discuss

- Name the "lucky drops" phenomenon casually here, or leave it nameless until L4?
- Email classifier vs. ticket assignment as the running example — pick one and stay with it, or use both?
- Time budget — is 50 min realistic with a live audience, or do we cut Beat 4 (the two thought experiments)?
