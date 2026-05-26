# Experiment Design and Evaluation of AI Systems

> Reliable eval means efficient product cycles, informed decisions, better products, happier customers.

This course is for *everybody* who uses AI. The only prerequisites are curiosity, love for your products, and passion for your customers.

## Who this course is for

- **Engineers and engineering managers** — Iterate towards success. Understand what works, what doesn't, and where to put resources.
- **Product managers** — Translate goals into measurable evals. Ship with confidence.
- **Executives** *(compact course)* — Understand what results say and don't say. Ask impactful questions that help teams grow. Promote a culture that recognizes and manages uncertainty.

## How the course works

Every lesson is built the same way:

- It has **tiles describing what we cover**, so you can tell if the content is familiar, needs a refresher, or is new to you.
- It **isolates one source of uncertainty**, so you meet it on its own before it combines with the others.
- It teaches the same three moves on it: **understand it, report it, manage it.**
- It contrasts **good practice against the anti-pattern** that fools people, each shown through a worked example.
- It hands you a **hands-on playground** — a web tool you first use to feel the concept, then point at your own real eval numbers.
- It includes an optional **code example** for those who want to go deeper.

Optional **deep dives** go further for the curious. Two course-wide tools run across every lesson: **Ask the Playground** (ask any page why a number moved) and the **Experiment-Report Evaluator** (submit a write-up, get structured feedback).

## The sessions

Six **1-hour sessions**, each tackling one source of uncertainty. Every session comes with *totally optional* hands-on labs and deep dives.

| # | Session | Uncertainty dimension unlocked | Playground |
|---|---------|--------------------------------|------------|
| L1 | "Eval" is now an Experiment | Sampling noise | Uncertainty calculator |
| L2 | What we measure | Variability across customers, domains, and time | Distribution explorer |
| L3 | Beyond binary: multiple classes & regression | Noise around a mean & multiple comparisons | Lucky-winner finder |
| L4 | Overfitting & validation datasets | Overfitting | Holdout simulator |
| L5 | Measuring the wrong thing (bias) | Bias | Bias auditor |
| L6 | Doing evals well over time | Drift & compounding | Compounding visualizer |

## What you will learn

By the end of the course, evaluation will look like the feedback loop that lets a team improve faster, ship more often, and still ship reliable AI products. You will be able to:

- Design experiments that make product iteration faster and releases safer.
- Use evals to find where quality is actually blocked.
- Apply a practical toolkit: metrics, test sets, holdouts, breakdowns, judge checks, and re-baselining.
- Spot the traps that make teams move fast in the wrong direction: bias, overfitting, noisy judges, drift, and too many metrics.
- Report results in a shared language: what works, what is uncertain, what the limitations are, and what decision the evidence supports.

The point is speed and quality at once. Better evals shorten the loop between idea and evidence, so teams improve faster without guessing — and make frequent shipping safer.

## Running the site locally

The course site is plain HTML/CSS.

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>.

---

Content by Fabio Casati.
