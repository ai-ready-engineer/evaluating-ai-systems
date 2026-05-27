# Lesson 5 — Open-Book Overfitting and "Overfitting to the CTO"

**Number:** L5
**Tagline:** Every AI team faces overfitting — to the prompt, to the benchmark, to the eval set. Spot it before it ships.
**Source of uncertainty:** Overfitting

## What we cover

- Overfitting as the gap between the score you measured and behaviour in the wild
- The three flavours of AI overfitting:
  - **Model overfitting** — the model memorizes train data, or a leaked test set
  - **Prompt overfitting** — the prompt or pipeline is tuned to the eval set
  - **Leaderboard overfitting** — benchmark "wins" that are really memorization or fishing (the MHT side, covered in L4)
- Benchmark contamination — when the system has already seen the test
- The structural defense (sealed test set, validation discipline) was set up in L4; here we focus on recognizing the symptom

## Skills you unlock

- Spot a score inflated by tuning against its own benchmark
- Tell model overfitting from prompt overfitting from leaderboard overfitting
- Recognize benchmark contamination — when a "win" is really memorization
- Report the validation-vs-test gap so the inflation is visible, not hidden
- Resist the institutional pressure to "improve the eval number"

## Uncertainty dimension discussed

**Overfitting** — the optimistic gap between the measured score and behaviour in the wild. Grows every time the system, the prompt, or the team is tuned against the eval. In modern AI it shows up everywhere — at the model, the prompt, and the leaderboard.

## Tools and Playgrounds

Overfitting visualizer · Holdout simulator (`holdout_simulator.html`)
