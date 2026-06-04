# Lesson 1 — lab

All the Python for Lesson 1 lives here. The decks, charts, and other assets it reads and
writes stay one level up in `lesson_1/` (`assets/charts/`, `L1_generated.pptx`, etc.).

There are two independent groups of code: the **Software 1.0 teaching example**, and the
**slide-deck build pipeline**.

## Software 1.0 teaching example

The running example for "an eval is a *check* in Software 1.0": a small, deterministic,
human-written rule that you can test exactly. It computes a host-health anomaly score as the
Euclidean distance of two normalized metric deviations from the all-normal point `(0, 0)`.

- `host_health.py` — the function `deviation_score(cpu_dev, mem_dev)`. Both inputs are in
  `[0, 1]` (validated); the result is `sqrt(cpu_dev**2 + mem_dev**2)`: `0` = healthy, larger =
  more anomalous.
- `test_host_health.py` — `pytest` tests. Landmark example cases (via `pytest.approx`, so float
  equality isn't fragile), boundary acceptance at `0` and `1`, `ValueError` on out-of-range
  inputs, and property checks over an 11×11 grid (output range, symmetry, monotonicity,
  reduction to a single axis). Run with `uv run pytest` from this folder.
- `L1_sw1_example.ipynb` — the same example as a notebook: the function, the exact tests, a
  vector plot of one point's distance, and a contour of the full score surface with the tested
  points overlaid. It makes the lesson's point — a few tested points pin down the whole
  continuous space, *because the rule is smooth and known*. That trust is exactly what breaks
  once the rule is learned (SW 2.0) or rented (SW 3.0).

## Slide-deck build pipeline

Generates the Lesson 1 deck, `../L1_generated.pptx`. Follows the project slide conventions
(see `../../CLAUDE.md`): every text run ≥ 22pt, minimal elements per slide, light mode.

- `build_deck.py` — the entry point. Run `python build_deck.py` for a full, clean rebuild:
  it copies the pristine opening (`../AI_Design_Course_Opening.pptx`), reframes it, then appends
  the L1 teaching slides. Re-running always reproduces the same deck with no orphaned slide parts.
- `reframe_opening.py` — rewrites the 17 opening course slides' text to this course (eval,
  estimation, experiment design). Edits text in place; formatting preserved.
- `build_l1_slides.py` — appends the L1 teaching slides (title + short body + at most one chart),
  pulling charts from `../assets/charts/`. Append-only, so run it through `build_deck.py`.

## Running

```
uv run pytest                 # the Software 1.0 tests
uv run jupyter notebook L1_sw1_example.ipynb
python build_deck.py          # rebuild ../L1_generated.pptx
```
