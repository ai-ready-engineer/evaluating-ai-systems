"""Build L1_generated.pptx — the full Lesson-1 narrative — cleanly and reproducibly.

    python build_deck.py

The deck (L1_main.pptx) was restructured by hand into a 49-slide narrative, so the build is
now driven by build_l1_full.py, which reconstructs the text slides from lesson_1/assets/
l1_spec.json and places faithful renders for the hand-designed slides. It never touches
L1_main.pptx. To refresh the inputs after editing L1_main by hand, run extract_l1_spec.py first.

    python extract_l1_spec.py   # re-capture spec + images + slide renders from L1_main
    python build_deck.py        # rebuild L1_generated.pptx

Legacy pipeline (pre-restructure): start from private/AI_Design_Course_Opening.pptx, run
reframe_opening.py then build_l1_slides.py. Kept in the repo for reference only.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LESSON = HERE.parent
DECK = LESSON / "L1_generated.pptx"


def main():
    subprocess.run([sys.executable, str(HERE / "build_l1_full.py")], check=True)
    print(f"built {DECK.name}")


if __name__ == "__main__":
    main()
