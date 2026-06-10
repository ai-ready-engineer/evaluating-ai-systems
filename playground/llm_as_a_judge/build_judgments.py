"""Synthesize the L2 judge-calibrator dataset and pre-compute every judge's verdict.

The L2 mechanic: freeze a set of agent outputs, then score the SAME outputs with several
judges and watch the headline number move with the judge. So we need a fixed, committed set
of (ticket, reference, reply) triples plus per-judge scores — analogous to L1's committed
predictions/. Run once, commit the JSON, and both the notebook and the HTML browse it offline.

The dataset is ITSM-flavored and fully synthetic (numpy, fixed seed): for each support ticket
we have a senior engineer's reference answer and FOUR reply variants of known intended quality:

  * good     — a correct, concise paraphrase of the reference        (quality 1)
  * verbose  — correct content, padded with courtesy and filler      (quality 1)
  * partial  — right topic, missing the actual fix                   (quality 0)
  * wrong    — confidently wrong advice                              (quality 0)

Two simulated human annotators label each reply (mostly agreeing with the intended quality,
occasionally noisy), so the lab can ask the L2 "judge of judges" question: does a judge agree
with a human, and what is Cohen's kappa?

    python build_judgments.py            # synthesize + score, write judgments/ + data.js
    python build_judgments.py --bundle   # just re-bundle existing JSON into data.js

Reads .env (LIVE, OPENROUTER_* / OPENAI_* / ANTHROPIC_*) exactly like the notebook; with
LIVE != true every LLM judge falls back to its deterministic mock, so this runs with no key.
"""
import sys
import json
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import numpy as np

import lab

HERE = Path(__file__).resolve().parent
OUT = HERE / "judgments"
SEED = 42


# --- A small ITSM ticket bank: ticket + reference (golden) resolution. -------------------
TICKETS = [
    ("I can't log in to the VPN from home since this morning.",
     "Reset your VPN credentials in the self-service portal, then reconnect using the new one-time password."),
    ("My laptop won't connect to the office Wi-Fi.",
     "Forget the office network on the laptop, then reconnect and re-enter the Wi-Fi password from the welcome email."),
    ("Outlook keeps asking for my password in a loop.",
     "Clear the cached credentials in Credential Manager, then restart Outlook and sign in once."),
    ("I need access to the shared finance drive.",
     "Submit an access request for the Finance share in the portal; your manager must approve it before access is granted."),
    ("The printer on the 3rd floor shows offline.",
     "Power-cycle the printer and re-add it by IP address 10.2.3.40 from the Printers settings."),
    ("My account is locked after too many login attempts.",
     "Wait 15 minutes for the auto-unlock, or request an immediate unlock from the self-service portal."),
    ("Teams audio is not working in meetings.",
     "Set the correct microphone and speaker under Teams Devices settings, then make a test call."),
    ("I forgot my password and can't reset it.",
     "Use the Forgot Password link on the login page and verify with your registered phone to set a new password."),
    ("Software install is blocked by admin rights.",
     "Request the application from the Software Center; approved apps install without admin rights."),
    ("My screen flickers after the latest update.",
     "Roll back the display driver in Device Manager, then check Windows Update for a newer stable driver."),
    ("Email attachments over 20MB won't send.",
     "Upload the file to the shared drive and send the share link instead; attachments are capped at 20MB."),
    ("Two-factor codes are not arriving on my phone.",
     "Re-register your device in the authenticator app, or switch to email codes in your security settings."),
    ("I deleted an important file from my desktop.",
     "Restore it from the Recycle Bin, or recover an earlier copy from File History or OneDrive version history."),
    ("The website is loading very slowly for me.",
     "Clear the browser cache and disable extensions; if it persists, try the corporate VPN to rule out routing."),
    ("My corporate phone won't sync email.",
     "Remove and re-add the Exchange account on the phone, then accept the device management prompt."),
    ("I can't open a .docx file sent by a colleague.",
     "Open Word first, then use File > Open and choose the document; if it is corrupt, ask them to re-save as .docx."),
]

# --- Reply variants. Each is a transform of the reference of KNOWN intended quality. ------
COURTESY_PRE = (
    "Thank you so much for reaching out to IT support, and I'm really sorry for the "
    "inconvenience this has caused you today. I completely understand how frustrating "
    "this must be. ")
COURTESY_POST = (
    " Please don't hesitate to reply to this ticket if you need anything else at all — "
    "we're always happy to help and your satisfaction is very important to us.")


def make_variants(reference):
    """Return {variant_name: (reply_text, intended_quality)} for one reference answer."""
    good = reference                                  # correct, concise (paraphrase-ish)
    verbose = COURTESY_PRE + reference + COURTESY_POST  # correct content, heavily padded
    # partial: keep the first clause / topic but drop the actual fix
    first = reference.split(",")[0].split(";")[0]
    partial = (f"This is a known issue with that. {first}. Our team is aware and looking "
               f"into it, so please hang tight for now.")
    wrong = ("You should simply reboot your machine three times and the problem will sort "
             "itself out; there's nothing else that needs to be done on your end.")
    return {
        "good":    (good,    1),
        "verbose": (verbose, 1),
        "partial": (partial, 0),
        "wrong":   (wrong,   0),
    }


def _annotator(quality, rng, flip_p):
    """Simulate one human label: usually the intended quality, flipped with prob flip_p."""
    return int(1 - quality) if rng.random() < flip_p else int(quality)


def synthesize():
    """Build the frozen ITSM judgments DataFrame-like list of item dicts."""
    rng = np.random.RandomState(SEED)
    items = []
    for ti, (ticket, reference) in enumerate(TICKETS):
        for variant, (reply, quality) in make_variants(reference).items():
            # Two noisy human annotators. Annotator 1 is a bit stricter, annotator 2 a bit
            # more lenient — so they sometimes disagree, which drives the kappa demo.
            h1 = _annotator(quality, rng, flip_p=0.10)
            h2 = _annotator(quality, rng, flip_p=0.18)
            items.append({
                "id": f"{ti}:{variant}",
                "ticket": ticket,
                "reference": reference,
                "reply": reply,
                "variant": variant,
                "quality": int(quality),       # intended ground truth
                "human_1": h1,
                "human_2": h2,
                "length": len(lab._tokens(reply)),
            })
    return items


def score(items):
    """Attach every judge's verdict to each item (programmatic, reference-based, LLM mocks)."""
    for it in items:
        reply, ref, ticket = it["reply"], it["reference"], it["ticket"]
        it["exact_match"]     = lab.exact_match(reply, ref)
        it["normalized"]      = lab.normalized_match(reply, ref)
        it["token_overlap"]   = round(lab.token_overlap_score(reply, ref), 4)
        it["overlap_match"]   = lab.token_overlap_match(reply, ref, threshold=0.5)
        it["judge_strict"]    = lab.llm_judge("strict",    ticket, ref, reply)
        it["judge_lenient"]   = lab.llm_judge("lenient",   ticket, ref, reply)
        it["judge_scale_1_5"] = lab.llm_judge("scale_1_5", ticket, ref, reply)
    return items


JUDGE_COLS = ["exact_match", "normalized", "overlap_match",
              "judge_strict", "judge_lenient", "judge_scale_1_5"]


def headline(items):
    """A compact per-judge headline number, for the build log and the HTML."""
    n = len(items)
    out = {}
    for col in JUDGE_COLS:
        vals = [it[col] for it in items]
        if col == "judge_scale_1_5":
            out[col] = round(sum(vals) / n, 3)             # mean on 1-5
        else:
            out[col] = round(sum(vals) / n, 3)             # pass rate
    out["human_mean_quality"] = round(sum(it["quality"] for it in items) / n, 3)
    return out


def build():
    items = score(synthesize())
    obj = {
        "dataset": "itsm_replies",
        "n": len(items),
        "judges": JUDGE_COLS,
        "headline": headline(items),
        "items": items,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "itsm_replies.json"
    path.write_text(json.dumps(obj, indent=2))
    print(f"itsm_replies: wrote {len(items)} judged replies -> {path.relative_to(HERE)}")
    print("  headline by judge:")
    for k, v in obj["headline"].items():
        print(f"    {k:>20s}: {v}")
    return obj


def bundle():
    """Bundle the per-dataset JSON into a single data.js the HTML can load over file://."""
    data = {}
    for p in sorted(OUT.glob("*.json")):
        data[p.stem] = json.loads(p.read_text())
    (OUT / "data.js").write_text("window.L2_JUDGMENTS = " + json.dumps(data) + ";\n")
    print(f"bundled {list(data)} -> {(OUT / 'data.js').relative_to(HERE)}")


def main(argv):
    if "--bundle" in argv:
        bundle()
        return
    build()
    bundle()


if __name__ == "__main__":
    main(sys.argv[1:])
