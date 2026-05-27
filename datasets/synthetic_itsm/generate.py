"""Generate a synthetic ITSM multi-label dataset with controllable skew.

Usage:
    python generate.py --n 1000 --skew default
    python generate.py --n 1000 --skew flat
    python generate.py --n 1000 --skew heavy
"""
import argparse
import csv
import random
from pathlib import Path

LABELS = ["billing", "urgent", "refund", "account", "complaint",
          "hardware", "software", "network", "security", "login"]

SKEW_PROFILES = {
    "flat":    {l: 0.20 for l in LABELS},
    "default": {"billing":0.25, "urgent":0.18, "refund":0.12, "account":0.30, "complaint":0.20,
                "hardware":0.10, "software":0.22, "network":0.15, "security":0.05, "login":0.18},
    "heavy":   {"billing":0.45, "urgent":0.30, "refund":0.20, "account":0.50, "complaint":0.35,
                "hardware":0.04, "software":0.08, "network":0.06, "security":0.02, "login":0.10},
}

LABEL_PHRASES = {
    "billing":   ["my invoice is wrong", "I was double-billed", "question about my bill", "billing seems off"],
    "urgent":    ["this is urgent", "ASAP please", "need this fixed now", "critical issue"],
    "refund":    ["I want a refund", "please return the money", "refund my last charge", "I need my money back"],
    "account":   ["account issue", "can't see my account", "my account is locked", "account settings problem"],
    "complaint": ["this is a complaint", "I'm not happy", "terrible service", "I want to complain"],
    "hardware":  ["my laptop won't turn on", "hardware problem", "screen is broken", "keyboard not working"],
    "software":  ["the app crashes", "software bug", "application won't open", "software update failed"],
    "network":   ["can't connect to network", "wi-fi is down", "network keeps dropping", "VPN won't connect"],
    "security":  ["suspicious login attempt", "security alert", "I think I was hacked", "compromised account"],
    "login":     ["can't log in", "forgot my password", "login error", "MFA not working"],
}

GLUE = ["I would appreciate help.", "Please advise.", "Thanks in advance.",
        "Looking forward to your reply.", "Any guidance appreciated.", "What should I do?"]


def make_ticket(rng, skew):
    tags = [l for l, p in skew.items() if rng.random() < p]
    if not tags:
        tags = [rng.choice(LABELS)]
    parts = [rng.choice(LABEL_PHRASES[t]) for t in tags]
    rng.shuffle(parts)
    text = " ".join(parts).capitalize() + ". " + rng.choice(GLUE)
    return text, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--skew", choices=list(SKEW_PROFILES.keys()), default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "data.csv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    skew = SKEW_PROFILES[args.skew]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "text", "labels"])
        for i in range(args.n):
            text, tags = make_ticket(rng, skew)
            w.writerow([f"t{i:05d}", text, ";".join(tags)])
    print(f"Wrote {args.n} tickets to {args.out} (skew={args.skew})")


if __name__ == "__main__":
    main()
