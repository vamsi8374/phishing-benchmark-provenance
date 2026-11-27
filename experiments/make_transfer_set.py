"""
Build the out-of-distribution transfer set.

    python experiments/make_transfer_set.py

CEAS_08 is 2008 commercial spam and ham. It is deliberately kept out of the
training and evaluation splits and used only here, as a corpus nothing in the
system has ever seen.

This is the experiment that separates the approaches. A supervised classifier
fitted on 2015-2021 credential phishing has learned that corpus's vocabulary;
a 2008 pharmaceutical-spam corpus shares almost none of it. The cascade, by
contrast, has no fitted parameters at inference time at all — it never saw a
training label. Whatever the in-corpus table says, this is where the practical
difference between "needs labelled data per tenant" and "does not" shows up.

Report the result honestly in either direction. If the supervised baselines
transfer fine, that is a finding and the paper must say so.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR  # noqa: E402

csv.field_size_limit(10_000_000)

SRC   = DATA_DIR / "CEAS_08.csv"
OUT   = DATA_DIR / "ceas_transfer.csv"
SEED  = 1337          # different from the split seed, on purpose
N_PER_CLASS = 2000
MIN_BODY = 30
FIELDS = ["sender", "receiver", "date", "subject", "body", "urls", "label"]

import re  # noqa: E402
URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(s: str) -> str:
    return CTRL_RE.sub("", s or "").strip()


def main():
    if not SRC.exists():
        print(f"❌  {SRC} not found.")
        print("    Copy CEAS_08.csv from the Kaggle bundle into "
              f"{DATA_DIR.name}/ first.")
        sys.exit(1)

    buckets: dict[str, list[dict]] = {"0": [], "1": []}
    with SRC.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            lbl = (row.get("label") or "").strip()
            if lbl not in buckets:
                continue
            body = clean(row.get("body", ""))
            sender = clean(row.get("sender", ""))
            if len(body) < MIN_BODY or not sender:
                continue
            buckets[lbl].append({
                "sender": sender[:500],
                "receiver": clean(row.get("receiver", ""))[:300],
                "date": clean(row.get("date", ""))[:100],
                "subject": clean(row.get("subject", ""))[:300],
                "body": body[:8000],
                # Source column is a 0/1 flag, not URLs — rebuild it.
                "urls": ", ".join(list(dict.fromkeys(URL_RE.findall(body)))[:30]),
                "label": lbl,
            })

    rng = random.Random(SEED)
    n = min(N_PER_CLASS, len(buckets["0"]), len(buckets["1"]))
    rows = rng.sample(buckets["1"], n) + rng.sample(buckets["0"], n)
    rng.shuffle(rows)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"✅  {OUT.name}: {len(rows)} emails ({n} phishing / {n} legitimate)")
    print("    Out-of-distribution. Never used for fitting or tuning.")


if __name__ == "__main__":
    main()
