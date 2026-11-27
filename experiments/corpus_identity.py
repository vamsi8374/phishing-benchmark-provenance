"""
Control experiment: how easily is CORPUS IDENTITY learned, independent of label?

    python experiments/corpus_identity.py

THE ARGUMENT
------------
The leakage probe shows the Date header alone separates phishing from ham at
F1 99.91% on the standard pairing. A defender of the benchmark could reply
that phishing genuinely does arrive at different times of day, or that the
probe is picking up something real.

This experiment removes that reply. We take two corpora that carry the SAME
label — Nazario and Nigerian_Fraud are both phishing — and ask a classifier to
tell them apart. There is no phishing-versus-ham signal available, because
every message in both classes is phishing. Anything the classifier achieves is
pure provenance.

If corpus identity is separable at ~99% when the label is held constant, then
in the standard pairing — where corpus identity and label are perfectly
confounded — a model can score ~99% by learning provenance and nothing else.
That is the whole argument, and this experiment is what makes it airtight.

We also report a same-corpus control: split Nazario in half at random and try
to separate the halves. That must land at chance. If it does not, the
methodology itself is broken and nothing else here can be trusted.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR, RESULTS_DIR  # noqa: E402

csv.field_size_limit(10_000_000)

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import Pipeline                       # noqa: E402
from sklearn.model_selection import cross_val_score         # noqa: E402
import numpy as np                                          # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

# 95% CI half-width. With n=5 seeds the sampling distribution of the mean
# is Student's t with 4 df, so the critical value is 2.776, not the
# normal-approximation 1.96. Update this if SEEDS changes length.
T_CRIT_95 = 2.776

N_PER_SIDE = 800


def read(name: str, label_filter: str | None = None) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if label_filter is not None and (r.get("label") or "").strip() != label_filter:
                continue
            body = (r.get("body") or "").strip()
            if len(body) < 30:
                continue
            rows.append(r)
    return rows


def view_full(r) -> str:
    return f"{r.get('sender','')}\n{r.get('subject','')}\n{r.get('body','')}"


def view_body(r) -> str:
    return r.get("body", "") or ""


def view_date(r) -> str:
    return r.get("date", "") or ""


VIEWS = {"full": view_full, "body_only": view_body, "date_only": view_date}


def separate(a: list[dict], b: list[dict], view, seed: int) -> float:
    """Mean 5-fold CV accuracy for telling corpus A from corpus B."""
    rng = random.Random(seed)
    n = min(len(a), len(b), N_PER_SIDE)
    A = rng.sample(a, n)
    B = rng.sample(b, n)
    X = [view(r) for r in A] + [view(r) for r in B]
    y = [0] * n + [1] * n
    if not any(x.strip() for x in X):
        return float("nan")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                  ngram_range=(1, 2), min_df=2,
                                  max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    try:
        return float(np.mean(cross_val_score(pipe, X, y, cv=5,
                                             scoring="accuracy")))
    except ValueError:
        return float("nan")


def ci(vals: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% normal CI over seeds."""
    v = [x for x in vals if x == x]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    if len(v) < 2:
        return m, 0.0
    return m, T_CRIT_95 * float(np.std(v, ddof=1)) / (len(v) ** 0.5)


def main():
    naz  = read("Nazario.csv", "1")
    nig  = read("Nigerian_Fraud.csv", "1")
    spam = read("SpamAssasin.csv", "0")

    if not naz or not nig:
        print("❌  Need Nazario.csv and Nigerian_Fraud.csv in "
              f"{DATA_DIR.name}/")
        sys.exit(1)

    # Same-corpus control: split Nazario at random. Must be chance.
    rng = random.Random(99)
    shuf = naz[:]
    rng.shuffle(shuf)
    half = len(shuf) // 2
    naz_a, naz_b = shuf[:half], shuf[half:]

    pairs = [
        ("Nazario vs Nigerian_Fraud  (both PHISHING)", naz,   nig),
        ("Nazario vs SpamAssassin    (the benchmark)", naz,   spam),
        ("Nazario-A vs Nazario-B     (same corpus)",   naz_a, naz_b),
    ]

    print("=" * 88)
    print("  CORPUS IDENTITY CONTROL")
    print("  Can a classifier tell which CORPUS a message came from,")
    print(f"  independent of its label?   {len(SEEDS)} seeds, 5-fold CV, 95% CI")
    print("=" * 88)
    print()
    hdr = f"{'pair':<44}" + "".join(f"{v:>14}" for v in VIEWS)
    print(hdr); print("-" * len(hdr))

    out = {}
    for name, a, b in pairs:
        if not a or not b:
            print(f"{name:<44}{'(corpus missing)':>42}")
            continue
        cells, entry = [], {}
        for vname, vfn in VIEWS.items():
            scores = [separate(a, b, vfn, s) for s in SEEDS]
            m, h = ci(scores)
            entry[vname] = {"mean_accuracy": m, "ci95_halfwidth": h,
                            "seeds": scores}
            cells.append("     n/a      " if m != m else f"{100*m:>9.2f}±{100*h:<4.2f}")
        print(f"{name:<44}" + "".join(cells))
        out[name] = entry

    print()
    print("  HOW TO READ THIS")
    print("  Row 1 holds the label constant at PHISHING. Any accuracy above")
    print("  chance there is provenance and nothing else. If it is ~99%, then")
    print("  in row 2 — where corpus identity and label are perfectly")
    print("  confounded — a model can reach ~99% without learning anything")
    print("  about phishing.")
    print("  Row 3 is the sanity check. It must be ~50%. If it is not, the")
    print("  methodology is broken and rows 1 and 2 mean nothing.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "corpus_identity.json").write_text(json.dumps(out, indent=2))
    print()
    print("✅  corpus_identity.json")
    print("=" * 88)


if __name__ == "__main__":
    main()
