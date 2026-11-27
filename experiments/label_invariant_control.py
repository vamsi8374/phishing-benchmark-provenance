"""
Label-invariant separability control (paper Section sec:identity, Table tab:identity).

    python experiments/label_invariant_control.py

WHY THIS FILE EXISTS
--------------------
results/label_invariant_control.json was reported in the paper but no script
in experiments/ produced it: the code was lost when the experiment was
changed from the original Nazario-vs-Nigerian comparison (which
corpus_identity.py still runs) to the pairs below. This file rebuilds it so
every table in the paper can be regenerated from released code.

THE ARGUMENT (unchanged from the paper)
---------------------------------------
Hold the class label constant and ask whether a classifier can still tell two
corpora apart from body text. Anything above chance is provenance, not class:

    SpamAssassin ham  vs CEAS ham     both legitimate
    SpamAssassin spam vs CEAS spam    both malicious
    SpamAssassin ham A vs B           same corpus, random halves -> must be ~50%

The Nazario-vs-Nigerian comparison was dropped because advance-fee fraud and
credential phishing differ in content, so a classifier could read threat
subtype rather than provenance. The pairs above admit no such difference.

METHOD (identical to corpus_identity.py)
----------------------------------------
Body text only; 800 messages per side sampled per seed; TF-IDF word 1-2grams +
logistic regression; mean 5-fold cross-validated accuracy; 5 seeds; 95% CI
with the t critical value for n=5.

HOW TO TELL WHETHER THE REBUILD IS FAITHFUL
-------------------------------------------
The original per-seed accuracies were preserved. If this reconstruction is
exact, it reproduces them digit for digit:

    SA ham vs CEAS ham     0.950625 0.960625 0.946875 0.950000 0.951250
    SA spam vs CEAS spam   0.978750 0.981250 0.976250 0.980625 0.973750
    SA ham A vs B          0.511875 0.515000 0.503125 0.498750 0.493125

The script prints the comparison against results_backup_before_tfix/ if
that backup exists. Close but not exact means a detail of the original
differs (e.g. how the A/B halves were drawn); the conclusion would stand but
the paper's numbers should be updated to this script's output.
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


def read(name: str, label: str) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("label") or "").strip() != label:
                continue
            if len((r.get("body") or "").strip()) < 30:
                continue
            rows.append(r)
    return rows


def view_body(r) -> str:
    return r.get("body", "") or ""


def separate(a: list[dict], b: list[dict], seed: int) -> float:
    """Mean 5-fold CV accuracy for telling corpus A from corpus B (body only)."""
    rng = random.Random(seed)
    n = min(len(a), len(b), N_PER_SIDE)
    A = rng.sample(a, n)
    B = rng.sample(b, n)
    X = [view_body(r) for r in A] + [view_body(r) for r in B]
    y = [0] * n + [1] * n
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                  ngram_range=(1, 2), min_df=2,
                                  max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    return float(np.mean(cross_val_score(pipe, X, y, cv=5, scoring="accuracy")))


def ci(vals: list[float]) -> tuple[float, float]:
    m = float(np.mean(vals))
    h = T_CRIT_95 * float(np.std(vals, ddof=1)) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return m, h


def main():
    sa_ham = read("SpamAssasin.csv", "0")
    sa_spam = read("SpamAssasin.csv", "1")
    ce_ham = read("CEAS_08.csv", "0")
    ce_spam = read("CEAS_08.csv", "1")
    if not (sa_ham and sa_spam and ce_ham and ce_spam):
        print("Need SpamAssasin.csv and CEAS_08.csv in the data folder."); sys.exit(1)

    # Same-corpus null: split SpamAssassin ham at random. Must be chance.
    # Mirrors corpus_identity.py's Nazario split (fixed seed 99, shuffle, halve).
    rng = random.Random(99)
    shuf = sa_ham[:]
    rng.shuffle(shuf)
    half = len(shuf) // 2
    sa_a, sa_b = shuf[:half], shuf[half:]

    pairs = [
        ("SpamAssassin-ham vs CEAS-ham (both LEGITIMATE)", sa_ham, ce_ham),
        ("SpamAssassin-spam vs CEAS-spam (both MALICIOUS)", sa_spam, ce_spam),
        ("SpamAssassin-ham A vs B (same corpus)", sa_a, sa_b),
    ]

    print("=" * 78)
    print("  LABEL-INVARIANT SEPARABILITY (body text, 800/side, 5-fold CV, 5 seeds)")
    print("=" * 78)

    prior_path = Path(__file__).resolve().parent.parent / "results_backup_before_tfix" / "label_invariant_control.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}

    out = {}
    for name, a, b in pairs:
        scores = [separate(a, b, s) for s in SEEDS]
        m, h = ci(scores)
        # Same JSON shape as the original file: mean and ci95 in percent,
        # per-seed accuracies as fractions.
        out[name] = {"mean": 100 * m, "ci95": 100 * h, "seeds": scores}
        print(f"\n  {name}")
        print(f"    accuracy {100*m:.2f} ± {100*h:.2f}")
        print(f"    seeds    {' '.join(f'{x:.6f}' for x in scores)}")
        if name in prior:
            old = prior[name]["seeds"]
            exact = all(abs(x - y) < 1e-9 for x, y in zip(scores, old))
            maxd = max(abs(x - y) for x, y in zip(scores, old))
            print(f"    original {' '.join(f'{x:.6f}' for x in old)}")
            print(f"    -> {'EXACT REPRODUCTION' if exact else f'differs (max per-seed gap {100*maxd:.2f} points)'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "label_invariant_control.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {RESULTS_DIR / 'label_invariant_control.json'}")


if __name__ == "__main__":
    main()
