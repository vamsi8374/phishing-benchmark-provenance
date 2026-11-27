"""
Testing our own explanation for the Ling-Spam failure.

    python experiments/why_ling_fails.py

THE CLAIM UNDER TEST
--------------------
We asserted that provenance transfer fails with Ling-Spam as the stand-in
corpus because Ling is "narrow" -- one linguistics mailing list -- so the
"not SpamAssassin" region it induces does not contain the phishing
distribution. That was a hypothesis stated as fact, and it conflates two
distinct explanations:

  (H1) SIZE.     Ling has 2,399 usable messages; Enron has 15,669. Maybe the
                 detector is simply undertrained.
  (H2) BREADTH.  Ling is one mailing list on one topic; Enron is heterogeneous
                 corporate mail. Maybe coverage, not count, is what matters.

These make opposite predictions under a controlled test. Subsample Enron down
to Ling's exact size and re-run:

  If transfer still succeeds at n=2,399  -> size is not the cause; H2 holds.
  If transfer collapses at n=2,399       -> H1 holds and our stated
                                            precondition is wrong: it should
                                            say "large enough", not "broad
                                            enough".

We also measure, rather than assert, (a) lexical breadth of each corpus and
(b) where the failing detector actually puts Nazario, with what confidence.
An out-of-distribution input should produce LOW-confidence predictions; a
confidently wrong detector would indicate something other than OOD.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR, RESULTS_DIR, TEST_CSV, PHISHGUARD_DIR  # noqa: E402

csv.field_size_limit(10_000_000)
sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402
import numpy as np                                           # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

# 95% CI half-width. With n=5 seeds the sampling distribution of the mean
# is Student's t with 4 df, so the critical value is 2.776, not the
# normal-approximation 1.96. Update this if SEEDS changes length.
T_CRIT_95 = 2.776

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
TOK = re.compile(r"[a-z]{2,}")
BENIGN, PHISH = 0, 1
LING_N = 2399          # Ling's usable ham count, from independence.py


def clean(s):
    return CTRL.sub(" ", s or "")


def bodies(name, label):
    p = DATA_DIR / name
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("label") or "").strip() != label:
                continue
            b = clean(r.get("body", "")).strip()
            if len(b) >= 30:
                out.append(b)
    return out


def pipe():
    return Pipeline([
        ("v", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                              ngram_range=(1, 2), min_df=2,
                              max_features=200_000)),
        ("c", LogisticRegression(max_iter=2000, C=4.0,
                                 class_weight="balanced")),
    ])


def f1_of(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def agg(v):
    m = float(np.mean(v))
    h = T_CRIT_95 * float(np.std(v, ddof=1)) / (len(v) ** 0.5) if len(v) > 1 else 0.0
    return m, h


def breadth(docs, sample=2000, seed=0):
    """Lexical breadth: distinct types per 1k tokens, over a fixed sample."""
    rng = random.Random(seed)
    d = rng.sample(docs, min(len(docs), sample))
    types, tokens = set(), 0
    for doc in d:
        t = TOK.findall(doc.lower())
        tokens += len(t)
        types.update(t)
    return {"n_docs": len(d), "types": len(types), "tokens": tokens,
            "types_per_1k_tokens": 1000.0 * len(types) / tokens if tokens else 0.0}


def transfer(stand_in, sa_ham, Xt, yt, n_side, seed):
    rng = random.Random(seed)
    n = min(len(sa_ham), len(stand_in), n_side)
    X = rng.sample(sa_ham, n) + rng.sample(stand_in, n)
    y = [0] * n + [1] * n
    P = pipe(); P.fit(X, y)
    pred = [BENIGN if p == 0 else PHISH for p in P.predict(Xt)]
    return f1_of(yt, pred), P


def main():
    sa_ham = bodies("SpamAssasin.csv", "0")
    ling = bodies("Ling.csv", "0")
    enron = bodies("Enron.csv", "0")
    ceas = bodies("CEAS_08.csv", "0")
    naz = bodies("Nazario.csv", "1")

    test = load_emails(str(TEST_CSV))
    Xt = [clean(r.body) for r in test]
    yt = [PHISH if r.label == "phishing" else BENIGN for r in test]

    out = {}

    # ── The decisive test: size or breadth? ─────────────────────────────────
    print("=" * 88)
    print("  SIZE OR BREADTH? Enron subsampled to Ling's exact size.")
    print("=" * 88)
    print()
    hdr = f"{'stand-in corpus':<34}{'n/side':>9}{'transfer F1%':>16}"
    print(hdr); print("-" * len(hdr))

    conds = [
        ("Ling ham (full)",              ling,  min(len(ling), 1200)),
        ("Enron ham (full)",             enron, 1200),
        ("Enron ham, subsampled to Ling", enron, min(LING_N, 1200)),
    ]
    # Use an identical per-side cap so the comparison is like-for-like.
    cap = min(len(ling), 1200)
    conds = [("Ling ham",                       ling,  cap),
             ("Enron ham, capped to Ling's n",  enron, cap),
             ("Enron ham, full budget",         enron, 1200),
             ("CEAS ham, capped to Ling's n",   ceas,  cap)]

    res = {}
    for nm, corpus, n in conds:
        if len(corpus) < n:
            print(f"{nm:<34}{'insufficient':>25}")
            continue
        scores = [transfer(corpus, sa_ham, Xt, yt, n, s)[0] for s in SEEDS]
        m, h = agg(scores)
        print(f"{nm:<34}{n:>9}{100*m:>10.2f}±{100*h:<5.2f}")
        res[nm] = {"n_per_side": n, "f1": {"mean": m, "ci95": h}}
    out["size_vs_breadth"] = res

    print()
    print("  Ling and the capped Enron condition use the SAME number of")
    print("  training messages per side. Any difference between them cannot")
    print("  be a sample-size effect.")

    # ── Lexical breadth ─────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("  LEXICAL BREADTH (measured, not asserted)")
    print("=" * 88)
    print()
    hdr2 = f"{'corpus':<22}{'docs':>8}{'types':>10}{'tokens':>11}{'types/1k tok':>14}"
    print(hdr2); print("-" * len(hdr2))
    br = {}
    for nm, c in (("Ling ham", ling), ("Enron ham", enron),
                  ("CEAS ham", ceas), ("SpamAssassin ham", sa_ham),
                  ("Nazario phishing", naz)):
        b = breadth(c)
        print(f"{nm:<22}{b['n_docs']:>8}{b['types']:>10}{b['tokens']:>11}"
              f"{b['types_per_1k_tokens']:>14.2f}")
        br[nm] = b
    out["breadth"] = br

    # ── Where does the failing detector put Nazario, and how surely? ───────
    print()
    print("=" * 88)
    print("  WHERE DOES THE FAILING DETECTOR PUT NAZARIO?")
    print("  OOD inputs should attract LOW-confidence predictions. A")
    print("  confidently wrong detector would mean something other than OOD.")
    print("=" * 88)
    print()
    hdr3 = f"{'stand-in':<20}{'% Nazario -> ham side':>24}{'mean |p-0.5|':>15}"
    print(hdr3); print("-" * len(hdr3))
    conf = {}
    for nm, corpus in (("Ling ham", ling), ("Enron ham", enron),
                       ("CEAS ham", ceas)):
        pcts, margins = [], []
        for s in SEEDS:
            rng = random.Random(s)
            n = min(len(sa_ham), len(corpus), cap)
            X = rng.sample(sa_ham, n) + rng.sample(corpus, n)
            y = [0] * n + [1] * n
            P = pipe(); P.fit(X, y)
            sub = rng.sample(naz, min(len(naz), 900))
            pr = P.predict(sub)
            pp = P.predict_proba(sub)[:, 1]
            pcts.append(100.0 * float(np.mean([p == 0 for p in pr])))
            margins.append(float(np.mean(np.abs(pp - 0.5))))
        pm, ph = agg(pcts); mm, mh = agg(margins)
        print(f"{nm:<20}{pm:>18.1f}±{ph:<4.1f}{mm:>13.3f}±{mh:<4.3f}")
        conf[nm] = {"pct_nazario_to_ham_side": {"mean": pm, "ci95": ph},
                    "mean_abs_margin": {"mean": mm, "ci95": mh}}
    out["nazario_placement"] = conf

    print()
    print("  'ham side' = the class the benchmark's ham source occupies, so")
    print("  Nazario landing there is what produces the transfer failure.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "why_ling_fails.json").write_text(json.dumps(out, indent=2))
    print()
    print("✅  why_ling_fails.json")
    print("=" * 88)


if __name__ == "__main__":
    main()
