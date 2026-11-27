"""
Two gaps closed at once.

    python experiments/generalize.py

GAP 1 — the claim rested on one benchmark pairing.
We repeat the provenance-transfer test on several benchmarks composed from
different source combinations. If provenance transfer succeeds across all of
them, the finding is about the *construction method*, not about one dataset.

GAP 2 — the label-invariant control had a temporal confound.
SpamAssassin ham (2002) and CEAS ham (2008) sit six years apart, so part of
their 95.19% separability might be genuine drift in how people write e-mail
rather than a collection artefact. We cannot obtain a same-year ham pair from
public corpora, but we can bound the drift term directly: split ONE corpus by
year and measure how separable consecutive years are within it. Whatever that
number is, it is an upper estimate of how much of the cross-corpus figure
plain drift could explain.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR, RESULTS_DIR  # noqa: E402

csv.field_size_limit(10_000_000)

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402
from sklearn.model_selection import cross_val_score          # noqa: E402
import numpy as np                                           # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

# 95% CI half-width. With n=5 seeds the sampling distribution of the mean
# is Student's t with 4 df, so the critical value is 2.776, not the
# normal-approximation 1.96. Update this if SEEDS changes length.
T_CRIT_95 = 2.776

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
YEAR = re.compile(r"\b(19|20)\d{2}\b")
BENIGN, PHISH = 0, 1


def clean(s: str) -> str:
    return CTRL.sub(" ", s or "")


def read(name: str, label: str) -> list[dict]:
    p = DATA_DIR / name
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("label") or "").strip() != label:
                continue
            b = clean(r.get("body", "")).strip()
            if len(b) < 30:
                continue
            m = YEAR.search(r.get("date", "") or "")
            out.append({"body": b, "year": int(m.group(0)) if m else None})
    return out


def pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                  ngram_range=(1, 2), min_df=2,
                                  max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced")),
    ])


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(y_true) if y_true else 0.0}


def agg(vals):
    m = float(np.mean(vals))
    h = T_CRIT_95 * float(np.std(vals, ddof=1)) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return m, h


# ── Gap 1: provenance transfer across several composed benchmarks ────────────
def transfer_once(prov_a, prov_b, bench_ph, bench_ham, seed, n_train=1200,
                  n_test=900, shuffle=False):
    """prov_a plays 'the ham source'; prov_b plays 'not the ham source'."""
    rng = random.Random(seed)
    n = min(len(prov_a), len(prov_b), n_train)
    X = [r["body"] for r in rng.sample(prov_a, n)] + \
        [r["body"] for r in rng.sample(prov_b, n)]
    y = [0] * n + [1] * n
    if shuffle:
        rng.shuffle(y)

    P = pipe()
    P.fit(X, y)

    m = min(len(bench_ph), len(bench_ham), n_test)
    bx = [r["body"] for r in rng.sample(bench_ph, m)] + \
         [r["body"] for r in rng.sample(bench_ham, m)]
    by = [PHISH] * m + [BENIGN] * m
    pred = [BENIGN if p == 0 else PHISH for p in P.predict(bx)]
    return metrics(by, pred)


def main():
    sa_ham = read("SpamAssasin.csv", "0")
    sa_spam = read("SpamAssasin.csv", "1")
    ce_ham = read("CEAS_08.csv", "0")
    ce_spam = read("CEAS_08.csv", "1")
    naz = read("Nazario.csv", "1")
    nig = read("Nigerian_Fraud.csv", "1")

    missing = [n for n, v in [("SpamAssasin", sa_ham), ("CEAS_08", ce_ham),
                              ("Nazario", naz), ("Nigerian_Fraud", nig)] if not v]
    if missing:
        print(f"❌  Missing corpora: {missing}")
        sys.exit(1)

    print("=" * 92)
    print("  GAP 1 — DOES PROVENANCE TRANSFER GENERALISE ACROSS BENCHMARKS?")
    print("  Each row is a different benchmark built from different sources.")
    print("  The detector is trained only on the two PROVENANCE corpora and")
    print("  never sees a phishing label.")
    print("=" * 92)
    print()

    # (benchmark name, phishing side, ham side, provenance-A = ham source,
    #  provenance-B = a stand-in for 'not the ham source')
    setups = [
        ("Nazario / SpamAssassin-ham", naz,     sa_ham, sa_ham, ce_ham),
        ("Nigerian / SpamAssassin-ham", nig,    sa_ham, sa_ham, ce_ham),
        ("Nazario / CEAS-ham",          naz,    ce_ham, ce_ham, sa_ham),
        ("Nigerian / CEAS-ham",         nig,    ce_ham, ce_ham, sa_ham),
        ("SpamAssassin-spam / CEAS-ham", sa_spam, ce_ham, ce_ham, sa_ham),
    ]

    hdr = (f"{'benchmark':<32}{'transfer F1%':>16}{'null F1%':>14}"
           f"{'prec%':>10}{'rec%':>10}")
    print(hdr)
    print("-" * len(hdr))

    out = {"generalisation": {}, "drift_bound": {}}
    for name, ph, ham, pa, pb in setups:
        real = [transfer_once(pa, pb, ph, ham, s) for s in SEEDS]
        null = [transfer_once(pa, pb, ph, ham, s, shuffle=True) for s in SEEDS]
        f1m, f1h = agg([r["f1"] for r in real])
        nlm, nlh = agg([r["f1"] for r in null])
        pm, _ = agg([r["precision"] for r in real])
        rm, _ = agg([r["recall"] for r in real])
        print(f"{name:<32}{100*f1m:>9.2f}±{100*f1h:<5.2f}"
              f"{100*nlm:>8.2f}±{100*nlh:<4.2f}{100*pm:>10.2f}{100*rm:>10.2f}")
        out["generalisation"][name] = {
            "transfer_f1": {"mean": f1m, "ci95": f1h},
            "null_f1": {"mean": nlm, "ci95": nlh},
            "precision": pm, "recall": rm,
        }

    print()
    print("  Every one of these benchmarks is composed by pairing two")
    print("  separately-collected corpora. If transfer succeeds on all of")
    print("  them, the finding concerns the construction method rather than")
    print("  any single dataset.")

    # ── Gap 2: bound the drift term ──────────────────────────────────────────
    print()
    print("=" * 92)
    print("  GAP 2 — HOW MUCH OF CROSS-CORPUS SEPARABILITY COULD BE PLAIN DRIFT?")
    print("  Within ONE corpus, how separable are different collection years?")
    print("  This upper-bounds the drift contribution to the 95.19% figure.")
    print("=" * 92)
    print()

    def year_pairs(rows, label):
        by_year = {}
        for r in rows:
            if r["year"]:
                by_year.setdefault(r["year"], []).append(r)
        usable = {y: v for y, v in by_year.items() if len(v) >= 200}
        if len(usable) < 2:
            print(f"  {label}: fewer than two years with >=200 messages; skipped")
            return
        ys = sorted(usable)
        for a, b in zip(ys, ys[1:]):
            A, B = usable[a], usable[b]
            scores = []
            for s in SEEDS:
                rng = random.Random(s)
                n = min(len(A), len(B), 600)
                X = [r["body"] for r in rng.sample(A, n)] + \
                    [r["body"] for r in rng.sample(B, n)]
                y = [0] * n + [1] * n
                scores.append(float(np.mean(
                    cross_val_score(pipe(), X, y, cv=5, scoring="accuracy"))))
            m, h = agg(scores)
            print(f"  {label}: {a} vs {b}  (n={min(len(A),len(B),600)}/side)"
                  f"   accuracy {100*m:.2f}±{100*h:.2f}")
            out["drift_bound"][f"{label} {a}v{b}"] = {"mean": m, "ci95": h}

    year_pairs(sa_ham, "SpamAssassin ham")
    year_pairs(naz, "Nazario phishing")

    print()
    print("  Read against the 95.19% cross-corpus figure. Within-corpus")
    print("  year-to-year separability is the most drift alone can buy;")
    print("  the remainder is collection artefact.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "generalize.json").write_text(json.dumps(out, indent=2))
    print()
    print("✅  generalize.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
