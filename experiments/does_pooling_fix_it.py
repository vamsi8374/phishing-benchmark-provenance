"""
Does pooled multi-corpus training actually remove the confound?

    python experiments/does_pooling_fix_it.py

WHY THIS EXPERIMENT
-------------------
Recent work (Fragility of Phishing Detection Models, BDCC 10(7):211, 2026)
establishes that corpus identity is highly learnable across the standard public
phishing corpora, and recommends *pooled multi-corpus training* --- in
particular leave-one-corpus-out pooling --- as a mitigation.

We test whether that remedy works, and we argue it cannot, for a reason that
is structural rather than empirical:

    Pooling is a TRAINING-side intervention.
    Provenance confounding is an EVALUATION-side property.

A pooled model may well be more robust. But if it is then *scored on a
confounded benchmark*, the resulting number is still uninterpretable, because
the benchmark cannot distinguish a model that learned phishing from one that
learned provenance. Fixing how you train does not fix what your test set can
measure.

THE TEST
--------
1. Build a POOLED training set from corpora that each contribute BOTH classes
   (CEAS, Ling, Enron). Within this pool, corpus identity does NOT determine
   the label, so the model cannot learn provenance-as-label from training.

2. Evaluate that pooled model on the confounded Nazario/SpamAssassin
   benchmark.

3. Evaluate the PROVENANCE-ONLY detector (never saw a phishing label) on the
   same benchmark.

4. Compare. If both score highly and agree with each other, then the benchmark
   cannot separate them --- and pooling, whatever else it achieves, has not
   made the evaluation valid.

We also report the pooled model's reliance on the Date channel, to check
whether pooling reduces artifact dependence at test time.
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
YEAR = re.compile(r"\b(19|20)\d{2}\b")
BENIGN, PHISH = 0, 1


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


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return {"f1": 2*prec*rec/(prec+rec) if (prec+rec) else 0.0,
            "precision": prec, "recall": rec,
            "accuracy": (tp+tn)/len(y_true) if y_true else 0.0,
            "mcc": ((tp*tn - fp*fn)/den) if den else 0.0}


def agg(v):
    m = float(np.mean(v))
    h = T_CRIT_95*float(np.std(v, ddof=1))/(len(v)**0.5) if len(v) > 1 else 0.0
    return m, h


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a)/n, sum(b)/n
    pe = pa*pb + (1-pa)*(1-pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    # Corpora that contribute BOTH classes -> provenance cannot equal label.
    pool_sources = [("CEAS_08.csv", "CEAS"), ("Ling.csv", "Ling"),
                    ("Enron.csv", "Enron")]
    pool = {}
    for fn, nm in pool_sources:
        ph, hm = bodies(fn, "1"), bodies(fn, "0")
        if len(ph) >= 300 and len(hm) >= 300:
            pool[nm] = (ph, hm)
    if len(pool) < 2:
        print("❌  Need at least two corpora with both classes.")
        sys.exit(1)

    sa_ham = bodies("SpamAssasin.csv", "0")
    ce_ham = bodies("CEAS_08.csv", "0")

    test = load_emails(str(TEST_CSV))
    Xt = [clean(r.body) for r in test]
    yt = [PHISH if r.label == "phishing" else BENIGN for r in test]

    print("=" * 90)
    print("  DOES POOLED MULTI-CORPUS TRAINING FIX THE CONFOUND?")
    print(f"  pool: {sorted(pool)}   (each contributes both classes)")
    print(f"  evaluated on the confounded benchmark test split (n={len(yt)})")
    print("=" * 90)
    print()

    pooled_runs, prov_runs, kappas, date_runs = [], [], [], []
    for s in SEEDS:
        rng = random.Random(s)

        # ── (1) POOLED model: multi-corpus, both classes per corpus ─────────
        Xp, yp = [], []
        per = 700
        for nm, (ph, hm) in pool.items():
            k = min(per, len(ph), len(hm))
            Xp += rng.sample(ph, k); yp += [PHISH]*k
            Xp += rng.sample(hm, k); yp += [BENIGN]*k
        M = pipe(); M.fit(Xp, yp)
        pooled_pred = list(M.predict(Xt))
        pooled_runs.append(metrics(yt, pooled_pred))

        # ── (2) PROVENANCE-only detector: no phishing label anywhere ────────
        n = min(len(sa_ham), len(ce_ham), 1500)
        Xq = rng.sample(sa_ham, n) + rng.sample(ce_ham, n)
        yq = [0]*n + [1]*n
        P = pipe(); P.fit(Xq, yq)
        prov_pred = [BENIGN if p == 0 else PHISH for p in P.predict(Xt)]
        prov_runs.append(metrics(yt, prov_pred))

        kappas.append(kappa(pooled_pred, prov_pred))

        # ── (3) does the pooled model still lean on the Date channel? ───────
        Xpd, ypd = [], []
        # rebuild the pool using the date field only
        for fn, nm in pool_sources:
            if nm not in pool:
                continue
            p = DATA_DIR / fn
            with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                rows = [r for r in csv.DictReader(fh)]
            has_date = [r for r in rows if (r.get("date") or "").strip()]
            if len(has_date) < 200:
                continue
            for r in rng.sample(has_date, min(1000, len(has_date))):
                Xpd.append(r.get("date", ""))
                ypd.append(PHISH if (r.get("label") or "").strip() == "1" else BENIGN)
        if len(set(ypd)) == 2 and len(Xpd) > 200:
            D = pipe(); D.fit(Xpd, ypd)
            date_pred = list(D.predict([r.date or "" for r in test]))
            date_runs.append(metrics(yt, date_pred))

    hdr = f"{'model':<46}{'F1%':>10}{'MCC':>9}"
    print(hdr); print("-" * len(hdr))
    for nm, runs in (("Pooled multi-corpus (the proposed remedy)", pooled_runs),
                     ("Provenance-only (never saw a phishing label)", prov_runs)):
        f1m, f1h = agg([r["f1"] for r in runs])
        mc, _ = agg([r["mcc"] for r in runs])
        print(f"{nm:<46}{100*f1m:>6.2f}±{100*f1h:<4.2f}{mc:>9.3f}")

    km, kh = agg(kappas)
    print()
    print(f"  Agreement between the two, Cohen's kappa: {km:.3f} ± {kh:.3f}")

    if date_runs:
        dm, dh = agg([r["f1"] for r in date_runs])
        print(f"  Pooled model trained on the DATE FIELD ALONE: "
              f"F1 {100*dm:.2f}% ± {100*dh:.2f}")

    print()
    print("  HOW TO READ THIS")
    print("  Pooling is a training-side fix. The benchmark is an")
    print("  evaluation-side problem. If the pooled model and a model that")
    print("  never saw a phishing label score similarly on this test set,")
    print("  then the test set cannot tell them apart --- and no amount of")
    print("  pooling makes a number computed on it interpretable.")

    out = {
        "pool_corpora": sorted(pool),
        "n_test": len(yt),
        "pooled": {k: dict(zip(("mean", "ci95"), agg([r[k] for r in pooled_runs])))
                   for k in ("f1", "precision", "recall", "accuracy", "mcc")},
        "provenance_only": {k: dict(zip(("mean", "ci95"),
                                        agg([r[k] for r in prov_runs])))
                            for k in ("f1", "precision", "recall", "accuracy", "mcc")},
        "kappa_pooled_vs_provenance": {"mean": km, "ci95": kh},
        "pooled_date_only": ({k: dict(zip(("mean", "ci95"),
                                          agg([r[k] for r in date_runs])))
                              for k in ("f1", "mcc")} if date_runs else None),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "does_pooling_fix_it.json").write_text(json.dumps(out, indent=2))
    print()
    print("✅  does_pooling_fix_it.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
