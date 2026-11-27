"""
Is the pooled-vs-provenance comparison fair? An in-distribution control.

    python experiments/pooled_indistribution_sweep.py

WHY THIS EXPERIMENT
-------------------
`does_pooling_fix_it.py` reports that a provenance-only detector outscores the
pooled multi-corpus model by ~25 F1 points on the benchmark, and the paper
reads that as the benchmark *inverting* model ordering.

There is an objection to that reading, and it is the first one a reviewer will
raise. The two models do not have equal access to the benchmark:

    pooled model      trained on CEAS + Ling + Enron
                      -> never sees SpamAssassin or Nazario
                      -> out-of-distribution on BOTH test classes

    provenance model  trained on SpamAssassin-ham vs CEAS-ham
                      -> SpamAssassin IS the benchmark's ham class
                      -> in-distribution on HALF the test set

So some unknown share of the 25-point gap is simply in-distribution access,
not "provenance beats phishing". Cohen's kappa = 0.490 shows the two models
are different functions; it does NOT show the comparison was fair.

WHAT THIS DOES
--------------
Rather than defend against the objection, absorb it into the result. We give
the pooled model a controlled amount of the benchmark's own provenance and
watch what happens, by adding k messages per class from the benchmark's own
DEV partition (phishguard_dev.csv) to the pool, then scoring on the unchanged
test split.

Three model families per (k, seed):

    A. pooled + k dev/class   the fair control -- the curve we care about
    B. dev-only, k/class      how much comes from benchmark data ALONE
    C. provenance-only        the reference line (unchanged from the paper)

HOW TO READ THE RESULT
----------------------
If A rises steeply with k and approaches C (~98.45):
    the 25-point gap is substantially an in-distribution-access effect. The
    claim "the benchmark inverts model ordering" must be restated as something
    like "the benchmark rewards familiarity with its own corpora rather than
    phishing-relevant learning" -- which is a weaker headline but a cleaner and
    more defensible one, and it still supports the paper's thesis.

If A stays well below C even at the largest k:
    the inversion is robust to equalised access, the original claim stands,
    and this experiment becomes the control that defends it.

Either outcome is publishable. The design choice is that we report the curve,
not a single number, so the reader can see the effect size rather than trust
our framing of it.

DESIGN NOTES (push back on any of these)
----------------------------------------
* k is capped at 465 because the dev partition holds exactly 465 messages per
  class (930 rows; test holds 2,170; 930+2,170 = 3,100, the paper's balanced
  total). An earlier draft of this sweep used k=700, which does not exist.
* The pool contributes 700/class/corpus x 3 corpora = 2,100 per class, so even
  at k=465 the dev data is ~18% of each class -- meaningful but not dominant.
* The clean pool sample is drawn with the same seed and the same `per` as
  `does_pooling_fix_it.py`, so k=0 should REPRODUCE that script's 73.48. The
  script checks this and complains if it does not; if k=0 disagrees, something
  differs between the two setups and the sweep is not comparable.
* Dev/test disjointness is verified rather than assumed, and reported.
* Per-seed values are written into the JSON, so the confidence interval can be
  recomputed later without re-running anything.
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

# Dev partition holds 465 per class; 465 is therefore "all available".
K_SWEEP = [0, 50, 100, 200, 300, 465]
PER_CORPUS = 700          # matches does_pooling_fix_it.py
DEV_CSV = DATA_DIR / "phishguard_dev.csv"

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BENIGN, PHISH = 0, 1


def clean(s):
    return CTRL.sub(" ", s or "")


def rows(name):
    p = DATA_DIR / name
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def bodies(name, label):
    """Bodies for one label, same >=30 char filter as the rest of the suite."""
    out = []
    for r in rows(name):
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


def main():
    # ── corpora ───────────────────────────────────────────────────────────
    pool_sources = [("CEAS_08.csv", "CEAS"), ("Ling.csv", "Ling"),
                    ("Enron.csv", "Enron")]
    pool = {}
    for fn, nm in pool_sources:
        ph, hm = bodies(fn, "1"), bodies(fn, "0")
        if len(ph) >= 300 and len(hm) >= 300:
            pool[nm] = (ph, hm)
    if len(pool) < 2:
        print("Need at least two pool corpora with both classes."); sys.exit(1)

    dev_ph, dev_hm = bodies("phishguard_dev.csv", "1"), bodies("phishguard_dev.csv", "0")
    sa_ham, ce_ham = bodies("SpamAssasin.csv", "0"), bodies("CEAS_08.csv", "0")

    test = load_emails(str(TEST_CSV))
    Xt = [clean(r.body) for r in test]
    yt = [PHISH if r.label == "phishing" else BENIGN for r in test]

    # ── disjointness check: verified, not assumed ─────────────────────────
    tset = set(Xt)
    leak = sum(1 for b in dev_ph + dev_hm if b in tset)
    print("=" * 78)
    print("  IN-DISTRIBUTION CONTROL FOR THE POOLED COMPARISON")
    print(f"  pool: {sorted(pool)}   dev available: {len(dev_ph)} phish / {len(dev_hm)} ham")
    print(f"  test n={len(yt)}   exact body matches between dev and test: {leak}")
    if leak:
        print("  !! dev and test share bodies -- investigate before trusting the curve")
    print("=" * 78)

    max_k = min(len(dev_ph), len(dev_hm))
    ks = [k for k in K_SWEEP if k <= max_k]
    if ks != K_SWEEP:
        print(f"  note: k capped at {max_k} by dev size; sweeping {ks}")

    results = {}

    # ── C. provenance-only reference line ─────────────────────────────────
    prov_runs = []
    for s in SEEDS:
        rng = random.Random(s)
        n = min(len(sa_ham), len(ce_ham), 1500)
        Xq = rng.sample(sa_ham, n) + rng.sample(ce_ham, n)
        yq = [0]*n + [1]*n
        P = pipe(); P.fit(Xq, yq)
        pred = [BENIGN if p == 0 else PHISH for p in P.predict(Xt)]
        prov_runs.append(metrics(yt, pred))
    results["provenance_only"] = prov_runs

    # ── A and B across the sweep ──────────────────────────────────────────
    for k in ks:
        a_runs, b_runs = [], []
        for s in SEEDS:
            rng = random.Random(s)

            # A: pooled + k dev per class. The clean pool is drawn FIRST with
            # this seed so that at k=0 the sample is identical to the one in
            # does_pooling_fix_it.py.
            Xp, yp = [], []
            for nm, (ph, hm) in pool.items():
                m = min(PER_CORPUS, len(ph), len(hm))
                Xp += rng.sample(ph, m); yp += [PHISH]*m
                Xp += rng.sample(hm, m); yp += [BENIGN]*m
            if k:
                Xp += rng.sample(dev_ph, k); yp += [PHISH]*k
                Xp += rng.sample(dev_hm, k); yp += [BENIGN]*k
            M = pipe(); M.fit(Xp, yp)
            a_runs.append(metrics(yt, list(M.predict(Xt))))

            # B: dev-only at the same k -- how much comes from benchmark data
            # alone, with no pool at all.
            if k:
                rng2 = random.Random(1000 + s)
                Xd = rng2.sample(dev_ph, k) + rng2.sample(dev_hm, k)
                yd = [PHISH]*k + [BENIGN]*k
                D = pipe(); D.fit(Xd, yd)
                b_runs.append(metrics(yt, list(D.predict(Xt))))

        results[f"pooled_plus_dev_k{k}"] = a_runs
        if b_runs:
            results[f"dev_only_k{k}"] = b_runs

    # ── report ────────────────────────────────────────────────────────────
    print()
    hdr = f"{'model':<34}{'F1%':>14}{'MCC':>9}"
    print(hdr); print("-" * len(hdr))
    for k in ks:
        f1m, f1h = agg([r["f1"] for r in results[f"pooled_plus_dev_k{k}"]])
        mc, _ = agg([r["mcc"] for r in results[f"pooled_plus_dev_k{k}"]])
        tag = "pooled (no dev)" if k == 0 else f"pooled + {k}/class dev"
        print(f"{tag:<34}{100*f1m:>9.2f}±{100*f1h:<4.2f}{mc:>9.3f}")
    print("-" * len(hdr))
    for k in ks:
        key = f"dev_only_k{k}"
        if key not in results:
            continue
        f1m, f1h = agg([r["f1"] for r in results[key]])
        print(f"{'dev-only ' + str(k) + '/class':<34}{100*f1m:>9.2f}±{100*f1h:<4.2f}")
    print("-" * len(hdr))
    pf, ph_ = agg([r["f1"] for r in prov_runs])
    pm, _ = agg([r["mcc"] for r in prov_runs])
    print(f"{'provenance-only (reference)':<34}{100*pf:>9.2f}±{100*ph_:<4.2f}{pm:>9.3f}")

    # sanity: k=0 must reproduce does_pooling_fix_it.py
    k0, _ = agg([r["f1"] for r in results["pooled_plus_dev_k0"]])
    prior = RESULTS_DIR / "does_pooling_fix_it.json"
    if prior.exists():
        want = json.loads(prior.read_text())["pooled"]["f1"]["mean"]
        delta = abs(k0 - want)
        flag = "OK" if delta < 0.01 else "MISMATCH -- setups differ, do not compare"
        print(f"\n  sanity: k=0 F1 {100*k0:.2f} vs does_pooling_fix_it {100*want:.2f}"
              f"  (delta {100*delta:.2f}) [{flag}]")

    print("\n  HOW TO READ THIS")
    print("  If the pooled curve climbs toward the provenance line as k grows,")
    print("  the 25-point gap was substantially in-distribution access and the")
    print("  'inverts model ordering' claim needs restating. If it stays flat")
    print("  and low, the inversion survives an equalised-access control.")

    out = {
        "pool_corpora": sorted(pool),
        "n_test": len(yt),
        "dev_available_per_class": {"phish": len(dev_ph), "ham": len(dev_hm)},
        "dev_test_exact_body_overlap": leak,
        "k_sweep": ks,
        "per_corpus_per_class": PER_CORPUS,
        "seeds": SEEDS,
        "t_crit_95": T_CRIT_95,
        "per_seed": {name: runs for name, runs in results.items()},
        "summary": {
            name: {m: dict(zip(("mean", "ci95"), agg([r[m] for r in runs])))
                   for m in ("f1", "precision", "recall", "accuracy", "mcc")}
            for name, runs in results.items()
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pooled_indistribution_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {RESULTS_DIR / 'pooled_indistribution_sweep.json'}")


if __name__ == "__main__":
    main()
