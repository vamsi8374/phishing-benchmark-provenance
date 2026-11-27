"""
Near-duplicate check for the source-exposure sweep.

    python3 experiments/sweep_near_duplicate_check.py

WHY
---
The sweep adds k benchmark *development* messages per class to a pool built
from other corpora and scores on the benchmark *test* split. The split was
de-duplicated on a hash of sender, subject and body prefix, and the sweep
script verifies zero exact body matches between dev and test. Campaign
templates can still produce near-identical messages that a prefix hash
misses. If the steep rise at small k came from such near-copies, the sweep
would be measuring memorisation of specific messages rather than familiarity
with the collections.

WHAT
----
1. Count test messages that have a near-duplicate in dev (word 5-shingle
   Jaccard >= 0.8; 0.5 reported for sensitivity), per class.
2. Remove every dev message with a near-duplicate in test (>= 0.5, the
   stricter cut) and rerun the sweep points k = 50, 100, 200 and the full
   remaining dev partition, for the pooled+dev and dev-only arms, with the
   same seeds and sampling order as pooled_indistribution_sweep.py.

Writes results/sweep_near_duplicate_check.json.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "phishguard_dataset_builder"
RESULTS = ROOT / "results"
csv.field_size_limit(10_000_000)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SEEDS = [0, 1, 2, 3, 4]
T975_4 = 2.776
PER_CORPUS = 700


def clean(s):
    return CTRL.sub(" ", s or "")


def bodies(fn, lab):
    out = []
    with open(DATA / fn, encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("label") or "").strip() != lab:
                continue
            b = clean(r.get("body", "")).strip()
            if len(b) >= 30:
                out.append(b)
    return out


def split(fn):
    with open(DATA / fn, encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return ([clean(r["body"]).strip() for r in rows],
            [1 if r["label"].strip() == "1" else 0 for r in rows])


def pipe():
    return Pipeline([
        ("t", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                              ngram_range=(1, 2), min_df=2, max_features=200_000)),
        ("c", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])


def f1(yt, yp):
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 == p)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def max_jaccard(src, tgt):
    """For each message in tgt, its maximum 5-shingle Jaccard to any message in src."""
    vec = CountVectorizer(analyzer="word", ngram_range=(5, 5), binary=True,
                          lowercase=True, token_pattern=r"(?u)\b\w+\b",
                          dtype=np.int32)
    vec.fit(src + tgt)
    A, B = vec.transform(tgt), vec.transform(src)
    a, b = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    best = np.zeros(A.shape[0])
    inter = (A @ B.T).tocoo()
    union = a[inter.row] + b[inter.col] - inter.data
    jac = np.divide(inter.data, union, out=np.zeros_like(union, dtype=float),
                    where=union > 0)
    np.maximum.at(best, inter.row, jac)
    return best


def summ(v):
    v = np.array(v) * 100
    return {"mean": float(v.mean()), "ci95_t": float(T975_4 * v.std(ddof=1) / len(v) ** 0.5),
            "per_seed": [float(x) for x in v]}


def main():
    pool = {}
    for fn, nm in (("CEAS_08.csv", "CEAS"), ("Ling.csv", "Ling"), ("Enron.csv", "Enron")):
        ph, hm = bodies(fn, "1"), bodies(fn, "0")
        if len(ph) >= 300 and len(hm) >= 300:
            pool[nm] = (ph, hm)
    dev_ph, dev_hm = bodies("phishguard_dev.csv", "1"), bodies("phishguard_dev.csv", "0")
    Xt, yt = split("phishguard_test.csv")
    test_ph = [b for b, y in zip(Xt, yt) if y == 1]
    test_hm = [b for b, y in zip(Xt, yt) if y == 0]

    out = {"config": {"shingle": "word 5-gram", "seeds": SEEDS, "t_crit_95": T975_4}}
    # 1. how many test messages have a near-copy in dev?
    counts = {}
    for cls, dv, ts in (("phishing", dev_ph, test_ph), ("ham", dev_hm, test_hm)):
        mj = max_jaccard(dv, ts)
        counts[cls] = {"n_test": len(ts),
                       "test_with_dev_neighbour_ge_0.8": int((mj >= 0.8).sum()),
                       "test_with_dev_neighbour_ge_0.5": int((mj >= 0.5).sum())}
    out["test_near_duplicates_in_dev"] = counts
    print("Near-duplicates (test message has a dev neighbour):")
    for c, v in counts.items():
        print(f"  {c:<9} n={v['n_test']}  J>=0.8: {v['test_with_dev_neighbour_ge_0.8']}"
              f"   J>=0.5: {v['test_with_dev_neighbour_ge_0.5']}")

    # 2. remove dev messages with any test neighbour at J >= 0.5 and rerun
    keep_ph = max_jaccard(Xt, dev_ph) < 0.5
    keep_hm = max_jaccard(Xt, dev_hm) < 0.5
    dph = [b for b, k in zip(dev_ph, keep_ph) if k]
    dhm = [b for b, k in zip(dev_hm, keep_hm) if k]
    out["dev_after_filter"] = {"phishing": len(dph), "ham": len(dhm),
                               "removed_phishing": len(dev_ph) - len(dph),
                               "removed_ham": len(dev_hm) - len(dhm)}
    print(f"\nDev after removing messages with a test neighbour (J>=0.5): "
          f"{len(dph)} phishing / {len(dhm)} ham "
          f"(removed {len(dev_ph)-len(dph)} / {len(dev_hm)-len(dhm)})")

    kmax = min(len(dph), len(dhm))
    res = {}
    for label, dp, dh, ks in (("original_dev", dev_ph, dev_hm, [50, 100, 200, 465]),
                              ("filtered_dev", dph, dhm, [50, 100, 200, kmax])):
        res[label] = {}
        for k in ks:
            a, b = [], []
            for s in SEEDS:
                rng = random.Random(s)
                Xp, yp = [], []
                for nm, (ph, hm) in pool.items():
                    m = min(PER_CORPUS, len(ph), len(hm))
                    Xp += rng.sample(ph, m); yp += [1] * m
                    Xp += rng.sample(hm, m); yp += [0] * m
                Xp += rng.sample(dp, k); yp += [1] * k
                Xp += rng.sample(dh, k); yp += [0] * k
                M = pipe(); M.fit(Xp, yp)
                a.append(f1(yt, list(M.predict(Xt))))
                rng2 = random.Random(1000 + s)
                D = pipe(); D.fit(rng2.sample(dp, k) + rng2.sample(dh, k), [1] * k + [0] * k)
                b.append(f1(yt, list(D.predict(Xt))))
            res[label][f"k{k}"] = {"pooled_plus_dev": summ(a), "dev_only": summ(b)}
            print(f"  {label:<13} k={k:<4} pooled+dev F1 {100*np.mean(a):6.2f}"
                  f"   dev-only F1 {100*np.mean(b):6.2f}")
    out["sweep"] = res
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sweep_near_duplicate_check.json").write_text(json.dumps(out, indent=1))
    print("\n  wrote results/sweep_near_duplicate_check.json")


if __name__ == "__main__":
    main()
