"""
Overlap audit for every result that depends on the provenance detector P.

    python3 experiments/provenance_overlap_audit.py          # all parts
    python3 experiments/provenance_overlap_audit.py --part A # one part

WHY
---
P is trained to separate SpamAssassin ham from CEAS ham. SpamAssassin also
supplies the benchmark's ham class, so benchmark test messages are eligible for
P's training pool. In the composed-benchmark table (generalize.py) the training
pool and the benchmark's ham side are sampled from the very same list. The
overlap is intrinsic to the construction -- recognising SpamAssassin-ness is
what P is for -- but it must not be the mechanism behind the scores.

This script recomputes each affected figure three ways:

    as reported   the original construction
    exact-disjoint every message whose body also appears on the evaluation side
                  is removed from P's training pool
    near-disjoint  additionally removes training messages whose word 5-shingle
                  Jaccard similarity to any evaluation message is >= 0.8

PARTS
-----
A  Benchmark provenance transfer (Table "Provenance transfer"): F1, balanced
   accuracy, MCC, share called phishing, and the label-shuffled null.
B  Composed benchmarks with a SpamAssassin ham side (Table "generalise"): the
   ham source is split into two halves with an independent RNG; P trains on one
   half, the benchmark draws its ham from the other. The CEAS-ham rows are not
   recomputed: they already fail, and removing overlap can only lower a
   transfer score.

All intervals are 95% Student-t half-widths over five seeds (t_{0.975,4} =
2.776), matching the paper. Writes results/provenance_overlap_audit.json.

Run from the repository root with the corpora in ../phishguard_dataset_builder/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
JACCARD = 0.8


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


def scores(yt, yp):
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 == p)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 == p)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {"f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "bal_acc": (rc + tnr) / 2,
            "mcc": (tp * tn - fp * fn) / den if den else 0.0,
            "pct_phish": (tp + fp) / len(yt)}


def summarise(runs):
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(v.mean()),
                  "ci95_t": float(T975_4 * v.std(ddof=1) / len(v) ** 0.5),
                  "per_seed": [float(x) for x in v]}
    return out


def near_dup_mask(pool, targets, thr=JACCARD):
    """True for each pool message whose 5-shingle Jaccard with any target >= thr."""
    vec = CountVectorizer(analyzer="word", ngram_range=(5, 5), binary=True,
                          lowercase=True, token_pattern=r"(?u)\b\w+\b",
                          dtype=np.int32)
    vec.fit(pool + targets)
    A = vec.transform(pool)
    B = vec.transform(targets)
    a = np.asarray(A.sum(1)).ravel()
    b = np.asarray(B.sum(1)).ravel()
    hit = np.zeros(A.shape[0], dtype=bool)
    step = 512
    for i in range(0, A.shape[0], step):
        inter = (A[i:i + step] @ B.T).tocoo()
        if inter.nnz == 0:
            continue
        union = a[i + inter.row] + b[inter.col] - inter.data
        jac = np.divide(inter.data, union, out=np.zeros_like(union, dtype=float),
                        where=union > 0)
        rows = inter.row[jac >= thr] + i
        hit[rows] = True
    return hit


def part_A():
    sa, ce = bodies("SpamAssasin.csv", "0"), bodies("CEAS_08.csv", "0")
    Xt, yt = split("phishguard_test.csv")
    Xd, _ = split("phishguard_dev.csv")
    eval_set = set(Xt) | set(Xd)
    test_ham = [b for b, y in zip(Xt, yt) if y == 0]
    exact = [b for b in sa if b not in eval_set]
    nd = near_dup_mask(exact, Xt + Xd)
    near = [b for b, h in zip(exact, nd) if not h]
    sa_set = set(sa)
    report = {
        "test_ham_in_training_pool_exact": sum(1 for b in test_ham if b in sa_set),
        "test_ham_total": len(test_ham),
        "pool_sizes": {"as_reported": len(sa), "exact_disjoint": len(exact),
                       "near_disjoint": len(near)},
        "near_dup_threshold_jaccard_5shingle": JACCARD,
    }

    def run(pool, shuffle):
        out = []
        for s in SEEDS:
            rng = random.Random(s)
            n = min(len(pool), len(ce), 1500)
            X = rng.sample(pool, n) + rng.sample(ce, n)
            y = [0] * n + [1] * n
            if shuffle:
                rng.shuffle(y)
            P = pipe(); P.fit(X, y)
            out.append(scores(yt, [0 if p == 0 else 1 for p in P.predict(Xt)]))
        return summarise(out)

    for tag, pool in (("as_reported", sa), ("exact_disjoint", exact),
                      ("near_disjoint", near)):
        report[tag] = {"provenance": run(pool, False), "null": run(pool, True)}
        r = report[tag]["provenance"]
        print(f"  A {tag:<15} F1 {100*r['f1']['mean']:6.2f} +/-{100*r['f1']['ci95_t']:.2f}"
              f"  bal {100*r['bal_acc']['mean']:6.2f}  MCC {r['mcc']['mean']:.3f}"
              f"  null F1 {100*report[tag]['null']['f1']['mean']:.2f}"
              f" +/-{100*report[tag]['null']['f1']['ci95_t']:.2f}")
    return report


def part_B():
    sa = bodies("SpamAssasin.csv", "0")
    ce = bodies("CEAS_08.csv", "0")
    naz = bodies("Nazario.csv", "1")
    nig = bodies("Nigerian_Fraud.csv", "1")
    report = {}

    def once(prov_a, prov_b, bench_ph, bench_ham, seed, shuffle):
        rng = random.Random(seed)
        n = min(len(prov_a), len(prov_b), 1200)
        X = rng.sample(prov_a, n) + rng.sample(prov_b, n)
        y = [0] * n + [1] * n
        if shuffle:
            rng.shuffle(y)
        P = pipe(); P.fit(X, y)
        m = min(len(bench_ph), len(bench_ham), 900)
        bx = rng.sample(bench_ph, m) + rng.sample(bench_ham, m)
        by = [1] * m + [0] * m
        return scores(by, [0 if p == 0 else 1 for p in P.predict(bx)])

    for name, ph in (("Nazario / SpamAssassin ham", naz),
                     ("Nigerian / SpamAssassin ham", nig)):
        report[name] = {}
        for tag in ("as_reported", "exact_disjoint", "near_disjoint"):
            real, null = [], []
            for s in SEEDS:
                if tag == "as_reported":
                    tr, bh = sa, sa
                else:
                    h = list(sa); random.Random(10_000 + s).shuffle(h)
                    tr, bh = h[:len(h) // 2], h[len(h) // 2:]
                    bhs = set(bh)
                    tr = [b for b in tr if b not in bhs]
                    if tag == "near_disjoint":
                        m = near_dup_mask(tr, bh)
                        tr = [b for b, hh in zip(tr, m) if not hh]
                real.append(once(tr, ce, ph, bh, s, False))
                null.append(once(tr, ce, ph, bh, s, True))
            report[name][tag] = {"provenance": summarise(real), "null": summarise(null)}
            r = report[name][tag]["provenance"]["f1"]
            nl = report[name][tag]["null"]["f1"]
            print(f"  B {name:<28} {tag:<15} F1 {100*r['mean']:6.2f} +/-{100*r['ci95_t']:.2f}"
                  f"   null {100*nl['mean']:6.2f} +/-{100*nl['ci95_t']:.2f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], default=None)
    a = ap.parse_args()
    out_p = RESULTS / "provenance_overlap_audit.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    out["config"] = {"seeds": SEEDS, "t_crit_95": T975_4, "jaccard": JACCARD,
                     "shingle": "word 5-gram"}
    print("=" * 78)
    print("  PROVENANCE OVERLAP AUDIT")
    print("=" * 78)
    if a.part in (None, "A"):
        out["A_benchmark_transfer"] = part_A()
    if a.part in (None, "B"):
        out["B_composed_benchmarks"] = part_B()
    RESULTS.mkdir(exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {out_p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
