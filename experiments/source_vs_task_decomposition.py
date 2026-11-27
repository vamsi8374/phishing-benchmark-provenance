"""
Source exposure or phishing knowledge? Decomposing the source-exposure sweep.

    python experiments/source_vs_task_decomposition.py

WHY THIS EXPERIMENT
-------------------
`pooled_indistribution_sweep.py` shows that adding k benchmark dev messages
per class to a pooled multi-corpus training set lifts benchmark F1 from 73.48
(k=0) to 93.37 (k=50) and 99.07 (k=465). The paper wants to read this as
"the benchmark score tracks source exposure".

A reviewer can object: the pool corpora (CEAS, Ling, Enron) label SPAM as the
positive class, not phishing. So the added dev messages give the model two
things at once:

    (1) the benchmark's SOURCES   (SpamAssassin ham, Nazario phishing)
    (2) PHISHING examples at all  (the pool had none)

The sweep cannot tell these apart. This script adds arms that separate them.

ARMS (same pool, same seeds, same k values as the sweep)
--------------------------------------------------------
    A  pooled + k dev phish + k dev ham   the original sweep (reproduced)
    H  pooled + k dev HAM only            benchmark ham source, NO phishing
    P  pooled + k dev PHISH only          Nazario phishing (source + task)
    N  pooled + k Nigerian_Fraud          phishing-type fraud from a
                                          NON-benchmark source (task, no source)

HOW TO READ IT
--------------
    H rises like A, N stays low      -> source exposure drives the score
    N rises like P, H stays low      -> phishing knowledge drives the score
    both rise                        -> both matter; report the split

CAVEATS (state them in the paper if you use this)
-------------------------------------------------
* Nigerian_Fraud is advance-fee fraud, not credential phishing. It is the
  only non-Nazario phishing-labelled corpus in the compilation. N therefore
  measures "fraud/phishing-type content from another source", not "the same
  task from another source".
* H and P add one class only, so the class ratio of the training set shifts.
  The classifier uses class_weight="balanced", which compensates in the loss
  but not in the vocabulary.
* Everything is body-only TF-IDF + logistic regression, as in the sweep.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

csv.field_size_limit(10_000_000)

HERE = Path(__file__).resolve().parent
try:                                    # inside phishcascade/experiments
    sys.path.insert(0, str(HERE.parent))
    from phishcascade.config import DATA_DIR, RESULTS_DIR   # noqa: E402
except Exception:                       # standalone copy
    DATA_DIR = Path(os.environ.get("PHISH_DATA_DIR", HERE / "data"))
    RESULTS_DIR = HERE / "results"
if os.environ.get("PHISH_DATA_DIR"):
    DATA_DIR = Path(os.environ["PHISH_DATA_DIR"])

SEEDS = [0, 1, 2, 3, 4]
T_CRIT_95 = 2.776                      # Student t, 4 df (n = 5 seeds)
K_SWEEP = [0, 50, 100, 200, 465]
PER_CORPUS = 700                        # matches the sweep
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BENIGN, PHISH = 0, 1


def clean(s):
    return CTRL.sub(" ", s or "")


def rows(name):
    p = DATA_DIR / name
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def bodies(name, label):
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
    return {"f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(y_true),
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0}


def agg(v):
    m = float(np.mean(v))
    h = T_CRIT_95 * float(np.std(v, ddof=1)) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return m, h


def draw_pool(rng, pool):
    """Identical draw order to pooled_indistribution_sweep.py."""
    X, y = [], []
    for nm, (ph, hm) in pool.items():
        m = min(PER_CORPUS, len(ph), len(hm))
        X += rng.sample(ph, m); y += [PHISH] * m
        X += rng.sample(hm, m); y += [BENIGN] * m
    return X, y


def main():
    pool = {}
    for fn, nm in (("CEAS_08.csv", "CEAS"), ("Ling.csv", "Ling"),
                   ("Enron.csv", "Enron")):
        ph, hm = bodies(fn, "1"), bodies(fn, "0")
        if len(ph) >= 300 and len(hm) >= 300:
            pool[nm] = (ph, hm)

    dev_ph = bodies("phishguard_dev.csv", "1")
    dev_hm = bodies("phishguard_dev.csv", "0")
    nig = bodies("Nigerian_Fraud.csv", "1")

    test = rows("phishguard_test.csv")
    Xt = [clean(r.get("body", "")) for r in test]
    yt = [PHISH if (r.get("label") or "").strip() == "1" else BENIGN for r in test]

    tset = set(Xt)
    overlap = {"dev": sum(b in tset for b in dev_ph + dev_hm),
               "nigerian": sum(b in tset for b in nig)}
    nig = [b for b in nig if b not in tset]      # never train on test bodies

    print("=" * 78)
    print("  SOURCE vs TASK DECOMPOSITION OF THE SOURCE-EXPOSURE SWEEP")
    print(f"  pool {sorted(pool)} | dev {len(dev_ph)}/{len(dev_hm)} | "
          f"nigerian {len(nig)} | test {len(yt)} "
          f"({sum(yt)} phish / {len(yt) - sum(yt)} ham)")
    print(f"  exact body overlap with test: {overlap}")
    print("=" * 78)

    per_seed = {}
    for k in K_SWEEP:
        for arm in ("A", "H", "P", "N"):
            if k == 0 and arm != "A":
                continue                         # k=0 is the same model for all
            runs = []
            for s in SEEDS:
                rng = random.Random(s)
                X, y = draw_pool(rng, pool)
                if arm == "A" and k:             # exact sweep code path
                    X += rng.sample(dev_ph, k); y += [PHISH] * k
                    X += rng.sample(dev_hm, k); y += [BENIGN] * k
                elif arm != "A":
                    r2 = random.Random(2000 + s)
                    if arm == "H":
                        X += r2.sample(dev_hm, k); y += [BENIGN] * k
                    elif arm == "P":
                        X += r2.sample(dev_ph, k); y += [PHISH] * k
                    elif arm == "N":
                        X += r2.sample(nig, k); y += [PHISH] * k
                M = pipe(); M.fit(X, y)
                runs.append(metrics(yt, list(M.predict(Xt))))
            per_seed[f"{arm}_k{k}"] = runs
            f, h = agg([r["f1"] for r in runs])
            fp, _ = agg([r["fpr"] for r in runs])
            rc, _ = agg([r["recall"] for r in runs])
            print(f"  {arm} k={k:<4} F1 {100*f:6.2f} ±{100*h:5.2f}   "
                  f"recall {100*rc:6.2f}   FPR {100*fp:6.2f}")

    summary = {name: {m: dict(zip(("mean", "ci95"), agg([r[m] for r in runs])))
                      for m in ("f1", "precision", "recall", "accuracy", "fpr")}
               for name, runs in per_seed.items()}

    # sanity: A must reproduce the sweep
    prior = RESULTS_DIR / "pooled_indistribution_sweep.json"
    if prior.exists():
        want = json.loads(prior.read_text())["summary"]
        for k in K_SWEEP:
            key = f"pooled_plus_dev_k{k}"
            if key in want:
                d = abs(summary[f"A_k{k}"]["f1"]["mean"] - want[key]["f1"]["mean"])
                print(f"  sanity A k={k}: delta vs sweep {100*d:.3f} "
                      f"[{'OK' if d < 1e-4 else 'MISMATCH'}]")

    out = {"design": "A=dev both, H=dev ham only, P=dev phish only, "
                     "N=Nigerian_Fraud phish only; all added to the same pool",
           "pool_corpora": sorted(pool), "k_sweep": K_SWEEP, "seeds": SEEDS,
           "t_crit_95": T_CRIT_95, "overlap_with_test": overlap,
           "per_seed": per_seed, "summary": summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "source_vs_task_decomposition.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
