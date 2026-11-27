"""
Provenance audit of the six-corpus phishing compilation (DMLR gap D2).

    python experiments/compilation_audit.py            # all parts
    python experiments/compilation_audit.py --quick    # 1 seed, for a smoke test

WHY THIS EXPERIMENT
-------------------
The prevalence audit found that most 2024-2026 papers do not use the
Nazario-vs-SpamAssassin pair directly. They use the Kaggle "Phishing Email
Dataset" (Al-Subaiey et al. 2024): six public corpora merged into 82,486
emails, usually as subject+body text with a 0/1 label. A reviewer will ask
whether the provenance problem exists in THAT dataset.

WHAT IS IN THE COMPILATION (checked 24 Sep 2026)
    CEAS_08         39,154   spam 21,842 / ham 17,312   dated 2008
    Enron           29,767   spam 13,976 / ham 15,791   no date column
    Ling             2,859   spam    458 / ham  2,401   no date column
    SpamAssassin     5,809   spam  1,718 / ham  4,091   dated 2002
    Nazario          1,565   phishing only               dated 2015-2022
    Nigerian_Fraud   3,332   fraud only                  dated ~2005-2007
Four corpora contain both classes (spam vs ham). Only Nazario and
Nigerian_Fraud are single-class -- and they are the only sources of phishing
or fraud proper. Everything else labelled 1 is spam.

PARTS
-----
A  Standard evaluation: stratified random 80/20 split, TF-IDF + LR on
   subject+body (the merged-file view). Overall F1, plus a per-source
   breakdown of the same predictions.
B  Source-majority oracle: predict each test email's label as the majority
   label of its source corpus (majority taken from the training split).
   How much does knowing the source alone give?
C  Label-free provenance detector: a 6-way corpus classifier trained on the
   training split with CORPUS labels only (no per-email class labels), whose
   predicted corpus is mapped to that corpus's majority label.
D  Corpus identity with the class held constant: among label-1 emails only,
   and among label-0 emails only, classify the source corpus (balanced
   sample, 5-fold CV). Null: one corpus split into two random halves.
E  Leave-one-source-out: train on the other five sources, test on the held-
   out one. For single-class sources (Nazario, Nigerian) this is the share
   of emails flagged, i.e. recall on a phishing/fraud collection the model
   never saw.
F  Datasheet: per source and class, share of rows with a parseable Date and
   the year range. Descriptive only.

All models: TF-IDF (1-2 grams, min_df 2, 200k features, sublinear tf) +
LogisticRegression(C=4, class_weight="balanced"), the same pipeline as the
rest of the paper. 95% intervals use Student t with n-1 df.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from email.utils import parsedate_tz
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

csv.field_size_limit(10_000_000)

HERE = Path(__file__).resolve().parent
try:
    sys.path.insert(0, str(HERE.parent))
    from phishcascade.config import DATA_DIR, RESULTS_DIR   # noqa: E402
except Exception:
    DATA_DIR = HERE / "data"
    RESULTS_DIR = HERE / "results"
if os.environ.get("PHISH_DATA_DIR"):
    DATA_DIR = Path(os.environ["PHISH_DATA_DIR"])

SOURCES = [("CEAS_08.csv", "CEAS"), ("Enron.csv", "Enron"), ("Ling.csv", "Ling"),
           ("SpamAssasin.csv", "SpamAssassin"), ("Nazario.csv", "Nazario"),
           ("Nigerian_Fraud.csv", "Nigerian")]
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
N_PER_GROUP_D = 400          # per corpus in part D (Ling has only 458 spam)


def clean(s):
    return CTRL.sub(" ", s or "")


def pipe():
    return Pipeline([
        ("v", TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                              min_df=2, max_features=200_000)),
        ("c", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


def f1_of(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(y) if y else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def agg(v):
    v = [float(x) for x in v]
    m = float(np.mean(v))
    h = T_CRIT[len(v) - 1] * float(np.std(v, ddof=1)) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return {"mean_pct": round(100 * m, 2), "ci95_pct": round(100 * h, 2),
            "n": len(v), "per_seed": [round(x, 6) for x in v]}


def year_of(d):
    if not d:
        return None
    t = parsedate_tz(d)
    if t and 1990 <= t[0] <= 2030:
        return t[0]
    m = re.search(r"\b(199\d|20[0-2]\d)\b", d)
    return int(m.group(1)) if m else None


def load():
    X, y, src, dates = [], [], [], []
    for fn, name in SOURCES:
        with (DATA_DIR / fn).open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                lab = (r.get("label") or "").strip()
                if lab not in ("0", "1"):
                    continue
                X.append(clean(r.get("subject", "")) + "\n" + clean(r.get("body", "")))
                y.append(int(lab))
                src.append(name)
                dates.append(r.get("date", "") or "")
    return X, y, src, dates


def per_source(y, p, s):
    out = {}
    for name in sorted(set(s)):
        idx = [i for i, v in enumerate(s) if v == name]
        yy = [y[i] for i in idx]
        pp = [p[i] for i in idx]
        m = f1_of(yy, pp)
        out[name] = {"n": len(idx), "n_pos": sum(yy),
                     "flagged_share": sum(pp) / len(pp),
                     "recall": m["recall"] if sum(yy) else None,
                     "fpr": m["fpr"] if len(yy) - sum(yy) else None,
                     "accuracy": m["accuracy"]}
    return out


def part_abc(X, y, src, seeds):
    res = {"A": [], "B": [], "C": [], "C_corpus_id_acc": []}
    per_src = {"A": [], "C": []}
    for s in seeds:
        idx = list(range(len(y)))
        tr, te = train_test_split(idx, test_size=0.2, random_state=s, stratify=y)
        Xtr, ytr, str_ = [X[i] for i in tr], [y[i] for i in tr], [src[i] for i in tr]
        Xte, yte, ste = [X[i] for i in te], [y[i] for i in te], [src[i] for i in te]

        # A: standard supervised evaluation
        t0 = time.time()
        M = pipe(); M.fit(Xtr, ytr)
        pA = [int(v) for v in M.predict(Xte)]
        res["A"].append(f1_of(yte, pA)); per_src["A"].append(per_source(yte, pA, ste))

        # B: source-majority oracle (majority from the TRAIN split)
        maj = {}
        for name in set(str_):
            labs = [ytr[i] for i, v in enumerate(str_) if v == name]
            maj[name] = int(sum(labs) * 2 >= len(labs))
        pB = [maj[v] for v in ste]
        res["B"].append(f1_of(yte, pB))

        # C: label-free detector: corpus classifier, corpus -> majority label
        P = pipe(); P.set_params(c__class_weight=None); P.fit(Xtr, str_)
        pc = list(P.predict(Xte))
        res["C_corpus_id_acc"].append(sum(a == b for a, b in zip(pc, ste)) / len(ste))
        pC = [maj[v] for v in pc]
        res["C"].append(f1_of(yte, pC)); per_src["C"].append(per_source(yte, pC, ste))
        print(f"  seed {s}: A F1 {100*res['A'][-1]['f1']:.2f} | B F1 {100*res['B'][-1]['f1']:.2f}"
              f" | C F1 {100*res['C'][-1]['f1']:.2f} (corpus-ID acc "
              f"{100*res['C_corpus_id_acc'][-1]:.2f}) | {time.time()-t0:.0f}s", flush=True)
        res.setdefault("majority_map", maj)
    return res, per_src


def part_d(X, y, src, seeds):
    out = {}
    for lab in (1, 0):
        groups = defaultdict(list)
        for i, (l, s) in enumerate(zip(y, src)):
            if l == lab:
                groups[s].append(i)
        names = sorted(n for n, g in groups.items() if len(g) >= N_PER_GROUP_D)
        accs, null_accs = [], []
        big = max(names, key=lambda n: len(groups[n]))
        for s in seeds:
            rng = random.Random(s)
            Xs, ys = [], []
            for n in names:
                for i in rng.sample(groups[n], N_PER_GROUP_D):
                    Xs.append(X[i]); ys.append(n)
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            accs.append(float(np.mean(cross_val_score(pipe(), Xs, ys, cv=cv))))
            # null: two random halves of the largest single corpus
            samp = rng.sample(groups[big], 2 * N_PER_GROUP_D)
            Xn = [X[i] for i in samp]; yn = [0] * N_PER_GROUP_D + [1] * N_PER_GROUP_D
            null_accs.append(float(np.mean(cross_val_score(pipe(), Xn, yn, cv=cv))))
        key = "label1_spam_or_phish" if lab == 1 else "label0_ham"
        out[key] = {"corpora": names, "chance_pct": round(100 / len(names), 2),
                    "corpus_id_acc": agg(accs), "null_corpus": big,
                    "null_acc": agg(null_accs), "n_per_corpus": N_PER_GROUP_D}
        print(f"  D {key}: {len(names)}-way corpus ID acc {out[key]['corpus_id_acc']['mean_pct']}"
              f" (chance {out[key]['chance_pct']}), null {out[key]['null_acc']['mean_pct']}",
              flush=True)
    return out


def part_e(X, y, src):
    out = {}
    for _, name in SOURCES:
        tr = [i for i, s in enumerate(src) if s != name]
        te = [i for i, s in enumerate(src) if s == name]
        M = pipe(); M.fit([X[i] for i in tr], [y[i] for i in tr])
        p = [int(v) for v in M.predict([X[i] for i in te])]
        yy = [y[i] for i in te]
        m = f1_of(yy, p)
        single = len(set(yy)) == 1
        out[name] = {"n": len(te), "single_class": single,
                     "flagged_share": sum(p) / len(p),
                     "f1": None if single else m["f1"],
                     "recall": m["recall"] if sum(yy) else None,
                     "fpr": m["fpr"] if len(yy) - sum(yy) else None}
        shown = (f"flagged {100*out[name]['flagged_share']:.2f}%" if single
                 else f"F1 {100*m['f1']:.2f}")
        print(f"  E leave-out {name:<13} n={len(te):<6} {shown}", flush=True)
    return out


def part_f(y, src, dates):
    out = {}
    for _, name in SOURCES:
        for lab in (0, 1):
            ds = [d for l, s, d in zip(y, src, dates) if s == name and l == lab]
            if not ds:
                continue
            ys = [year_of(d) for d in ds]
            ok = [v for v in ys if v]
            c = Counter(ok)
            out[f"{name}_label{lab}"] = {
                "n": len(ds), "has_date_share": round(len(ok) / len(ds), 4),
                "year_min": min(ok) if ok else None, "year_max": max(ok) if ok else None,
                "top_years": c.most_common(3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="ABCDEF")
    args = ap.parse_args()
    seeds = [0] if args.quick else SEEDS

    X, y, src, dates = load()
    print(f"loaded {len(y)} emails from {len(SOURCES)} sources "
          f"({sum(y)} label 1 / {len(y)-sum(y)} label 0)")
    assert len(y) == 82486, "expected the 82,486-email compilation"

    out = {"n_emails": len(y), "seeds": seeds, "sources": [n for _, n in SOURCES],
           "view": "subject + body text (merged-file view)"}
    if set("ABC") & set(args.parts):
        print("PART A/B/C"); r, ps = part_abc(X, y, src, seeds)
        for k in ("A", "B", "C"):
            out[f"{k}_f1"] = agg([m["f1"] for m in r[k]])
            out[f"{k}_runs"] = r[k]
        out["C_corpus_id_acc"] = agg(r["C_corpus_id_acc"])
        out["majority_map"] = r["majority_map"]
        out["A_per_source_seed0"] = ps["A"][0]
        out["C_per_source_seed0"] = ps["C"][0]
    if "D" in args.parts:
        print("PART D"); out["D"] = part_d(X, y, src, seeds)
    if "E" in args.parts:
        print("PART E"); out["E"] = part_e(X, y, src)
    if "F" in args.parts:
        out["F"] = part_f(y, src, dates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / ("compilation_audit_QUICK.json" if args.quick else "compilation_audit.json")
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
