"""
Corpus-overlap control for every result that depends on the provenance detector.

The provenance detector P separates SpamAssassin ham from CEAS ham.
SpamAssassin also supplies the benchmark's ham class, so benchmark messages are
eligible for P's training pool: 95.6% of test-split ham is in that pool and
roughly a third is drawn in a given seed. The overlap is intrinsic to the
construction -- recognising SpamAssassin-ness is what P is for -- but it must
not be the mechanism behind the scores.

This script recomputes each affected quantity with the training pool made
disjoint from the evaluation set, and prints the as-published figure beside it.
Reproduces Table III of the paper.

    python3 experiments/overlap_control.py

Run from the repository root with the corpora in ../phishguard_dataset_builder/
(see dataset_builder/SETUP.md).

Not affected, and not recomputed here: the model-free rule (no fitted model),
the date-only control and cross-corpus transfer (trained on pooled corpora that
do not supply the test split), and the supervised reference (trained on the
benchmark's own disjoint development split).
"""
import csv, os, random, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "phishguard_dataset_builder"))
csv.field_size_limit(10_000_000)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
clean = lambda s: CTRL.sub(" ", s or "")
SEEDS = [0, 1, 2, 3, 4]



# 95% CI half-width. With n=5 seeds the sampling distribution of the mean
# is Student's t with 4 df, so the critical value is 2.776, not the
# normal-approximation 1.96. Update this if SEEDS changes length.
T_CRIT_95 = 2.776

def bodies(fn, lab):
    out = []
    for r in csv.DictReader(open(fn, encoding="utf-8", errors="replace", newline="")):
        if (r.get("label") or "").strip() != lab:
            continue
        b = clean(r.get("body", "")).strip()
        if len(b) >= 30:
            out.append(b)
    return out


def split(fn):
    rows = list(csv.DictReader(open(fn, encoding="utf-8", errors="replace", newline="")))
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
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            (tp * tn - fp * fn) / den if den else 0.0)


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


ci = lambda v: T_CRIT_95 * np.std(v, ddof=1) / len(v) ** 0.5


def main():
    sa, ce = bodies("SpamAssasin.csv", "0"), bodies("CEAS_08.csv", "0")
    nig = bodies("Nigerian_Fraud.csv", "1")
    pool = {}
    for fn, nm in (("CEAS_08.csv", "CEAS"), ("Ling.csv", "Ling"), ("Enron.csv", "Enron")):
        ph, hm = bodies(fn, "1"), bodies(fn, "0")
        if len(ph) >= 300 and len(hm) >= 300:
            pool[nm] = (ph, hm)

    Xt, yt = split("phishguard_test.csv")
    Xd, yd = split("phishguard_dev.csv")
    seen = set(Xt) | set(Xd)
    saD = [b for b in sa if b not in seen]

    elig = sum(1 for b, l in zip(Xt, yt) if l == 0 and b in set(sa))
    nham = sum(1 for l in yt if l == 0)
    print("=" * 78)
    print("  CORPUS-OVERLAP CONTROL")
    print(f"  test-split ham eligible for P's training pool: "
          f"{elig}/{nham} = {100*elig/nham:.1f}%")
    print(f"  SpamAssassin pool {len(sa)} -> {len(saD)} after exclusion")
    print("  Each quantity uses an independent RNG per seed, so figures do not")
    print("  depend on the order in which they are computed.")
    print("=" * 78)

    def provenance(P_pool, seed):
        rng = random.Random(seed)
        n = min(len(P_pool), len(ce), 1500)
        P = pipe(); P.fit(rng.sample(P_pool, n) + rng.sample(ce, n), [0]*n + [1]*n)
        return [0 if p == 0 else 1 for p in P.predict(Xt)]

    def pooled(seed):
        rng = random.Random(seed)
        Xp, yp = [], []
        for nm, (ph, hm) in pool.items():
            k = min(700, len(ph), len(hm))
            Xp += rng.sample(ph, k); yp += [1]*k
            Xp += rng.sample(hm, k); yp += [0]*k
        M = pipe(); M.fit(Xp, yp)
        return list(M.predict(Xt))

    def supervised():
        sup = pipe(); sup.fit(Xd, yd)
        return list(sup.predict(Xt))

    def nigerian(disjoint, seed):
        rng = random.Random(seed)
        if disjoint:
            part = random.Random(10_000 + seed)      # independent of rng
            h = list(sa); part.shuffle(h)
            tp_, bh_ = h[:len(h)//2], h[len(h)//2:]
        else:
            tp_, bh_ = sa, sa
        n = min(len(tp_), len(ce), 1200)
        Q = pipe(); Q.fit(rng.sample(tp_, n) + rng.sample(ce, n), [0]*n + [1]*n)
        m = min(len(nig), len(bh_), 900)
        bx = rng.sample(nig, m) + rng.sample(bh_, m)
        return scores([1]*m + [0]*m, [0 if p == 0 else 1 for p in Q.predict(bx)])[0]

    sup_pred = supervised()
    res = {}
    for tag, P_pool in (("as reported", sa), ("disjoint", saD)):
        pv, pl, kp, ag, agk, nigf = [], [], [], [], [], []
        for s in SEEDS:
            prov = provenance(P_pool, s); pool_pred = pooled(s)
            pv.append(scores(yt, prov)[0]*100)
            pl.append(scores(yt, pool_pred)[0]*100)
            kp.append(kappa(pool_pred, prov))
            ag.append(100*sum(1 for a, b in zip(sup_pred, prov) if a == b)/len(sup_pred))
            agk.append(kappa(sup_pred, prov))
            nigf.append(nigerian(tag == "disjoint", s)*100)
        res[tag] = dict(pv=pv, pl=pl, kp=kp, ag=ag, agk=agk, nig=nigf)

    a, b = res["as reported"], res["disjoint"]
    print(f"\n{'quantity':<42}{'as reported':>16}{'disjoint':>16}")
    print("-" * 74)
    def line(lab, ka, kb=None, fmt="%.2f", pm=True):
        va, vb = a[ka], b[kb or ka]
        sa_ = (fmt % np.mean(va)) + (" +/-%.2f" % ci(va) if pm else "")
        sb_ = (fmt % np.mean(vb)) + (" +/-%.2f" % ci(vb) if pm else "")
        print(f"{lab:<42}{sa_:>16}{sb_:>16}")
    line("Provenance F1 on benchmark", "pv")
    print(f"{'  margin over pooled model':<42}{np.mean(a['pv'])-np.mean(a['pl']):>16.1f}"
          f"{np.mean(b['pv'])-np.mean(b['pl']):>16.1f}")
    print(f"{'  kappa versus pooled model':<42}{np.mean(a['kp']):>16.3f}{np.mean(b['kp']):>16.3f}")
    line("Agreement with supervised detector", "ag")
    print(f"{'  kappa':<42}{np.mean(a['agk']):>16.3f}{np.mean(b['agk']):>16.3f}")
    line("Nigerian-fraud swap F1", "nig")
    print("\n  Constant-classifier floor for F1 on a balanced set: 66.67")
    print("  Pooled model is unchanged between columns (it never trains on")
    print("  SpamAssassin); it is shown only to locate the margin.")
    print(f"  Pooled F1: {np.mean(a['pl']):.2f} +/-{ci(a['pl']):.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
