"""
Follow-ups to compilation_audit.py that close DMLR gap D2.

    python experiments/compilation_followups.py            # both parts, 5 seeds
    python experiments/compilation_followups.py --quick    # 1 seed, fewer k

Needs compilation_audit.py in the same folder (it reuses its loader and
pipeline, so every setting is identical).

G  NAZARIO EXPOSURE SWEEP
   compilation_audit.py part E showed: a model trained on the other five
   sources flags only ~60% of Nazario phishing; with Nazario in training,
   recall is ~98%. This part draws the curve in between.

   Per seed:
     * Nazario is split into a fixed TEST set (565 emails) and an ADD pool
       (1,000 emails).
     * The other five sources are split 80/20, stratified by source+label.
     * Train on the 80% of the other sources + k Nazario emails from the pool.
     * Report: recall on the Nazario test set; F1 and FPR on the 20% of the
       other sources (so we see whether adding Nazario costs anything).
   k = 0 should land near part E's leave-Nazario-out result (~60%).

H  DATE CHECKS (Date exists only for CEAS, SpamAssassin, Nazario, Nigerian;
   Enron and Ling have no date column; the merged Kaggle file has no date)
   H1  Year-majority rule on the dated subset: learn P(label | year) on a
       train split, predict the majority label for each year, test on the
       held-out split. Compared with the same rule applied WITHIN each dated
       mixed corpus (CEAS alone, SpamAssassin alone), where both classes come
       from one collection period and the rule should be near chance.
   H2  The two-corpus analogue: Nazario phishing vs ALL legitimate mail in the
       compilation that has a date (CEAS ham 2008, SpamAssassin ham 2002),
       with the fixed rule "phishing iff year >= 2009". No fitting at all.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compilation_audit as ca  # noqa: E402

K_SWEEP = [0, 10, 25, 50, 100, 250, 500, 1000]
N_NAZ_TEST = 565


# ── G ───────────────────────────────────────────────────────────────────────
def part_g(X, y, src, seeds, ks):
    naz = [i for i, s in enumerate(src) if s == "Nazario"]
    other = [i for i, s in enumerate(src) if s != "Nazario"]
    runs = defaultdict(list)
    for seed in seeds:
        rng = random.Random(seed)
        naz_sh = naz[:]; rng.shuffle(naz_sh)
        naz_test, naz_pool = naz_sh[:N_NAZ_TEST], naz_sh[N_NAZ_TEST:]
        # stratified 80/20 split of the other sources by (source, label)
        strata = defaultdict(list)
        for i in other:
            strata[(src[i], y[i])].append(i)
        tr, te = [], []
        for key in sorted(strata):
            g = strata[key][:]; rng.shuffle(g)
            cut = int(round(0.8 * len(g)))
            tr += g[:cut]; te += g[cut:]
        for k in ks:
            add = naz_pool[:k]
            idx = tr + add
            M = ca.pipe(); M.fit([X[i] for i in idx], [y[i] for i in idx])
            p_naz = M.predict([X[i] for i in naz_test])
            p_oth = [int(v) for v in M.predict([X[i] for i in te])]
            m_oth = ca.f1_of([y[i] for i in te], p_oth)
            r = {"seed": seed, "k": k,
                 "nazario_recall": float(sum(p_naz)) / len(p_naz),
                 "other_f1": m_oth["f1"], "other_fpr": m_oth["fpr"],
                 "other_recall": m_oth["recall"]}
            runs[k].append(r)
            print(f"  G seed {seed} k={k:<5} Nazario recall {100*r['nazario_recall']:6.2f}"
                  f" | other F1 {100*r['other_f1']:6.2f} FPR {100*r['other_fpr']:5.2f}",
                  flush=True)
    summary = {}
    for k in ks:
        summary[str(k)] = {m: ca.agg([r[m] for r in runs[k]])
                           for m in ("nazario_recall", "other_f1", "other_fpr")}
    return {"k_sweep": ks, "n_nazario_test": N_NAZ_TEST,
            "runs": {str(k): v for k, v in runs.items()}, "summary": summary}


# ── H ───────────────────────────────────────────────────────────────────────
def year_majority(train, test):
    """train/test: lists of (year, label). Predict majority label per year."""
    by = defaultdict(Counter)
    for yr, lab in train:
        by[yr][lab] += 1
    overall = Counter(l for _, l in train).most_common(1)[0][0]
    pred = [by[yr].most_common(1)[0][0] if yr in by else overall for yr, _ in test]
    return ca.f1_of([l for _, l in test], pred)


def part_h(y, src, dates, seeds):
    yrs = [ca.year_of(d) for d in dates]
    dated_sources = ["CEAS", "SpamAssassin", "Nazario", "Nigerian"]
    out = {"H1": {}, "H2": {}}

    def pairs(names):
        return [(yrs[i], y[i]) for i in range(len(y)) if src[i] in names and yrs[i]]

    for label, names in (("dated_subset_4_corpora", dated_sources),
                         ("within_CEAS", ["CEAS"]),
                         ("within_SpamAssassin", ["SpamAssassin"])):
        data = pairs(names)
        f1s, bals = [], []
        for seed in seeds:
            rng = random.Random(seed)
            d = data[:]; rng.shuffle(d)
            cut = int(0.8 * len(d))
            m = year_majority(d[:cut], d[cut:])
            f1s.append(m["f1"])
            bals.append((m["recall"] + 1 - m["fpr"]) / 2)   # balanced accuracy
        share = sum(l for _, l in data) / len(data)
        out["H1"][label] = {"n": len(data), "f1": ca.agg(f1s),
                            "balanced_accuracy": ca.agg(bals),
                            "label1_share": round(share, 4),
                            # F1 of predicting everything as label 1 (the floor)
                            "all_positive_f1_floor": round(2 * share / (1 + share), 4)}
        print(f"  H1 year-majority {label:<24} n={len(data):<6} F1 "
              f"{out['H1'][label]['f1']['mean_pct']} (all-positive floor "
              f"{100*out['H1'][label]['all_positive_f1_floor']:.2f}) | balanced acc "
              f"{out['H1'][label]['balanced_accuracy']['mean_pct']} (chance 50)", flush=True)

    # H2: Nazario vs all dated legitimate mail, fixed rule year >= 2009
    naz = [yrs[i] for i in range(len(y)) if src[i] == "Nazario" and yrs[i]]
    ham = [yrs[i] for i in range(len(y)) if y[i] == 0 and yrs[i]]
    yt = [1] * len(naz) + [0] * len(ham)
    pr = [int(v >= 2009) for v in naz + ham]
    m = ca.f1_of(yt, pr)
    out["H2"] = {"rule": "phishing iff year >= 2009", "n_nazario": len(naz),
                 "n_ham_dated": len(ham), "ham_sources": sorted({src[i] for i in range(len(y))
                                                                  if y[i] == 0 and yrs[i]}),
                 **{k: m[k] for k in ("f1", "recall", "fpr", "tp", "fp", "fn", "tn")}}
    print(f"  H2 Nazario vs dated ham, rule year>=2009: F1 {100*m['f1']:.2f} "
          f"recall {100*m['recall']:.2f} FPR {100*m['fpr']:.2f} "
          f"(n={len(naz)} phishing / {len(ham)} ham)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="GH")
    args = ap.parse_args()
    seeds = [0] if args.quick else ca.SEEDS
    ks = [0, 50, 500] if args.quick else K_SWEEP

    X, y, src, dates = ca.load()
    assert len(y) == 82486
    out = {"seeds": seeds}
    if "G" in args.parts:
        print("PART G — Nazario exposure sweep"); out["G"] = part_g(X, y, src, seeds, ks)
        print("  summary (mean ± 95% CI):")
        for k in ks:
            s = out["G"]["summary"][str(k)]
            print(f"    k={k:<5} Nazario recall {s['nazario_recall']['mean_pct']:6.2f} "
                  f"±{s['nazario_recall']['ci95_pct']:.2f} | other F1 "
                  f"{s['other_f1']['mean_pct']:6.2f} | other FPR {s['other_fpr']['mean_pct']:5.2f}")
    if "H" in args.parts:
        print("PART H — date checks"); out["H"] = part_h(y, src, dates, seeds)

    ca.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = ca.RESULTS_DIR / ("compilation_followups_QUICK.json" if args.quick
                          else "compilation_followups.json")
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
