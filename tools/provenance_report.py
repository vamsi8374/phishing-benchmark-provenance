#!/usr/bin/env python3
"""
provenance_report.py — audit a labelled email dataset for provenance confounding.

Given one or more CSV files, each treated as a SOURCE collection, it writes a
datasheet and a set of diagnostics that answer one question:

    How much of this dataset's label can be predicted from WHERE an email
    came from (its source collection, or its date), rather than from what
    it says?

USAGE
-----
    python provenance_report.py \
        --source Nazario=data/Nazario.csv:1 \
        --source SpamAssassin=data/SpamAssasin.csv:0 \
        --transfer SpamAssassin:CEAS=data/CEAS_08.csv:0 \
        --out reports/nazario_spamassassin

    --source NAME=PATH[:LABEL]   one per collection; ":LABEL" keeps only rows
                                 with that label (e.g. ham only). Repeatable.
    --transfer SRC:REF=PATH[:LABEL]
                                 optional provenance-transfer test: train a
                                 detector for "is this from SRC?" against an
                                 EXTERNAL reference corpus REF of the same
                                 class (no labels of the other class used),
                                 then score it on the dataset.
    --target NAME                optional exposure sweep for one source: hold
                                 it out, add k of its emails back to training.
    --licence NAME="CC BY 4.0"   optional, shown in the datasheet. Repeatable.
    --text-cols subject,body  --label-col label  --date-col date
    --positive 1                 value of the positive class in --label-col
    --max-per-group N            optional cap per (source, class) to bound run
                                 time; default 0 = no cap. A cap changes the class
                                 balance and can shift the numbers noticeably.
    --seeds 3

OUTPUT (in --out)
    report.md    human-readable report with a verdict and warnings
    report.json  every number, per seed

DIAGNOSTICS
    1 Datasheet           counts per source and class, date coverage, years,
                          licence.
    2 Label-source link   purity of each source; supervised F1 on a random
                          split vs. a source-majority oracle vs. a label-free
                          detector (source classifier -> source's majority
                          label). If the label-free detector gets close to the
                          supervised model, the score is explained by
                          provenance.
    3 Transfer (opt.)     the paper's provenance-transfer test.
    4 Separability        can the source be recognised with the CLASS held
                          fixed? Null: one source split into random halves.
    5 Date                year-majority rule, overall and within each mixed
                          source (balanced accuracy; 50 = chance).
    6 Leave-one-source-out
                          train without a source, test on it.
    7 Exposure (opt.)     how many emails of --target the model needs to see.

MODEL: TF-IDF (1-2 grams, min_df 2, 200k features, sublinear tf) + logistic
regression (C=4, balanced). 95% intervals: Student t over seeds.
Requires: scikit-learn, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from email.utils import parsedate_tz
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

csv.field_size_limit(10_000_000)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262}
EXPOSURE_K = [0, 10, 25, 50, 100, 250, 500]


# ── helpers ───────────────────────────────────────────────────────────────────
def pipe(balanced=True):
    return Pipeline([
        ("v", TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                              min_df=2, max_features=200_000)),
        ("c", LogisticRegression(max_iter=2000, C=4.0,
                                 class_weight="balanced" if balanced else None)),
    ])


def metrics(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    has_both = (tp + fn) > 0 and (tn + fp) > 0
    return {"f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "recall": rec, "fpr": 1 - spec if tn + fp else 0.0,
            "balanced_accuracy": (rec + spec) / 2 if has_both else None,
            "accuracy": (tp + tn) / len(y) if y else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def agg(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return None
    m = float(np.mean(v))
    h = T_CRIT.get(len(v) - 1, 1.96) * float(np.std(v, ddof=1)) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return {"mean": round(100 * m, 2), "ci95": round(100 * h, 2), "n": len(v)}


def fmt(a):
    return "n/a" if a is None else (f"{a['mean']:.2f} ± {a['ci95']:.2f}" if a["n"] > 1
                                    else f"{a['mean']:.2f}")


def year_of(d):
    if not d:
        return None
    t = parsedate_tz(d)
    if t and 1990 <= t[0] <= 2030:
        return t[0]
    m = re.search(r"\b(199\d|20[0-2]\d)\b", d)
    return int(m.group(1)) if m else None


def parse_spec(spec):
    """NAME=PATH[:LABEL] -> (name, path, label or None)."""
    name, rest = spec.split("=", 1)
    label = None
    if ":" in rest and not rest.endswith(".csv"):
        rest, label = rest.rsplit(":", 1)
    return name.strip(), Path(rest.strip()), label


def read_source(path, label_filter, a):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            lab = (r.get(a.label_col) or "").strip()
            if lab == "" or (label_filter is not None and lab != label_filter):
                continue
            text = "\n".join(CTRL.sub(" ", r.get(c, "") or "") for c in a.text_cols)
            rows.append({"text": text, "y": int(lab == a.positive),
                         "date": r.get(a.date_col, "") or ""})
    return rows


# ── the diagnostics ──────────────────────────────────────────────────────────
def build_dataset(a):
    data, sheet = [], {}
    rng = random.Random(12345)
    for spec in a.source:
        name, path, lf = parse_spec(spec)
        rows = read_source(path, lf, a)
        by = defaultdict(list)
        for r in rows:
            by[r["y"]].append(r)
        sheet[name] = {"file": path.name, "label_filter": lf}
        for y, g in by.items():
            yrs = [year_of(r["date"]) for r in g]
            ok = [v for v in yrs if v]
            c = Counter(ok)
            sheet[name][f"class_{y}"] = {
                "n": len(g), "dated_share": round(len(ok) / len(g), 4),
                "year_min": min(ok) if ok else None, "year_max": max(ok) if ok else None,
                # 1st-99th percentile: robust to a few mis-dated emails
                "year_p01": int(np.percentile(ok, 1)) if ok else None,
                "year_p99": int(np.percentile(ok, 99)) if ok else None,
                "top_years": c.most_common(3)}
            if a.max_per_group and len(g) > a.max_per_group:
                g = rng.sample(g, a.max_per_group)
            for r in g:
                data.append({**r, "src": name, "year": year_of(r["date"])})
        sheet[name]["licence"] = a.licence_map.get(name, "not stated")
    return data, sheet


def part_alignment(data, sheet, seeds):
    X = [d["text"] for d in data]; y = [d["y"] for d in data]; s = [d["src"] for d in data]
    # purity from the FULL counts in the datasheet, not from the capped sample
    purity = {}
    for name in sorted(set(s)):
        ns = [v["n"] for k, v in sheet[name].items() if k.startswith("class_")]
        purity[name] = round(max(ns) / sum(ns), 4)
    res = defaultdict(list)
    if len(set(y)) < 2:
        return {"purity": purity, "note": "only one class present"}
    for seed in seeds:
        idx = list(range(len(y)))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        M = pipe(); M.fit([X[i] for i in tr], [y[i] for i in tr])
        res["supervised"].append(metrics([y[i] for i in te], [int(v) for v in M.predict([X[i] for i in te])]))
        # each source's majority label, from the full (uncapped) counts
        maj = {name: int(sheet[name].get("class_1", {}).get("n", 0) >=
                         sheet[name].get("class_0", {}).get("n", 0)) for name in set(s)}
        res["source_oracle"].append(metrics([y[i] for i in te], [maj[s[i]] for i in te]))
        if len(set(s)) > 1:
            P = pipe(balanced=False); P.fit([X[i] for i in tr], [s[i] for i in tr])
            ps = list(P.predict([X[i] for i in te]))
            res["corpus_id_acc"].append(sum(a == s[i] for a, i in zip(ps, te)) / len(te))
            res["label_free"].append(metrics([y[i] for i in te], [maj[v] for v in ps]))
    out = {"purity": purity}
    for k in ("supervised", "source_oracle", "label_free"):
        if res[k]:
            out[k] = {"f1": agg([m["f1"] for m in res[k]]),
                      "balanced_accuracy": agg([m["balanced_accuracy"] for m in res[k]])}
    if res["corpus_id_acc"]:
        out["corpus_id_accuracy"] = agg(res["corpus_id_acc"])
    return out


def part_transfer(data, a, seeds):
    """--transfer SRC:REF=PATH[:LABEL]"""
    out = {}
    for spec in a.transfer:
        src_name, refspec = spec.split(":", 1)
        ref_name, ref_path, ref_lf = parse_spec(refspec)
        ref_rows = read_source(ref_path, ref_lf, a)
        src_rows = [d for d in data if d["src"] == src_name]
        src_label = Counter(d["y"] for d in src_rows).most_common(1)[0][0]
        f1s, bals, nulls = [], [], []
        for seed in seeds:
            rng = random.Random(seed)
            # evaluate on a held-out 20% so the detector never sees test emails
            idx = list(range(len(data)))
            _, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                     stratify=[d["y"] for d in data])
            te_set = set(te)
            train_src = [d["text"] for i, d in enumerate(data) if d["src"] == src_name and i not in te_set]
            n = min(len(train_src), len(ref_rows))
            if a.max_per_group:
                n = min(n, a.max_per_group)
            Xp = rng.sample(train_src, n) + [r["text"] for r in rng.sample(ref_rows, n)]
            for shuffled, store in ((False, None), (True, nulls)):
                yp = [1] * n + [0] * n
                if shuffled:
                    rng.shuffle(yp)
                D = pipe(); D.fit(Xp, yp)
                looks_src = D.predict([data[i]["text"] for i in te])
                pred = [src_label if v == 1 else 1 - src_label for v in looks_src]
                m = metrics([data[i]["y"] for i in te], pred)
                if shuffled:
                    nulls.append(m["f1"])
                else:
                    f1s.append(m["f1"]); bals.append(m["balanced_accuracy"])
        out[f"{src_name}_vs_{ref_name}"] = {
            "detector": f"is-{src_name} vs {ref_name} (both class {src_label}); "
                        f"looks like {src_name} -> class {src_label}",
            "f1": agg(f1s), "balanced_accuracy": agg(bals), "null_f1": agg(nulls)}
    return out


def part_separability(data, seeds, n_per=400):
    out = {}
    for cls in (1, 0):
        groups = defaultdict(list)
        for d in data:
            if d["y"] == cls:
                groups[d["src"]].append(d["text"])
        names = sorted(n for n, g in groups.items() if len(g) >= n_per)
        if len(names) < 2:
            out[f"class_{cls}"] = {"note": f"needs at least 2 sources with >= {n_per} class-{cls} "
                                           "emails; skipped"}
            continue
        big = max(names, key=lambda n: len(groups[n]))
        accs, nulls = [], []
        for seed in seeds:
            rng = random.Random(seed)
            Xs, ys = [], []
            for n in names:
                Xs += rng.sample(groups[n], n_per); ys += [n] * n_per
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            accs.append(float(np.mean(cross_val_score(pipe(False), Xs, ys, cv=cv))))
            half = rng.sample(groups[big], 2 * n_per)
            nulls.append(float(np.mean(cross_val_score(pipe(False), half, [0] * n_per + [1] * n_per, cv=cv))))
        out[f"class_{cls}"] = {"sources": names, "chance": round(100 / len(names), 2),
                               "accuracy": agg(accs), "null_source": big, "null_accuracy": agg(nulls)}
    return out


def year_majority(train, test):
    by = defaultdict(Counter)
    for yr, lab in train:
        by[yr][lab] += 1
    overall = Counter(l for _, l in train).most_common(1)[0][0]
    return metrics([l for _, l in test],
                   [by[yr].most_common(1)[0][0] if yr in by else overall for yr, _ in test])


def part_date(data, seeds):
    dated = [d for d in data if d["year"]]
    if not dated:
        return {"note": "no parseable dates"}
    out = {"dated_share": round(len(dated) / len(data), 4)}

    def run(rows):
        if len(set(d["y"] for d in rows)) < 2 or len(rows) < 50:
            return None
        bals, f1s = [], []
        for seed in seeds:
            r = rows[:]; random.Random(seed).shuffle(r)
            cut = int(0.8 * len(r))
            m = year_majority([(d["year"], d["y"]) for d in r[:cut]],
                              [(d["year"], d["y"]) for d in r[cut:]])
            bals.append(m["balanced_accuracy"]); f1s.append(m["f1"])
        return {"n": len(rows), "balanced_accuracy": agg(bals), "f1": agg(f1s)}

    out["overall"] = run(dated)
    out["within_source"] = {}
    for name in sorted(set(d["src"] for d in dated)):
        r = run([d for d in dated if d["src"] == name])
        if r:
            out["within_source"][name] = r
    by_cls = {c: [d["year"] for d in dated if d["y"] == c] for c in (0, 1)}
    if by_cls[0] and by_cls[1]:
        out["year_range_by_class"] = {str(c): [int(np.percentile(v, 1)), int(np.percentile(v, 99))]
                                      for c, v in by_cls.items()}   # 1st-99th percentile
    return out


def part_loso(data):
    out = {}
    names = sorted(set(d["src"] for d in data))
    if len(names) < 2:
        return {names[0]: {"note": "only one source; nothing to hold out"}}
    for name in names:
        tr = [d for d in data if d["src"] != name]
        te = [d for d in data if d["src"] == name]
        if len(set(d["y"] for d in tr)) < 2:
            out[name] = {"note": "remaining sources contain one class only; skipped"}
            continue
        M = pipe(); M.fit([d["text"] for d in tr], [d["y"] for d in tr])
        p = [int(v) for v in M.predict([d["text"] for d in te])]
        yy = [d["y"] for d in te]
        m = metrics(yy, p)
        single = len(set(yy)) == 1
        out[name] = {"n": len(te), "single_class": single,
                     "flagged_share": round(sum(p) / len(p), 4),
                     "f1": None if single else round(100 * m["f1"], 2),
                     "recall": round(100 * m["recall"], 2) if sum(yy) else None,
                     "fpr": round(100 * m["fpr"], 2) if len(yy) - sum(yy) else None}
    return out


def part_exposure(data, target, seeds, ks):
    tgt = [d for d in data if d["src"] == target]
    rest = [d for d in data if d["src"] != target]
    if len(set(d["y"] for d in rest)) < 2:
        return {"note": "other sources contain one class only"}
    n_test = max(1, len(tgt) // 3)
    ks = [k for k in ks if k <= len(tgt) - n_test]
    runs = defaultdict(list)
    for seed in seeds:
        rng = random.Random(seed)
        t = tgt[:]; rng.shuffle(t)
        t_test, t_pool = t[:n_test], t[n_test:]
        r = rest[:]; rng.shuffle(r)
        cut = int(0.8 * len(r))
        r_tr, r_te = r[:cut], r[cut:]
        for k in ks:
            tr = r_tr + t_pool[:k]
            M = pipe(); M.fit([d["text"] for d in tr], [d["y"] for d in tr])
            pt = [int(v) for v in M.predict([d["text"] for d in t_test])]
            mt = metrics([d["y"] for d in t_test], pt)
            po = [int(v) for v in M.predict([d["text"] for d in r_te])]
            mo = metrics([d["y"] for d in r_te], po)
            runs[k].append({"target_accuracy": mt["accuracy"], "other_f1": mo["f1"]})
    return {"target": target, "n_test": n_test,
            "by_k": {str(k): {"target_accuracy": agg([x["target_accuracy"] for x in v]),
                              "other_f1": agg([x["other_f1"] for x in v])}
                     for k, v in runs.items()}}


# ── report ───────────────────────────────────────────────────────────────────
def verdict(rep):
    w = []
    al = rep.get("alignment", {})
    single = [n for n, p in al.get("purity", {}).items() if p == 1.0]
    if single:
        w.append(f"Single-class sources (label is fixed by source): {', '.join(single)}. "
                 "Any detector that recognises these sources gets their label for free.")
    sup = al.get("supervised", {}).get("f1")
    lf = al.get("label_free", {}).get("f1")
    orc = al.get("source_oracle", {}).get("f1")
    for name, t in rep.get("transfer", {}).items():
        if sup and t.get("f1") and t["f1"]["mean"] >= sup["mean"] - 5:
            w.append(f"SEVERE: the provenance-transfer detector ({name}) reaches F1 "
                     f"{t['f1']['mean']:.2f} without any label of the other class, within 5 points "
                     f"of the supervised model ({sup['mean']:.2f}).")
    all_single = bool(al.get("purity")) and all(p == 1.0 for p in al["purity"].values())
    if all_single:
        w.append("Every source contains one class only, so source and label are the same variable "
                 "in this dataset: the supervised score cannot be separated from source recognition "
                 "without an external reference (use --transfer).")
    elif sup and lf and lf["mean"] >= sup["mean"] - 5:
        w.append(f"SEVERE: a label-free source detector reaches F1 {lf['mean']:.2f}, within 5 "
                 f"points of the supervised model ({sup['mean']:.2f}). The score is explained by provenance.")
    elif sup and orc and orc["mean"] < sup["mean"] - 20:
        w.append(f"Source alone gives F1 {orc['mean']:.2f} vs supervised {sup['mean']:.2f}: provenance "
                 "does not explain the overall score (sources mostly contain both classes).")
    dt = rep.get("date", {})
    ov = (dt.get("overall") or {}).get("balanced_accuracy")
    if ov and ov["mean"] >= 90:
        w.append(f"SEVERE: the email year alone gives balanced accuracy {ov['mean']:.2f}.")
    for cls, s in rep.get("separability", {}).items():
        acc = s.get("accuracy")
        if acc and acc["mean"] >= 80:
            w.append(f"Sources are recognisable with {cls.replace('_', ' ')} held fixed "
                     f"({acc['mean']:.2f}% vs chance {s['chance']}%): a provenance signal exists.")
    for name, r in rep.get("loso", {}).items():
        if r.get("single_class") and r.get("flagged_share") is not None:
            w.append(f"Leave-{name}-out: {100*r['flagged_share']:.2f}% of {name} is assigned the "
                     "positive class by a model that never saw it.")
    return w


def write_md(rep, path):
    L = [f"# Provenance report — {rep['title']}", "",
         f"Rows analysed: {rep['n_rows']} "
         + (f"(capped at {rep['max_per_group']} per source × class — numbers can differ from the uncapped data); "
            if rep['max_per_group'] else "(no cap); ")
         + f"seeds: {rep['seeds']}.", "", "## Verdict and warnings", ""]
    L += [f"- {w}" for w in rep["warnings"]] or ["- none"]
    L += ["", "## 1. Datasheet", "",
          "| Source | File | Class | n | Dated | Years (1st–99th pct.; all) | Licence |", "|---|---|---|---|---|---|---|"]
    for name, s in rep["datasheet"].items():
        for key in sorted(k for k in s if k.startswith("class_")):
            c = s[key]
            if not c["year_min"]:
                yrs = "—"
            else:
                core = (f"{c['year_p01']}" if c["year_p01"] == c["year_p99"]
                        else f"{c['year_p01']}–{c['year_p99']}")
                yrs = f"{core} (all: {c['year_min']}–{c['year_max']})"
            L.append(f"| {name} | {s['file']} | {key[-1]} | {c['n']} | "
                     f"{100*c['dated_share']:.1f}% | {yrs} | {s['licence']} |")
    al = rep["alignment"]
    L += ["", "## 2. Label–source link", "",
          "Purity (share of the majority class, from full counts) per source: " +
          ", ".join(f"{k} {100*v:.1f}%" for k, v in al.get("purity", {}).items()), "",
          "| Predictor | F1 | Balanced accuracy |", "|---|---|---|"]
    for k, lab in (("supervised", "Supervised, random 80/20 split"),
                   ("source_oracle", "Source-majority oracle (source → its majority label)"),
                   ("label_free", "Label-free source detector → majority label")):
        if k in al:
            L.append(f"| {lab} | {fmt(al[k]['f1'])} | {fmt(al[k]['balanced_accuracy'])} |")
    if "corpus_id_accuracy" in al:
        L.append(f"\nSource-identification accuracy: {fmt(al['corpus_id_accuracy'])}")
    if rep.get("transfer"):
        L += ["", "## 3. Provenance transfer", "", "| Detector | F1 | Balanced acc. | Null F1 |",
              "|---|---|---|---|"]
        for k, t in rep["transfer"].items():
            L.append(f"| {t['detector']} | {fmt(t['f1'])} | {fmt(t['balanced_accuracy'])} | {fmt(t['null_f1'])} |")
    L += ["", "## 4. Source separability with the class held fixed", ""]
    for cls, s in rep["separability"].items():
        if "note" in s:
            L.append(f"- {cls}: {s['note']}")
        else:
            L.append(f"- {cls}: {len(s['sources'])}-way accuracy {fmt(s['accuracy'])} "
                     f"(chance {s['chance']}); null ({s['null_source']} split in half) {fmt(s['null_accuracy'])}")
    dt = rep["date"]
    L += ["", "## 5. Date", ""]
    if "note" in dt:
        L.append(f"- {dt['note']}")
    else:
        ov = dt.get("overall")
        L.append(f"- Rows with a parseable date: {100*dt['dated_share']:.1f}%")
        if ov:
            L.append(f"- Year-majority rule, all dated rows: balanced accuracy {fmt(ov['balanced_accuracy'])} "
                     "(50 = chance)")
        for name, r in dt.get("within_source", {}).items():
            L.append(f"- Within {name} only: balanced accuracy {fmt(r['balanced_accuracy'])}")
        if "year_range_by_class" in dt:
            L.append("- Years by class (1st–99th percentile): " + "; ".join(
                f"class {c}: {v[0]}–{v[1]}" for c, v in dt["year_range_by_class"].items()))
    L += ["", "## 6. Leave one source out", "", "| Held-out source | n | Result |", "|---|---|---|"]
    for name, r in rep["loso"].items():
        if "note" in r:
            L.append(f"| {name} | — | {r['note']} |")
        elif r["single_class"]:
            L.append(f"| {name} | {r['n']} | {100*r['flagged_share']:.2f}% assigned the positive class |")
        else:
            L.append(f"| {name} | {r['n']} | F1 {r['f1']:.2f} (recall {r['recall']}, FPR {r['fpr']}) |")
    if rep.get("exposure"):
        ex = rep["exposure"]
        L += ["", f"## 7. Exposure sweep — {ex.get('target')}", ""]
        if "note" in ex:
            L.append(f"- {ex['note']}")
        else:
            L += [f"Held-out {ex['target']} test emails: {ex['n_test']}", "",
                  "| k added | Accuracy on held-out target | F1 on other sources |", "|---|---|---|"]
            for k, v in ex["by_k"].items():
                L.append(f"| {k} | {fmt(v['target_accuracy'])} | {fmt(v['other_f1'])} |")
    L += ["", "---", "Model: TF-IDF (1–2 grams) + logistic regression. Intervals: 95% Student t over seeds.",
          "This report measures provenance signals; it does not measure phishing-detection ability."]
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--transfer", action="append", default=[])
    ap.add_argument("--target")
    ap.add_argument("--licence", action="append", default=[])
    ap.add_argument("--text-cols", default="subject,body")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--positive", default="1")
    ap.add_argument("--max-per-group", type=int, default=0,
                    help="cap rows per source x class (0 = no cap, the default)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--title", default="dataset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.text_cols = [c.strip() for c in a.text_cols.split(",")]
    a.licence_map = dict(l.split("=", 1) for l in a.licence)
    seeds = list(range(a.seeds))

    data, sheet = build_dataset(a)
    print(f"[provenance_report] {len(data)} rows from {len(sheet)} sources", flush=True)
    rep = {"title": a.title, "n_rows": len(data), "max_per_group": a.max_per_group,
           "seeds": seeds, "datasheet": sheet}
    print("  2 label-source link", flush=True); rep["alignment"] = part_alignment(data, sheet, seeds)
    if a.transfer:
        print("  3 provenance transfer", flush=True); rep["transfer"] = part_transfer(data, a, seeds)
    print("  4 separability", flush=True); rep["separability"] = part_separability(data, seeds)
    print("  5 date", flush=True); rep["date"] = part_date(data, seeds)
    print("  6 leave one source out", flush=True); rep["loso"] = part_loso(data)
    if a.target:
        print(f"  7 exposure sweep for {a.target}", flush=True)
        rep["exposure"] = part_exposure(data, a.target, seeds, EXPOSURE_K)
    rep["warnings"] = verdict(rep)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    write_md(rep, out / "report.md")
    print(f"[provenance_report] wrote {out/'report.md'} and {out/'report.json'}")
    for w in rep["warnings"]:
        print("  !", w)


if __name__ == "__main__":
    main()
