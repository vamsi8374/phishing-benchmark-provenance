"""
evaluate_final.py  —  held-out TEST metrics for the paper
==========================================================

Applies the thresholds chosen by tune_thresholds.py (on dev) to
phishguard_test.csv, which was never used for calibration.

Produces:
  · Main results table   — PhishGuard at two operating points vs 3 baselines
  · Per-signal ablation  — leave-one-out, so you can say which rule earns its keep
  · Signal contingency   — how often each signal fires on phishing vs legit
  · PR + ROC curve data  — for the figure
  · results_test.json    — machine-readable, everything above
  · tables.tex           — paste-ready LaTeX for the main table + ablation

RUN:
    python evaluate_final.py
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "phishguard"))
csv.field_size_limit(10_000_000)

try:
    from phishguard.dataset import (
        load_emails, analyze_email, SIGNAL_WEIGHTS, _URGENT_KEYWORDS,
    )
except ImportError as e:
    print(f"❌  Cannot import PhishGuard: {e}")
    sys.exit(1)

TEST_CSV   = HERE / "phishguard_test.csv"
CALIB_JSON = HERE / "calibration.json"
OUT_JSON   = HERE / "results_test.json"
OUT_TEX    = HERE / "tables.tex"


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label=""):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    acc  = (tp + tn) / len(y_true) if y_true else 0.0
    # Matthews correlation — robust to any residual class imbalance
    den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc = ((tp*tn - fp*fn) / den) if den else 0.0
    return {"label": label, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fpr, "mcc": mcc}


def fmt(m):
    return (f"{100*m['accuracy']:>7.2f} {100*m['precision']:>7.2f} "
            f"{100*m['recall']:>7.2f} {100*m['f1']:>7.2f} "
            f"{100*m['fpr']:>7.2f} {m['mcc']:>7.3f}")


# ── Baselines ─────────────────────────────────────────────────────────────────
def b_keyword(r):
    return int(any(k in (r.subject + " " + r.body).lower() for k in _URGENT_KEYWORDS))


def b_url(r):
    return int("http" in (r.body or "").lower() or bool((r.urls or "").strip()))


def b_always(r):
    return 1   # majority-class baseline; exposes inflated-accuracy claims


def score_without(record, drop_signal):
    """Recompute the risk score with one signal zeroed — leave-one-out ablation."""
    w = dict(SIGNAL_WEIGHTS)
    w[drop_signal] = 0
    return analyze_email(record, weights=w).risk_score


# ── Curves ────────────────────────────────────────────────────────────────────
def curves(y, scores):
    """Return PR and ROC points plus AUCs, computed over all thresholds."""
    pts = []
    for t in sorted(set(scores) | {max(scores) + 1}):
        m = metrics(y, [int(s >= t) for s in scores])
        pts.append({"threshold": t, "precision": m["precision"],
                    "recall": m["recall"], "fpr": m["fpr"], "tpr": m["recall"]})
    pts.sort(key=lambda p: p["recall"])
    # Trapezoidal AUCs
    def auc(xs, ys):
        return sum((xs[i+1]-xs[i]) * (ys[i+1]+ys[i]) / 2 for i in range(len(xs)-1))
    pr_r = [p["recall"] for p in pts]
    pr_p = [p["precision"] for p in pts]
    roc = sorted(pts, key=lambda p: p["fpr"])
    roc_x = [p["fpr"] for p in roc]
    roc_y = [p["tpr"] for p in roc]
    return pts, abs(auc(pr_r, pr_p)), abs(auc(roc_x, roc_y))


def main():
    if not TEST_CSV.exists():
        print(f"❌  {TEST_CSV.name} not found. Run prepare_dataset.py first.")
        sys.exit(1)
    if not CALIB_JSON.exists():
        print(f"❌  {CALIB_JSON.name} not found. Run tune_thresholds.py first.")
        sys.exit(1)

    calib = json.loads(CALIB_JSON.read_text())
    ops = calib["operating_points"]
    t_f1 = ops["OP_F1"]["threshold"]
    t_pr = ops["OP_PREC"]["threshold"] if ops.get("OP_PREC") else None

    records = load_emails(str(TEST_CSV))
    y = [1 if r.label == "phishing" else 0 for r in records]
    analyses = [analyze_email(r) for r in records]
    scores = [a.risk_score for a in analyses]
    n_p, n_l = sum(y), len(y) - sum(y)

    print("=" * 84)
    print("  PHISHGUARD — HELD-OUT TEST RESULTS")
    print(f"  {len(records)} emails ({n_p} phishing / {n_l} legitimate)")
    print(f"  Thresholds fitted on dev ({calib['dev_rows']} emails), never on this set.")
    print("=" * 84)
    print()

    # ── Main table ───────────────────────────────────────────────────────────
    results = [
        metrics(y, [b_always(r) for r in records], "B0: flag-everything"),
        metrics(y, [b_keyword(r) for r in records], "B1: urgency keyword only"),
        metrics(y, [b_url(r) for r in records],     "B2: contains-a-URL"),
        metrics(y, [int(s >= 40) for s in scores],  "PhishGuard (shipped, >=40)"),
        metrics(y, [int(s >= t_f1) for s in scores], f"PhishGuard OP-F1 (>={t_f1})"),
    ]
    if t_pr is not None:
        results.append(metrics(y, [int(s >= t_pr) for s in scores],
                               f"PhishGuard OP-PREC (>={t_pr})"))

    hdr = f"{'Method':<34}{'Acc%':>7}{'Prec%':>8}{'Rec%':>8}{'F1%':>8}{'FPR%':>8}{'MCC':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(f"{m['label']:<34}{fmt(m)}")
    print("-" * len(hdr))
    print()
    print("  B0 exists to make a point: on a balanced set it scores 50% accuracy")
    print("  and ~66% F1 while being useless. Any method you report must clear it.")

    # ── Curves ───────────────────────────────────────────────────────────────
    pts, pr_auc, roc_auc = curves(y, scores)
    print()
    print(f"── Curves ──   PR-AUC = {pr_auc:.4f}    ROC-AUC = {roc_auc:.4f}")
    print("  (curve points are in results_test.json under 'curve_points')")

    # ── Signal contingency ───────────────────────────────────────────────────
    print()
    print("── Signal contingency: how often each rule fires ──")
    fire_p, fire_l = Counter(), Counter()
    for a, t in zip(analyses, y):
        for s in a.signals:
            (fire_p if t == 1 else fire_l)[s] += 1

    h2 = f"{'Signal':<24}{'wt':>5}{'phish%':>9}{'legit%':>9}{'lift':>8}"
    print(h2)
    print("-" * len(h2))
    contingency = {}
    for sig in sorted(SIGNAL_WEIGHTS, key=lambda s: -SIGNAL_WEIGHTS[s]):
        pp = 100 * fire_p[sig] / n_p if n_p else 0
        pl = 100 * fire_l[sig] / n_l if n_l else 0
        lift = (pp / pl) if pl else float("inf")
        lift_s = "∞" if lift == float("inf") else f"{lift:.2f}"
        print(f"{sig:<24}{SIGNAL_WEIGHTS[sig]:>5}{pp:>9.2f}{pl:>9.2f}{lift_s:>8}")
        contingency[sig] = {"weight": SIGNAL_WEIGHTS[sig],
                            "phishing_fire_pct": pp, "legit_fire_pct": pl,
                            "lift": None if pl == 0 else lift}
    print()
    print("  lift = phish% / legit%.  A signal at lift ~1.0 is noise: it fires")
    print("  equally on both classes and only adds score variance. Anything")
    print("  below 1.0 is actively harmful and should be dropped or reweighted.")

    # ── Ablation ─────────────────────────────────────────────────────────────
    print()
    print("── Leave-one-out ablation at OP-F1 ──")
    base = metrics(y, [int(s >= t_f1) for s in scores])
    h3 = f"{'Removed signal':<24}{'F1%':>9}{'ΔF1':>9}{'Prec%':>9}{'Rec%':>9}"
    print(h3)
    print("-" * len(h3))
    print(f"{'(none — full model)':<24}{100*base['f1']:>9.2f}{'—':>9}"
          f"{100*base['precision']:>9.2f}{100*base['recall']:>9.2f}")
    ablation = {}
    for sig in sorted(SIGNAL_WEIGHTS, key=lambda s: -SIGNAL_WEIGHTS[s]):
        sc = [score_without(r, sig) for r in records]
        m = metrics(y, [int(s >= t_f1) for s in sc])
        d = 100 * (m["f1"] - base["f1"])
        print(f"{sig:<24}{100*m['f1']:>9.2f}{d:>+9.2f}"
              f"{100*m['precision']:>9.2f}{100*m['recall']:>9.2f}")
        ablation[sig] = {"f1": m["f1"], "delta_f1_points": d,
                         "precision": m["precision"], "recall": m["recall"]}
    print()
    print("  ΔF1 is the cost of removing that signal. Near-zero means the rule")
    print("  is carrying no weight and belongs in the 'future work' pile, not")
    print("  in a contribution list.")

    # ── Write JSON ───────────────────────────────────────────────────────────
    payload = {
        "split": "test (held out)",
        "n_rows": len(records), "n_phishing": n_p, "n_legitimate": n_l,
        "thresholds_from": "calibration.json (fitted on dev split)",
        "thresholds": {"OP_F1": t_f1, "OP_PREC": t_pr, "shipped": 40},
        "main_results": results,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "curve_points": pts,
        "signal_contingency": contingency,
        "ablation_at_OP_F1": ablation,
        "score_histogram": dict(Counter(scores)),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # ── Write LaTeX ──────────────────────────────────────────────────────────
    tex = []
    tex.append("% Main results — held-out test split")
    tex.append("\\begin{table}[t]\\centering")
    tex.append("\\caption{PhishGuard on the held-out test split. Thresholds were "
               "selected on a disjoint 30\\% development split; the test split "
               "was used once, for this table.}")
    tex.append("\\label{tab:main}")
    tex.append("\\begin{tabular}{lrrrrrr}")
    tex.append("\\toprule")
    tex.append("Method & Acc. & Prec. & Rec. & F1 & FPR & MCC \\\\")
    tex.append("\\midrule")
    for m in results:
        lbl = m["label"].replace("_", "\\_").replace(">=", "$\\geq$")
        tex.append(f"{lbl} & {100*m['accuracy']:.2f} & {100*m['precision']:.2f} & "
                   f"{100*m['recall']:.2f} & {100*m['f1']:.2f} & "
                   f"{100*m['fpr']:.2f} & {m['mcc']:.3f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("% Leave-one-out ablation")
    tex.append("\\begin{table}[t]\\centering")
    tex.append("\\caption{Leave-one-out signal ablation at the OP-F1 operating point.}")
    tex.append("\\label{tab:ablation}")
    tex.append("\\begin{tabular}{lrrrr}")
    tex.append("\\toprule")
    tex.append("Removed signal & Weight & F1 & $\\Delta$F1 & Recall \\\\")
    tex.append("\\midrule")
    tex.append(f"\\textit{{none (full model)}} & -- & {100*base['f1']:.2f} & -- & "
               f"{100*base['recall']:.2f} \\\\")
    for sig, a in ablation.items():
        tex.append(f"\\texttt{{{sig.replace('_', chr(92)+'_')}}} & "
                   f"{SIGNAL_WEIGHTS[sig]} & {100*a['f1']:.2f} & "
                   f"{a['delta_f1_points']:+.2f} & {100*a['recall']:.2f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    OUT_TEX.write_text("\n".join(tex), encoding="utf-8")

    print()
    print(f"✅  {OUT_JSON.name}   full results, curve points, histogram")
    print(f"✅  {OUT_TEX.name}    paste-ready LaTeX for both tables")
    print("=" * 84)


if __name__ == "__main__":
    main()
