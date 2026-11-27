"""
tune_thresholds.py  —  calibrate PhishGuard on the DEV split only
==================================================================

Reads phishguard_dev.csv (30% of the data), sweeps every reachable risk
score, and picks two operating points:

  OP-F1    the score that maximises F1          → headline number
  OP-PREC  the lowest score whose FPR <= 1%     → SOC-realistic number

Writes calibration.json. evaluate_final.py reads that file and applies
these thresholds to the held-out test set without re-tuning.

WHY IT MATTERS: PhishGuard ships with bands at 40/70/90. Those were chosen
by hand, not fitted. On CEAS_08 they put 99.3% of all mail in the Low band
and produced F1 = 0.02%. Reporting a hand-set threshold as if it were a
result is the kind of thing reviewers reject over. Fitting it on dev and
reporting on test is defensible; fitting on the reported set is not.

RUN:
    python tune_thresholds.py
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "phishguard"))
csv.field_size_limit(10_000_000)

try:
    from phishguard.dataset import load_emails, analyze_email
except ImportError as e:
    print(f"❌  Cannot import PhishGuard: {e}")
    print(f"   Expected the package at: {HERE.parent / 'phishguard'}")
    sys.exit(1)

DEV_CSV      = HERE / "phishguard_dev.csv"
CALIB_JSON   = HERE / "calibration.json"
MAX_FPR      = 0.01     # 1% — a SOC drowning in false positives stops reading alerts


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    acc  = (tp + tn) / len(y_true) if y_true else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fpr}


def main():
    if not DEV_CSV.exists():
        print(f"❌  {DEV_CSV.name} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    records = load_emails(str(DEV_CSV))
    y = [1 if r.label == "phishing" else 0 for r in records]
    analyses = [analyze_email(r) for r in records]
    scores = [a.risk_score for a in analyses]

    n_p, n_l = sum(y), len(y) - sum(y)
    print("=" * 72)
    print("  THRESHOLD CALIBRATION  —  DEV SPLIT ONLY")
    print(f"  {len(records)} emails  ({n_p} phishing / {n_l} legitimate)")
    print("=" * 72)
    print()

    candidates = sorted(set(scores) | {s + 1 for s in scores})
    candidates = [c for c in candidates if c > 0]

    rows = []
    for t in candidates:
        m = metrics(y, [int(s >= t) for s in scores])
        m["threshold"] = t
        rows.append(m)

    print(f"{'thr':>5} {'acc%':>7} {'prec%':>7} {'rec%':>7} {'F1%':>7} {'FPR%':>7}")
    print("-" * 72)
    for m in rows:
        print(f"{m['threshold']:>5} {100*m['accuracy']:>7.2f} "
              f"{100*m['precision']:>7.2f} {100*m['recall']:>7.2f} "
              f"{100*m['f1']:>7.2f} {100*m['fpr']:>7.2f}")
    print("-" * 72)

    # ── OP-F1 ────────────────────────────────────────────────────────────────
    op_f1 = max(rows, key=lambda m: (m["f1"], -m["threshold"]))

    # ── OP-PREC: best recall subject to FPR <= MAX_FPR ───────────────────────
    feasible = [m for m in rows if m["fpr"] <= MAX_FPR]
    op_prec = max(feasible, key=lambda m: (m["recall"], -m["threshold"])) if feasible else None

    print()
    print("── Selected operating points ──")
    print(f"  OP-F1    threshold={op_f1['threshold']:<4} "
          f"F1={100*op_f1['f1']:.2f}%  P={100*op_f1['precision']:.2f}%  "
          f"R={100*op_f1['recall']:.2f}%  FPR={100*op_f1['fpr']:.2f}%")
    if op_prec:
        print(f"  OP-PREC  threshold={op_prec['threshold']:<4} "
              f"F1={100*op_prec['f1']:.2f}%  P={100*op_prec['precision']:.2f}%  "
              f"R={100*op_prec['recall']:.2f}%  FPR={100*op_prec['fpr']:.2f}%"
              f"   (constraint: FPR<={100*MAX_FPR:.0f}%)")
    else:
        print(f"  OP-PREC  no threshold achieves FPR <= {100*MAX_FPR:.0f}% on dev.")

    # ── Sanity check against the shipped bands ───────────────────────────────
    shipped = next((m for m in rows if m["threshold"] == 40), None)
    if shipped:
        print()
        print("── Shipped default (Medium band, score>=40) for comparison ──")
        print(f"  F1={100*shipped['f1']:.2f}%  P={100*shipped['precision']:.2f}%  "
              f"R={100*shipped['recall']:.2f}%  FPR={100*shipped['fpr']:.2f}%")
        delta = 100 * (op_f1["f1"] - shipped["f1"])
        print(f"  Calibration gain at OP-F1: {delta:+.2f} F1 points")

    payload = {
        "split": "dev",
        "dev_rows": len(records),
        "dev_phishing": n_p,
        "dev_legitimate": n_l,
        "max_fpr_constraint": MAX_FPR,
        "operating_points": {
            "OP_F1":   {k: op_f1[k] for k in
                        ("threshold", "accuracy", "precision", "recall", "f1", "fpr")},
            "OP_PREC": ({k: op_prec[k] for k in
                         ("threshold", "accuracy", "precision", "recall", "f1", "fpr")}
                        if op_prec else None),
        },
        "shipped_default": ({k: shipped[k] for k in
                             ("threshold", "accuracy", "precision", "recall", "f1", "fpr")}
                            if shipped else None),
        "full_sweep": rows,
    }
    CALIB_JSON.write_text(json.dumps(payload, indent=2))

    print()
    print(f"✅  Calibration written → {CALIB_JSON.name}")
    print("    NEXT: python evaluate_final.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
