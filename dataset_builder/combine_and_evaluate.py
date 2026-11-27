"""
combine_and_evaluate.py
────────────────────────
Step 3 of the PhishGuard real-dataset evaluation pipeline.

Combines phishing_nazario.csv + legitimate_enron.csv,
then runs PhishGuard heuristic engine against every email
and prints publication-quality metrics.

REQUIRES:
  • phishing_nazario.csv    (from parse_nazario.py)
  • legitimate_enron.csv    (from parse_enron.py)
  • PhishGuard installed in ../phishguard/ OR pip install phishguard

RUN:
    python combine_and_evaluate.py
"""

import csv
import json
import sys
import random
from pathlib import Path
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PHISHGUARD_DIR = HERE.parent / "phishguard"

# Add PhishGuard source to import path
if PHISHGUARD_DIR.exists():
    sys.path.insert(0, str(PHISHGUARD_DIR))

try:
    from phishguard.dataset import load_emails, analyze_email, _URGENT_KEYWORDS
    from phishguard.dataset import summarize_dataset
except ImportError as e:
    print(f"❌  Cannot import PhishGuard: {e}")
    print(f"   Ensure PhishGuard is at: {PHISHGUARD_DIR}")
    print("   Or run: pip install phishguard")
    sys.exit(1)

PHISHING_CSV   = HERE / "phishing_nazario.csv"
LEGIT_CSV      = HERE / "legitimate_enron.csv"
COMBINED_CSV   = HERE / "combined_real_dataset.csv"
RESULTS_JSON   = HERE / "real_evaluation_results.json"

THRESHOLD_MEDIUM = 40
THRESHOLD_HIGH   = 70
RANDOM_SEED      = 42


def load_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10_000_000)
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_combined_csv():
    """Merge phishing + legit CSVs into one balanced dataset."""
    if not PHISHING_CSV.exists():
        print(f"❌  Missing: {PHISHING_CSV.name}")
        print("   Run parse_nazario.py first.")
        sys.exit(1)
    if not LEGIT_CSV.exists():
        print(f"❌  Missing: {LEGIT_CSV.name}")
        print("   Run parse_enron.py first.")
        sys.exit(1)

    phishing_rows = load_csv(PHISHING_CSV)
    legit_rows    = load_csv(LEGIT_CSV)

    print(f"Loaded {len(phishing_rows)} phishing + {len(legit_rows)} legitimate emails")

    # Balance: take up to min(N_phishing, N_legit) from each side
    n = min(len(phishing_rows), len(legit_rows))
    random.seed(RANDOM_SEED)
    random.shuffle(phishing_rows)
    random.shuffle(legit_rows)
    combined = phishing_rows[:n] + legit_rows[:n]
    random.shuffle(combined)

    print(f"Balanced dataset: {n} phishing + {n} legitimate = {len(combined)} total")

    # Write combined CSV with PhishGuard-compatible columns
    fieldnames = ["sender", "receiver", "date", "subject", "body", "urls", "label"]
    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    print(f"✓ Combined CSV → {COMBINED_CSV.name}")
    return n


def compute_metrics(y_true, y_pred, label="Model") -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr       = fp / (fp + tn) if (fp + tn) else 0

    return {
        "label":     label,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy  * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall    * 100, 2),
        "f1":        round(f1        * 100, 2),
        "fpr":       round(fpr       * 100, 2),
    }


def baseline_keyword(record) -> int:
    text = (record.subject + " " + record.body).lower()
    return int(any(k in text for k in _URGENT_KEYWORDS))


def baseline_url(record) -> int:
    return int(bool(record.urls.strip()))


def phishguard_predict(record, threshold: int) -> int:
    analysis = analyze_email(record)
    return int(analysis.risk_score >= threshold)


def run_evaluation():
    print()
    print("Loading combined dataset …")
    records = load_emails(str(COMBINED_CSV))
    y_true  = [1 if r.label in ("1", "phishing") else 0 for r in records]

    n_phish = sum(y_true)
    n_legit = len(y_true) - n_phish
    print(f"Evaluating {len(records)} emails ({n_phish} phishing / {n_legit} legitimate)")
    print()

    y_b1   = [baseline_keyword(r)             for r in records]
    y_b2   = [baseline_url(r)                 for r in records]
    y_pg_m = [phishguard_predict(r, THRESHOLD_MEDIUM) for r in records]
    y_pg_h = [phishguard_predict(r, THRESHOLD_HIGH)   for r in records]

    results = [
        compute_metrics(y_true, y_b1,   "B1: Keyword-only"),
        compute_metrics(y_true, y_b2,   "B2: URL-presence"),
        compute_metrics(y_true, y_pg_m, "PhishGuard (Medium+, score≥40)"),
        compute_metrics(y_true, y_pg_h, "PhishGuard (High+,   score≥70)"),
    ]

    # ── Per-flag breakdown ──────────────────────────────────────────────────
    flag_counts = defaultdict(lambda: {"tp": 0, "fp": 0})
    for r, truth in zip(records, y_true):
        analysis = analyze_email(r)
        for flag in analysis.flags:
            key = flag.split(":")[0].strip()
            if truth == 1:
                flag_counts[key]["tp"] += 1
            else:
                flag_counts[key]["fp"] += 1

    # ── Risk band distribution ──────────────────────────────────────────────
    band_dist = {"phishing": defaultdict(int), "legitimate": defaultdict(int)}
    for r, truth in zip(records, y_true):
        analysis = analyze_email(r)
        cat = "phishing" if truth == 1 else "legitimate"
        band_dist[cat][analysis.risk_band] += 1

    # ── Campaign detection ──────────────────────────────────────────────────
    summary = summarize_dataset(str(COMBINED_CSV), records)

    # ── Print ───────────────────────────────────────────────────────────────
    print("=" * 74)
    print("  PHISHGUARD REAL-DATA EVALUATION RESULTS")
    print(f"  Dataset: {len(records)} emails | {n_phish} phishing (Nazario) | "
          f"{n_legit} legitimate (Enron)")
    print("=" * 74)
    header = f"{'Method':<46} {'Acc%':>6} {'Prec%':>7} {'Rec%':>6} {'F1%':>6} {'FPR%':>6}"
    print(f"\n{header}")
    print("-" * 74)
    for r in results:
        print(f"{r['label']:<46} {r['accuracy']:>6} {r['precision']:>7} "
              f"{r['recall']:>6} {r['f1']:>6} {r['fpr']:>6}")
    print("-" * 74)

    print("\n── Confusion Matrices ──")
    for r in results:
        print(f"\n{r['label']}")
        print(f"  TP={r['tp']:5d}  FP={r['fp']:5d}")
        print(f"  FN={r['fn']:5d}  TN={r['tn']:5d}")

    print("\n── PhishGuard Flag Contribution (Medium+ threshold) ──")
    print(f"{'Flag':<46} {'TP hits':>8} {'FP hits':>8}")
    print("-" * 64)
    for flag, counts in sorted(flag_counts.items(), key=lambda x: -x[1]["tp"]):
        print(f"{flag:<46} {counts['tp']:>8} {counts['fp']:>8}")

    print("\n── Risk Band Distribution ──")
    print(f"{'Band':<12} {'Phishing':>10} {'Legitimate':>12}")
    print("-" * 36)
    for band in ["Critical", "High", "Medium", "Low"]:
        ph = band_dist["phishing"].get(band, 0)
        le = band_dist["legitimate"].get(band, 0)
        print(f"{band:<12} {ph:>10} {le:>12}")

    print(f"\n── Campaign Detection ──")
    print(f"  Campaigns detected: {len(summary.campaigns)}")
    if summary.campaigns:
        largest = summary.campaigns[0]
        print(f"  Largest campaign:   {largest.email_count} emails, "
              f"{largest.unique_recipients} unique recipients")
        if summary.top_senders:
            print(f"  Top sender:         {summary.top_senders[0][0]}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    output = {
        "dataset_stats": {
            "total":      len(records),
            "phishing":   n_phish,
            "legitimate": n_legit,
            "sources":    {"phishing": "Nazario Phishing Corpus", "legitimate": "Enron Email Corpus"},
        },
        "metrics": results,
        "flag_contribution": {k: v for k, v in flag_counts.items()},
        "risk_band_distribution": {cat: dict(bands) for cat, bands in band_dist.items()},
        "campaign_detection": {
            "campaigns_found":       len(summary.campaigns),
            "largest_campaign_size": summary.campaigns[0].email_count if summary.campaigns else 0,
            "top_senders":           summary.top_senders[:5],
            "top_tlds":              summary.top_tlds[:5],
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"✅  Results saved → {RESULTS_JSON.name}")
    print("=" * 74)
    return output


def main():
    build_combined_csv()
    print()
    run_evaluation()


if __name__ == "__main__":
    main()
