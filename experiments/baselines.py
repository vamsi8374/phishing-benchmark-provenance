"""
Supervised baselines on identical splits, measured on the same axes.

    python experiments/baselines.py                 # primary split
    python experiments/baselines.py --cross-corpus  # + transfer test

WHY THESE HAVE TO BE HERE
-------------------------
A reviewer will ask what a bag-of-words classifier does on this data. If the
answer is absent, the paper is rejected for an unfair comparison. If the
answer is present and unflattering, the paper survives — provided it is honest
about what the comparison does and does not settle.

It settles: raw accuracy on a fixed in-distribution corpus.

It does not settle:
  · behaviour on a corpus the classifier was not trained on (see --cross-corpus,
    which is where TF-IDF characteristically collapses),
  · whether the system can explain a verdict to an analyst,
  · whether it can run with no labelled training data for a new tenant,
  · what happens when the attacker changes vocabulary next month.

Report all four. Do not report only the flattering one.

TRAINING DISCIPLINE
-------------------
The supervised baselines get the SAME dev/test split as the cascade. They are
fitted on dev and scored once on test. Fitting on dev is a real advantage the
cascade does not have — the cascade never sees a label during inference. Say
so in the paper rather than pretending the comparison is symmetric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishcascade.config import (          # noqa: E402
    DEV_CSV, TEST_CSV, RESULTS_DIR, PHISHGUARD_DIR, DATA_DIR,
)

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    print("❌  scikit-learn required:  pip install scikit-learn")
    sys.exit(1)


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den  = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return {"accuracy": (tp+tn)/len(y_true) if y_true else 0.0,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp/(fp+tn) if (fp+tn) else 0.0,
            "mcc": ((tp*tn - fp*fn)/den) if den else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def to_text(r) -> str:
    """Same fields the cascade sees. Anything else would be an unfair edge."""
    return f"{r.sender}\n{r.subject}\n{r.body}"


def load_split(path: Path):
    recs = load_emails(str(path))
    X = [to_text(r) for r in recs]
    y = [1 if r.label == "phishing" else 0 for r in recs]
    return recs, X, y


def build_models():
    return {
        "TF-IDF word 1-2gram + LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                      ngram_range=(1, 2), min_df=2,
                                      max_features=200_000,
                                      strip_accents="unicode")),
            ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                       class_weight="balanced")),
        ]),
        "TF-IDF char 3-5gram + LinSVC": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      sublinear_tf=True, min_df=3,
                                      max_features=300_000)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced"), cv=3)),
        ]),
    }


def run(dev_csv: Path, test_csv: Path, tag: str, cross: list[tuple[str, Path]]):
    _, Xd, yd = load_split(dev_csv)
    _, Xt, yt = load_split(test_csv)

    print("=" * 88)
    print(f"  SUPERVISED BASELINES  ·  {tag}")
    print(f"  fit on dev ({len(yd)}), score on test ({len(yt)})")
    print("=" * 88)
    print()

    out = {}
    hdr = (f"{'model':<32}{'F1%':>8}{'Prec%':>8}{'Rec%':>8}{'FPR%':>8}"
           f"{'MCC':>8}{'fit s':>8}{'inf ms/1k':>11}")
    print(hdr); print("-" * len(hdr))

    for name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(Xd, yd)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = model.predict(Xt)
        inf_s = time.perf_counter() - t1

        m = metrics(yt, list(pred))
        per_1k_ms = 1000.0 * inf_s / max(1, len(yt)) * 1000.0
        print(f"{name:<32}{100*m['f1']:>8.2f}{100*m['precision']:>8.2f}"
              f"{100*m['recall']:>8.2f}{100*m['fpr']:>8.2f}{m['mcc']:>8.3f}"
              f"{fit_s:>8.1f}{per_1k_ms:>11.1f}")

        entry = {"metrics": m, "fit_seconds": fit_s,
                 "inference_seconds_per_1k": 1000.0 * inf_s / max(1, len(yt)),
                 "cross_corpus": {}}

        # ── Transfer ─────────────────────────────────────────────────────────
        for cname, cpath in cross:
            if not cpath.exists():
                continue
            _, Xc, yc = load_split(cpath)
            cm = metrics(yc, list(model.predict(Xc)))
            entry["cross_corpus"][cname] = cm

        out[name] = entry

    if cross:
        print()
        print("── Cross-corpus transfer (trained on the split above) ──")
        h2 = f"{'model':<32}{'corpus':<22}{'F1%':>8}{'Prec%':>8}{'Rec%':>8}{'MCC':>8}"
        print(h2); print("-" * len(h2))
        for name, e in out.items():
            for cname, cm in e["cross_corpus"].items():
                print(f"{name:<32}{cname:<22}{100*cm['f1']:>8.2f}"
                      f"{100*cm['precision']:>8.2f}{100*cm['recall']:>8.2f}"
                      f"{cm['mcc']:>8.3f}")
        print()
        print("  A large in-corpus/out-of-corpus gap is the honest headline for")
        print("  these baselines. It is also the argument for a system that does")
        print("  not require per-tenant labelled training data.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"baselines_{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print()
    print(f"✅  {p.name}")
    print("=" * 88)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--cross-corpus", action="store_true",
                    help="also score on CEAS_08 (out-of-distribution)")
    args = ap.parse_args()

    dev  = Path(args.dev)  if args.dev  else DEV_CSV
    test = Path(args.test) if args.test else TEST_CSV
    if not dev.exists() or not test.exists():
        print(f"❌  Missing split files. Run prepare_dataset.py in {DATA_DIR}")
        sys.exit(1)

    cross = []
    if args.cross_corpus:
        ceas = DATA_DIR / "ceas_transfer.csv"
        if ceas.exists():
            cross.append(("CEAS_08 (2008 spam)", ceas))
        else:
            print(f"ℹ️   Cross-corpus file not found: {ceas.name}")
            print("    Build it with:  python experiments/make_transfer_set.py")
            print()

    run(dev, test, args.tag, cross)


if __name__ == "__main__":
    main()
