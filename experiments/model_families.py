"""
Is the leakage profile specific to logistic regression?

    python experiments/model_families.py

WHY
---
Every other result in this project uses TF--IDF with logistic regression. A
reviewer will reasonably ask whether the channel profile is an artefact of
that particular estimator. This runs several model families over the same
views.

WHAT THIS DOES NOT SETTLE
-------------------------
None of these is a pretrained transformer. Fine-tuned transformer models hold
the reported state of the art on these corpora, and verifying that DeBERTa
shows the same profile requires downloading pretrained weights, which this
environment cannot do. `baseline_deberta.py` is written and ready for a
machine that can. Until it is run, the transformer question stays open and the
paper says so.

What this DOES settle is narrower and still worth having: the profile is not
peculiar to one linear model. If a margin classifier, a probabilistic
classifier, a tree ensemble and a small neural network all show it, the
finding is a property of the data rather than of the estimator.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DEV_CSV, TEST_CSV, RESULTS_DIR, PHISHGUARD_DIR  # noqa: E402

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.svm import LinearSVC                             # noqa: E402
from sklearn.naive_bayes import ComplementNB                  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.neural_network import MLPClassifier              # noqa: E402
from sklearn.decomposition import TruncatedSVD                # noqa: E402
from sklearn.pipeline import Pipeline                          # noqa: E402

_DOMAIN = re.compile(r"@([A-Za-z0-9.\-]+)")
_NONWORD = re.compile(r"[A-Za-z0-9]+")

VIEWS = {
    "full":          lambda r: f"{r.sender}\n{r.subject}\n{r.body}",
    "body_only":     lambda r: r.body or "",
    "date_only":     lambda r: r.date or "",
    "domain_only":   lambda r: " ".join(_DOMAIN.findall(r.sender or "")),
}


def tfidf(**kw):
    return TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                           min_df=2, max_features=200_000, **kw)


def models():
    return {
        "LogisticRegression": Pipeline([
            ("v", tfidf()),
            ("c", LogisticRegression(max_iter=2000, C=4.0,
                                     class_weight="balanced"))]),
        "LinearSVC (margin)": Pipeline([
            ("v", tfidf()),
            ("c", LinearSVC(C=1.0, class_weight="balanced"))]),
        "ComplementNB (prob.)": Pipeline([
            ("v", tfidf()),
            ("c", ComplementNB())]),
        "SGD hinge": Pipeline([
            ("v", tfidf()),
            ("c", SGDClassifier(loss="hinge", max_iter=3000, tol=1e-4,
                                class_weight="balanced", random_state=42))]),
        "HistGradientBoosting": Pipeline([
            ("v", tfidf()),
            ("svd", TruncatedSVD(n_components=200, random_state=42)),
            ("c", HistGradientBoostingClassifier(random_state=42))]),
        "MLP (1x256, from scratch)": Pipeline([
            ("v", tfidf()),
            ("svd", TruncatedSVD(n_components=300, random_state=42)),
            ("c", MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                random_state=42))]),
    }


def f1_of(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def main():
    dev = load_emails(str(DEV_CSV))
    test = load_emails(str(TEST_CSV))
    yd = [1 if r.label == "phishing" else 0 for r in dev]
    yt = [1 if r.label == "phishing" else 0 for r in test]

    print("=" * 86)
    print("  MODEL-FAMILY ROBUSTNESS — test F1 (%) by view")
    print(f"  fit on dev ({len(yd)}), score on test ({len(yt)})")
    print("=" * 86)
    print()
    hdr = f"{'model':<28}" + "".join(f"{v:>14}" for v in VIEWS)
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for mname, model in models().items():
        cells, row = [], {}
        for vname, vfn in VIEWS.items():
            Xd = [vfn(r) for r in dev]
            Xt = [vfn(r) for r in test]
            if not any(x.strip() for x in Xd):
                cells.append(f"{'n/a':>14}")
                continue
            try:
                model.fit(Xd, yd)
                f1 = f1_of(yt, list(model.predict(Xt)))
                cells.append(f"{100*f1:>14.2f}")
                row[vname] = f1
            except Exception as e:
                cells.append(f"{'err':>14}")
                row[vname] = None
                print(f"    ({mname}/{vname}: {type(e).__name__})",
                      file=sys.stderr)
        print(f"{mname:<28}" + "".join(cells))
        out[mname] = row

    print()
    print("  The pattern to look for is not the absolute numbers but whether")
    print("  date_only stays near the full-view score across families. If it")
    print("  does, the leak is in the data, not in the estimator.")
    print()
    print("  NOT COVERED: pretrained transformers. See baseline_deberta.py.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_families.json").write_text(json.dumps(out, indent=2))
    print()
    print("✅  model_families.json")
    print("=" * 86)


if __name__ == "__main__":
    main()
