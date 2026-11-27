"""
Is the benchmark measuring phishing, or measuring which corpus a mail came from?

    python experiments/leakage_probe.py

MOTIVATION
----------
A TF-IDF + logistic regression model, fitted on 930 emails, scores F1 99.8% on
the standard Nazario-vs-SpamAssassin pairing. Models that simple do not solve
hard problems. When a linear bag-of-words model saturates a security
benchmark, the first hypothesis should be that the two classes differ by
provenance artefacts rather than by the property of interest.

The two corpora were collected years apart, by different people, through
different mail infrastructure. Anything that leaks that difference — MTA
footers, date formatting, encoding declarations, list-serv boilerplate,
recipient-domain conventions — is a free separator that has nothing to do with
whether a message is phishing.

METHOD
------
Fit the same classifier on progressively more impoverished views of the email.
If a view that contains no persuasive content still separates the classes,
that view is leaking provenance.

  full          sender + subject + body        (the standard setup)
  body_only     body                           (content)
  subject_only  subject                        (content)
  sender_only   the From line                  (identity, not persuasion)
  domain_only   the sender's domain            (pure provenance)
  date_only     the Date header                (pure provenance)
  nonword       only punctuation/whitespace/digit shape of the body
                                               (no vocabulary whatsoever)

A high score on domain_only, date_only, or nonword is not a detection result.
It is a measurement of how separable the collection process was.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DEV_CSV, TEST_CSV, RESULTS_DIR, PHISHGUARD_DIR, DATA_DIR  # noqa: E402

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import Pipeline                       # noqa: E402


def f1_of(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, (tp + tn) / len(y_true)


_DOMAIN = re.compile(r"@([A-Za-z0-9.\-]+)")
_NONWORD = re.compile(r"[A-Za-z0-9]+")


VIEWS = {
    "full  (sender+subject+body)": lambda r: f"{r.sender}\n{r.subject}\n{r.body}",
    "body_only":     lambda r: r.body or "",
    "subject_only":  lambda r: r.subject or "",
    "sender_only":   lambda r: r.sender or "",
    "domain_only":   lambda r: " ".join(_DOMAIN.findall(r.sender or "")),
    "date_only":     lambda r: r.date or "",
    "nonword_shape": lambda r: _NONWORD.sub("w", (r.body or "")[:3000]),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanitized", action="store_true",
                    help="probe the sanitised split instead of the raw one")
    ap.add_argument("--corpus", default=None,
                    help="probe an arbitrary <name>_dev.csv / <name>_test.csv pair")
    args = ap.parse_args()

    if args.corpus:
        dev_p  = DATA_DIR / f"{args.corpus}_dev.csv"
        test_p = DATA_DIR / f"{args.corpus}_test.csv"
        tag = args.corpus
    elif args.sanitized:
        dev_p, test_p = DATA_DIR / "san_dev.csv", DATA_DIR / "san_test.csv"
        if not dev_p.exists():
            print("Run experiments/sanitize.py first.")
            sys.exit(1)
        tag = "sanitized"
    else:
        dev_p, test_p, tag = DEV_CSV, TEST_CSV, "raw"

    dev  = load_emails(str(dev_p))
    test = load_emails(str(test_p))
    yd = [1 if r.label == "phishing" else 0 for r in dev]
    yt = [1 if r.label == "phishing" else 0 for r in test]

    print("=" * 84)
    print(f"  LEAKAGE PROBE [{tag}] — what is separating the classes?")
    print(f"  fit on dev ({len(yd)}), score on test ({len(yt)})")
    print("=" * 84)
    print()
    hdr = f"{'view':<30}{'F1%':>9}{'Prec%':>9}{'Rec%':>9}{'Acc%':>9}"
    print(hdr); print("-" * len(hdr))

    out = {}
    for name, fn in VIEWS.items():
        Xd = [fn(r) for r in dev]
        Xt = [fn(r) for r in test]
        if not any(x.strip() for x in Xd):
            print(f"{name:<30}{'(empty view)':>36}")
            continue
        analyzer = "char_wb" if name == "nonword_shape" else "word"
        ngram = (2, 4) if name == "nonword_shape" else (1, 2)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                      analyzer=analyzer, ngram_range=ngram,
                                      min_df=2, max_features=200_000)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                       class_weight="balanced")),
        ])
        try:
            pipe.fit(Xd, yd)
            f1, prec, rec, acc = f1_of(yt, list(pipe.predict(Xt)))
        except ValueError as e:
            print(f"{name:<30}{('skipped: ' + str(e)[:30]):>36}")
            continue
        print(f"{name:<30}{100*f1:>9.2f}{100*prec:>9.2f}"
              f"{100*rec:>9.2f}{100*acc:>9.2f}")
        out[name] = {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}

    print()
    print("  READ THIS CAREFULLY:")
    print("  domain_only, date_only and nonword_shape contain NO persuasive")
    print("  content. A high score on any of them means the benchmark is")
    print("  separable by collection artefacts. Any headline accuracy on this")
    print("  corpus pairing must be discounted by that amount.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"leakage_probe_{tag}.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"✅  leakage_probe_{tag}.json")
    print("=" * 84)


if __name__ == "__main__":
    main()
