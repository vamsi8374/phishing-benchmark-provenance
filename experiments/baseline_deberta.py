"""
DeBERTa-v3 fine-tune baseline.

    pip install torch transformers datasets accelerate
    python experiments/baseline_deberta.py --split primary
    python experiments/baseline_deberta.py --split sanitized --cross-corpus

This is the baseline a reviewer will insist on, because the transformer
fine-tune line of work (SecureNet and successors) is what currently holds the
reported state of the art on these corpora.

READ THIS BEFORE INTERPRETING THE OUTPUT
----------------------------------------
It will score very high — likely 99%+ — on the in-corpus test set. That number
is not evidence that the model detects phishing. The leakage probe shows the
Date header alone reaches F1 99.91% on this corpus pairing, so a model with
access to the full message can reach the same place without learning anything
about phishing at all.

The number that means something is the cross-corpus one. Run with
--cross-corpus and report both, adjacent, in the same table. Reporting only
the in-corpus figure is the error this project exists to document.

Weights are downloaded from HuggingFace on first run (~350 MB for
deberta-v3-small). If the machine is offline, fetch them first on a connected
machine and point --model at the local directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR, RESULTS_DIR, PHISHGUARD_DIR  # noqa: E402

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
except ImportError:
    print("❌  Install first:  pip install torch transformers accelerate")
    sys.exit(1)


SPLITS = {
    "primary":   ("phishguard_dev.csv", "phishguard_test.csv"),
    "sanitized": ("san_dev.csv",        "san_test.csv"),
    "ceas":      ("ceas_dev.csv",       "ceas_test.csv"),
}


class EmailDS(Dataset):
    def __init__(self, texts, labels, tok, max_len=384):
        self.enc = tok(texts, truncation=True, padding="max_length",
                       max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def to_text(r) -> str:
    """Identical view to the TF-IDF baseline and the cascade. No extra fields."""
    return f"{r.sender}\n{r.subject}\n{r.body}"


def load(path: Path):
    recs = load_emails(str(path))
    return [to_text(r) for r in recs], [1 if r.label == "phishing" else 0 for r in recs]


def metrics_from(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den  = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return {"accuracy": (tp+tn)/len(y_true), "precision": prec, "recall": rec,
            "f1": f1, "fpr": fp/(fp+tn) if (fp+tn) else 0.0,
            "mcc": ((tp*tn - fp*fn)/den) if den else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="primary")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--cross-corpus", action="store_true")
    args = ap.parse_args()

    dev_name, test_name = SPLITS[args.split]
    dev_p, test_p = DATA_DIR / dev_name, DATA_DIR / test_name
    if not dev_p.exists():
        print(f"❌  {dev_p} not found.")
        sys.exit(1)

    Xd, yd = load(dev_p)
    Xt, yt = load(test_p)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                               num_labels=2)

    train_ds = EmailDS(Xd, yd, tok, args.max_len)
    test_ds  = EmailDS(Xt, yt, tok, args.max_len)

    outdir = RESULTS_DIR / f"deberta_{args.split}"
    targs = TrainingArguments(
        output_dir=str(outdir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=50, save_strategy="no", report_to=[], seed=42,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)

    print("=" * 84)
    print(f"  DeBERTa fine-tune  ·  split={args.split}  model={args.model}")
    print(f"  train {len(yd)} / test {len(yt)}")
    print("=" * 84)

    t0 = time.perf_counter()
    trainer.train()
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    inf_s = time.perf_counter() - t1

    m = metrics_from(yt, list(pred))
    print()
    print(f"  IN-CORPUS   F1 {100*m['f1']:.2f}   Prec {100*m['precision']:.2f}   "
          f"Rec {100*m['recall']:.2f}   MCC {m['mcc']:.3f}")
    print(f"  fit {fit_s:.0f}s   inference {1000*inf_s/len(yt):.1f} s/1k emails")

    out = {"split": args.split, "model": args.model, "in_corpus": m,
           "fit_seconds": fit_s,
           "inference_seconds_per_1k": 1000 * inf_s / len(yt),
           "cross_corpus": {}}

    if args.cross_corpus:
        ceas = DATA_DIR / ("san_ceas_transfer.csv" if args.split == "sanitized"
                           else "ceas_transfer.csv")
        if ceas.exists():
            Xc, yc = load(ceas)
            cpred = np.argmax(trainer.predict(EmailDS(Xc, yc, tok, args.max_len))
                              .predictions, axis=-1)
            cm = metrics_from(yc, list(cpred))
            out["cross_corpus"]["CEAS_08"] = cm
            print()
            print(f"  CROSS-CORPUS (CEAS_08)   F1 {100*cm['f1']:.2f}   "
                  f"Prec {100*cm['precision']:.2f}   Rec {100*cm['recall']:.2f}   "
                  f"MCC {cm['mcc']:.3f}")
            print()
            print("  Report these two lines together. The gap between them is")
            print("  the result; the first line alone is not.")
        else:
            print(f"\nℹ️   {ceas.name} missing — run make_transfer_set.py")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"deberta_{args.split}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n✅  {p.name}")
    print("=" * 84)


if __name__ == "__main__":
    main()
