"""
Transformer version of the two key checks (reviewer Concern 3).

    pip install -q transformers accelerate sentencepiece
    python transformer_checks.py                 # all three parts, 5 seeds
    python transformer_checks.py --part supervised
    python transformer_checks.py --part provenance
    python transformer_checks.py --part null

WHAT IT RUNS (all with microsoft/deberta-v3-small, 5 seeds)

  supervised  Train on the benchmark dev split (930 msgs), test on the
              benchmark test split (2170) and on CEAS_08 (4000).
              Same text view as the TF-IDF baseline: sender, subject, body.
              Question: does a transformer also look perfect in-corpus and
              drop across corpora?

  provenance  Train on SpamAssassin ham vs CEAS ham (1500 per side, both
              legitimate, body only). No phishing label anywhere. Apply to
              the benchmark test set with the fixed rule
                  looks like SpamAssassin -> benign, otherwise -> phishing.
              Same sampling code and seeds as provenance_transfer.py, so the
              training sets are identical to the TF-IDF run.
              Question: can a transformer that never saw phishing also
              solve the benchmark?

  null        Same as provenance with the corpus labels shuffled.
              Question: does the score come from the learned provenance
              signal and not from the mapping rule?

RESUMABLE: results are written to results/transformer_checks.json after
every single run. If Colab disconnects, run the same command again and it
skips the runs that are already finished.

DATA (folder ./data next to this script)
  phishguard_dev.csv, phishguard_test.csv, ceas_transfer.csv
  sa_ham.csv, ceas_ham.csv   (ham bodies, pre-filtered exactly as
                              provenance_transfer.py filters them:
                              label 0, control chars removed, length >= 30,
                              original file order kept)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "transformer_checks.json"

SEEDS = [0, 1, 2, 3, 4]
N_PER_SIDE = 1500                      # same as provenance_transfer.py
# Two-sided 95% Student-t critical values by degrees of freedom (n - 1).
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BENIGN, PHISH = 0, 1
SPAMASSASSIN, NOT_SPAMASSASSIN = 0, 1

csv.field_size_limit(10_000_000)


# ── data ──────────────────────────────────────────────────────────────────────
def clean(s: str) -> str:
    return CTRL.sub(" ", s or "")


def read_rows(name: str) -> list[dict]:
    with (DATA / name).open("r", encoding="utf-8", errors="replace",
                            newline="") as fh:
        return list(csv.DictReader(fh))


def label_of(r: dict) -> int:
    v = (r.get("label") or "").strip()
    if v not in ("0", "1"):
        raise ValueError(f"unexpected label {v!r}")
    return PHISH if v == "1" else BENIGN


def full_view(r: dict) -> str:
    """Identical to baseline_deberta.py: sender, subject, body."""
    return f"{r.get('sender', '')}\n{r.get('subject', '')}\n{r.get('body', '')}"


def read_bodies(name: str) -> list[str]:
    return [r["body"] for r in read_rows(name)]


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": prec,
            "recall": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summarise(runs: list[dict], key: str) -> dict:
    v = [r[key] for r in runs]
    n = len(v)
    mean = float(np.mean(v))
    half = (T_CRIT[n - 1] * float(np.std(v, ddof=1)) / n ** 0.5) if n > 1 else 0.0
    return {"mean_pct": round(100 * mean, 2), "ci95_pct": round(100 * half, 2),
            "n_seeds": n, "per_seed": [round(x, 6) for x in v]}


# ── training ──────────────────────────────────────────────────────────────────
def train_and_predict(args, seed, train_X, train_y, eval_sets):
    """Fine-tune once; return argmax predictions for each eval set."""
    set_seed(seed)                               # also seeds the new head
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).float()   # fp32 master weights; AMP fp16 needs them
    # 10% linear warm-up, computed by hand: `warmup_ratio` was removed in
    # transformers 5.x, and an integer step count works in 4.x and 5.x.
    steps = int(np.ceil(len(train_X) / args.bs) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(HERE / "_tmp"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr, weight_decay=0.01, warmup_steps=int(0.1 * steps),
        logging_steps=50, save_strategy="no", report_to="none",
        seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available() and not args.no_fp16,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=TextDS(train_X, train_y, tok, args.max_len),
                      data_collator=DataCollatorWithPadding(tok))
    t0 = time.perf_counter()
    trainer.train()
    fit_s = time.perf_counter() - t0

    preds = {}
    for name, X in eval_sets.items():
        ds = TextDS(X, [0] * len(X), tok, args.max_len)
        logits = trainer.predict(ds).predictions
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite logits (fp16 overflow?) — "
                               "re-run with --no-fp16")
        preds[name] = np.argmax(logits, axis=-1).tolist()
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, fit_s


# ── the three parts ───────────────────────────────────────────────────────────
def run_supervised(args, seed, cache):
    dev, test, ceas = cache["dev"], cache["test"], cache["ceas"]
    preds, fit_s = train_and_predict(
        args, seed, [full_view(r) for r in dev], [label_of(r) for r in dev],
        {"in_corpus": [full_view(r) for r in test],
         "CEAS_08": [full_view(r) for r in ceas]})
    return {"seed": seed, "fit_seconds": round(fit_s, 1),
            "in_corpus": metrics([label_of(r) for r in test], preds["in_corpus"]),
            "CEAS_08": metrics([label_of(r) for r in ceas], preds["CEAS_08"])}


def run_provenance(args, seed, cache, shuffle_labels):
    # Exactly the sampling in provenance_transfer.run_once.
    sa_ham, ceas_ham = cache["sa_ham"], cache["ceas_ham"]
    rng = random.Random(seed)
    n = min(len(sa_ham), len(ceas_ham), N_PER_SIDE)
    X = rng.sample(sa_ham, n) + rng.sample(ceas_ham, n)
    y = [SPAMASSASSIN] * n + [NOT_SPAMASSASSIN] * n
    if shuffle_labels:
        rng.shuffle(y)

    test = cache["test"]
    bench_X = [clean(r.get("body", "")) for r in test]
    bench_y = [label_of(r) for r in test]
    preds, fit_s = train_and_predict(args, seed, X, y, {"bench": bench_X})
    mapped = [BENIGN if p == SPAMASSASSIN else PHISH for p in preds["bench"]]
    return {"seed": seed, "fit_seconds": round(fit_s, 1),
            "n_per_side": n, "benchmark": metrics(bench_y, mapped),
            "share_predicted_phishing": round(sum(mapped) / len(mapped), 4)}


# ── driver ────────────────────────────────────────────────────────────────────
def load_state():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"per_seed": {}, "summary": {}}


def save_state(state, args):
    state["config"] = {"model": args.model, "epochs": args.epochs,
                       "bs": args.bs, "lr": args.lr, "max_len": args.max_len,
                       "seeds": args.seeds, "n_per_side": N_PER_SIDE}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "supervised", "provenance", "null"],
                    default="all")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsets, for checking the code runs; "
                         "results go to a separate file")
    ap.add_argument("--disjoint", action="store_true",
                    help="overlap control: remove from the SpamAssassin-side "
                         "training pool every message whose body also appears "
                         "in the benchmark dev or test split; runs the "
                         "provenance and null parts only; results go to "
                         "transformer_checks_disjoint.json")
    args = ap.parse_args()

    global OUT
    if args.smoke:
        OUT = OUT.with_name("transformer_checks_SMOKE.json")
    if args.disjoint:
        OUT = OUT.with_name("transformer_checks_disjoint.json"
                            if not args.smoke else
                            "transformer_checks_disjoint_SMOKE.json")
        if args.part in ("all", "supervised"):
            args.part = "provenance+null"

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "none (this will be very slow)")

    cache = {"dev": read_rows("phishguard_dev.csv"),
             "test": read_rows("phishguard_test.csv"),
             "ceas": read_rows("ceas_transfer.csv"),
             "sa_ham": read_bodies("sa_ham.csv"),
             "ceas_ham": read_bodies("ceas_ham.csv")}
    if args.disjoint:
        seen = {clean(r.get("body", "")).strip()
                for r in cache["dev"] + cache["test"]}
        before = len(cache["sa_ham"])
        cache["sa_ham"] = [b for b in cache["sa_ham"]
                           if clean(b).strip() not in seen]
        print(f"disjoint: SpamAssassin training pool {before} -> "
              f"{len(cache['sa_ham'])} (expected about 4091 -> 2605)")
    print({k: len(v) for k, v in cache.items()})
    if args.smoke:
        rs = random.Random(0)
        for k in ("dev", "test", "ceas"):
            cache[k] = rs.sample(cache[k], 40)
        global N_PER_SIDE
        N_PER_SIDE = 20

    if args.part == "all":
        parts = ["supervised", "provenance", "null"]
    elif args.part == "provenance+null":
        parts = ["provenance", "null"]
    else:
        parts = [args.part]
    state = load_state()

    for part in parts:
        done = state["per_seed"].setdefault(part, [])
        for seed in args.seeds:
            if any(r["seed"] == seed for r in done):
                print(f"[skip] {part} seed {seed} already done")
                continue
            print(f"\n=== {part}  seed {seed} ===")
            if part == "supervised":
                res = run_supervised(args, seed, cache)
            else:
                res = run_provenance(args, seed, cache, part == "null")
            done.append(res)
            save_state(state, args)                    # after every run
            key = "in_corpus" if part == "supervised" else "benchmark"
            print(f"    F1 {100 * res[key]['f1']:.2f}")

    # summary tables
    s = {}
    sup = state["per_seed"].get("supervised", [])
    if sup:
        s["supervised_in_corpus_f1"] = summarise([r["in_corpus"] for r in sup], "f1")
        s["supervised_CEAS_08_f1"] = summarise([r["CEAS_08"] for r in sup], "f1")
    for part in ("provenance", "null"):
        runs = state["per_seed"].get(part, [])
        if runs:
            s[f"{part}_benchmark_f1"] = summarise([r["benchmark"] for r in runs], "f1")
            s[f"{part}_benchmark_acc"] = summarise([r["benchmark"] for r in runs],
                                                   "accuracy")
    state["summary"] = s
    save_state(state, args)

    print("\n" + "=" * 70)
    for k, v in s.items():
        print(f"  {k:<32} {v['mean_pct']:6.2f} ± {v['ci95_pct']:.2f}  "
              f"(n={v['n_seeds']})")
    print("=" * 70)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
