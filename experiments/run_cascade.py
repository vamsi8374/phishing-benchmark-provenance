"""
Run the cascade over a split and record everything.

    python experiments/run_cascade.py --split test
    python experiments/run_cascade.py --split dev --policy always_deliberate
    python experiments/run_cascade.py --split dev --backend mock --limit 50

Writes results/<split>_<policy>.jsonl (one CascadeResult per line) plus a
summary with accuracy and the compute budget. Runs are resumable: an
interrupted run picks up where it stopped, because a 3,000-email pass over a
14B model on a laptop is measured in hours, not minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishcascade.config import (          # noqa: E402
    DEV_CSV, TEST_CSV, RESULTS_DIR, PHISHGUARD_DIR,
    RunConfig, RoutingPolicy, SMALL_MODEL, LARGE_MODEL,
)
from phishcascade.graph import run_cascade  # noqa: E402
from phishcascade.llm import ollama_available, installed_models  # noqa: E402

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails  # noqa: E402

POLICIES = {
    "cascade":           RoutingPolicy(),
    "always_deliberate": RoutingPolicy.always_deliberate(),
    "heuristics_only":   RoutingPolicy.heuristics_only(),
}


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den  = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp+tn)/len(y_true) if y_true else 0.0,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp/(fp+tn) if (fp+tn) else 0.0,
            "mcc": ((tp*tn - fp*fn)/den) if den else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--policy", choices=list(POLICIES), default="cascade")
    ap.add_argument("--backend", choices=["ollama", "mock"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--small", default=None)
    ap.add_argument("--large", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint")
    args = ap.parse_args()

    cfg = RunConfig(policy=POLICIES[args.policy])
    if args.backend:
        cfg.backend = args.backend
    if args.small:
        cfg.small = SMALL_MODEL.__class__(name=args.small, tier="small")
    if args.large:
        cfg.large = LARGE_MODEL.__class__(name=args.large, tier="large")
    if args.no_cache:
        cfg.cache = False

    if cfg.backend == "ollama":
        if not ollama_available():
            print("❌  Ollama is not reachable. Start it with:  ollama serve")
            print("    Or run with --backend mock to exercise the plumbing.")
            sys.exit(1)
        have = installed_models()
        for m in {cfg.small.name, cfg.large.name}:
            if not any(h.split(":")[0] == m.split(":")[0] for h in have):
                print(f"❌  Model not installed: {m}")
                print(f"    ollama pull {m}")
                sys.exit(1)
    else:
        print("⚠️   MOCK BACKEND — verdicts are hash-derived nonsense.")
        print("    Use this only to verify plumbing. Never report these numbers.\n")

    csv_path = DEV_CSV if args.split == "dev" else TEST_CSV
    if not csv_path.exists():
        print(f"❌  {csv_path} not found. Run prepare_dataset.py first.")
        sys.exit(1)

    records = load_emails(str(csv_path))
    if args.limit:
        records = records[:args.limit]
    y_true = [1 if r.label == "phishing" else 0 for r in records]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{args.policy}_{cfg.backend}"
    out_jsonl = RESULTS_DIR / f"{tag}.jsonl"

    done: dict[int, dict] = {}
    if out_jsonl.exists() and not args.fresh:
        for line in out_jsonl.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["email_id"]] = d
            except Exception:
                pass
        if done:
            print(f"↻  Resuming: {len(done)} of {len(records)} already done.")

    print("=" * 78)
    print(f"  PhishCascade  ·  split={args.split}  policy={args.policy}  "
          f"backend={cfg.backend}")
    print(f"  small={cfg.small.name}   large={cfg.large.name}")
    print(f"  policy: {cfg.policy.as_dict()}")
    print(f"  {len(records)} emails ({sum(y_true)} phishing / "
          f"{len(y_true)-sum(y_true)} legitimate)")
    print("=" * 78)

    results, t_start = [], time.perf_counter()
    fh = out_jsonl.open("a" if done and not args.fresh else "w", encoding="utf-8")

    for i, rec in enumerate(records):
        if i in done:
            results.append(done[i])
            continue
        try:
            r = run_cascade(rec, cfg, email_id=i, label=y_true[i])
            d = r.to_json()
        except Exception as e:
            d = {"email_id": i, "label": y_true[i], "verdict": 0,
                 "confidence": 0.0, "exit_stage": "error", "error": repr(e),
                 "n_llm_calls": 0, "total_tokens": 0, "compute_s": 0.0,
                 "large_model_calls": 0, "heuristic_score": 0,
                 "heuristic_signals": [], "iocs": {}, "stages": [],
                 "opinions": []}
        results.append(d)
        fh.write(json.dumps(d) + "\n")
        fh.flush()

        if (i + 1) % 25 == 0 or i + 1 == len(records):
            el = time.perf_counter() - t_start
            newly = i + 1 - len(done)
            rate = newly / el if el > 0 and newly > 0 else 0
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>5}/{len(records)}]  {rate:5.2f} em/s   "
                  f"ETA {eta/60:6.1f} min")

    fh.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    ok = [d for d in results if d.get("exit_stage") != "error"]
    y_pred = [d["verdict"] for d in ok]
    y_gold = [d["label"] for d in ok]
    m = metrics(y_gold, y_pred)

    n = len(ok) or 1
    exits = {}
    for d in ok:
        exits[d["exit_stage"]] = exits.get(d["exit_stage"], 0) + 1

    tot_calls = sum(d["n_llm_calls"] for d in ok)
    tot_large = sum(d["large_model_calls"] for d in ok)
    tot_tok   = sum(d["total_tokens"] for d in ok)
    tot_gen   = sum(d.get("generated_tokens", 0) for d in ok)
    tot_s     = sum(d["compute_s"] for d in ok)

    print()
    print("── Accuracy ──")
    print(f"  Acc {100*m['accuracy']:.2f}   Prec {100*m['precision']:.2f}   "
          f"Rec {100*m['recall']:.2f}   F1 {100*m['f1']:.2f}   "
          f"FPR {100*m['fpr']:.2f}   MCC {m['mcc']:.3f}")
    print()
    print("── Routing ──")
    for k in ("stage0", "stage1", "stage2"):
        c = exits.get(k, 0)
        print(f"  exited at {k}: {c:>6}  ({100*c/n:5.1f}%)")
    if any(d.get("exit_stage") == "error" for d in results):
        print(f"  errors:        {len(results)-len(ok):>6}")
    print()
    print("── Compute budget (per 1,000 emails) ──")
    print(f"  LLM calls        {1000*tot_calls/n:10.1f}")
    print(f"  large-model calls{1000*tot_large/n:10.1f}")
    print(f"  tokens total     {1000*tot_tok/n:10.0f}")
    print(f"  tokens generated {1000*tot_gen/n:10.0f}")
    print(f"  inference seconds{1000*tot_s/n:10.1f}")

    summary = {"split": args.split, "policy": args.policy,
               "policy_params": cfg.policy.as_dict(),
               "backend": cfg.backend,
               "models": {"small": cfg.small.name, "large": cfg.large.name},
               "n": n, "metrics": m, "exit_distribution": exits,
               "per_1k": {"llm_calls": 1000*tot_calls/n,
                          "large_model_calls": 1000*tot_large/n,
                          "tokens_total": 1000*tot_tok/n,
                          "tokens_generated": 1000*tot_gen/n,
                          "inference_seconds": 1000*tot_s/n}}
    (RESULTS_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"✅  {out_jsonl.name}")
    print(f"✅  {tag}_summary.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
