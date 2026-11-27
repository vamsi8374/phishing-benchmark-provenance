"""
The main experiment: accuracy vs compute budget, swept over routing policies.

    python experiments/pareto.py --split dev

METHOD — and why it costs one pass, not one pass per policy
------------------------------------------------------------
Naively, sweeping N routing policies means N full runs. That is unaffordable
on local hardware. Instead we exploit the cascade's structure: for a fixed set
of models, every stage's *output* for a given email is independent of the
routing thresholds. The thresholds only decide which outputs get consulted.

So we run every email through all three stages once, in "oracle" mode, cache
the per-stage verdicts and their measured costs, then replay any policy over
that cache for free. This is exact, not an approximation — the replayed
verdict is identical to what a live run with that policy would produce,
because generation is deterministic (temperature 0, fixed seed).

The one thing replay must get right is that a live run SKIPS later stages.
That is exactly what the cost model does: it charges only the stages the
policy actually visits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishcascade.config import (        # noqa: E402
    DEV_CSV, TEST_CSV, RESULTS_DIR, PHISHGUARD_DIR, RunConfig, RoutingPolicy,
)
from phishcascade.stages import stage0, stage1, stage2   # noqa: E402
from phishcascade.llm import probability_of_phish, ollama_available  # noqa: E402
from phishcascade.schema import BENIGN, PHISH            # noqa: E402

sys.path.insert(0, str(PHISHGUARD_DIR))
from phishguard.dataset import load_emails               # noqa: E402


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


def build_oracle(records, y, cfg: RunConfig, cache_path: Path) -> list[dict]:
    """Run every email through all three stages once; cache verdicts + costs."""
    rows: list[dict] = []
    if cache_path.exists():
        rows = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
        if len(rows) >= len(records):
            print(f"↻  Using cached oracle pass ({len(rows)} emails)")
            return rows[:len(records)]
        print(f"↻  Resuming oracle pass at {len(rows)}/{len(records)}")

    forced = RunConfig(backend=cfg.backend, small=cfg.small, large=cfg.large,
                       policy=RoutingPolicy.always_deliberate(),
                       cache=cfg.cache, max_body_chars=cfg.max_body_chars)
    open_all = RoutingPolicy(tau_low=-1, tau_high=10**9)

    fh = cache_path.open("a", encoding="utf-8")
    for i in range(len(rows), len(records)):
        rec = records[i]
        try:
            s0 = stage0(rec, open_all)
            note = s0.detail.get("note", "")
            s1 = stage1(rec, forced, note)
            s2, ops = stage2(rec, forced, note)
            row = {
                "email_id": i, "label": y[i],
                "s0_score": s0.detail.get("risk_score", 0),
                "s1_verdict": s1.verdict,
                "s1_p": probability_of_phish(s1.verdict, s1.confidence),
                "s1_calls": len(s1.calls),
                "s1_tokens": sum(c.total_tokens for c in s1.calls),
                "s1_seconds": sum(c.latency_s for c in s1.calls),
                "s2_verdict": s2.verdict,
                "s2_conf": s2.confidence,
                "s2_calls": len(s2.calls),
                "s2_tokens": sum(c.total_tokens for c in s2.calls),
                "s2_seconds": sum(c.latency_s for c in s2.calls),
                "votes": {o.role: o.verdict for o in ops},
            }
        except Exception as e:
            row = {"email_id": i, "label": y[i], "error": repr(e),
                   "s0_score": 0, "s1_verdict": None, "s1_p": 0.5,
                   "s1_calls": 0, "s1_tokens": 0, "s1_seconds": 0.0,
                   "s2_verdict": BENIGN, "s2_conf": 0.0,
                   "s2_calls": 0, "s2_tokens": 0, "s2_seconds": 0.0,
                   "votes": {}}
        rows.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        if (i + 1) % 25 == 0:
            print(f"  oracle pass [{i+1}/{len(records)}]")
    fh.close()
    return rows


def replay(rows: list[dict], pol: RoutingPolicy) -> dict:
    """Apply a routing policy to cached oracle outputs. Exact, not estimated."""
    y_true, y_pred = [], []
    exits = {"stage0": 0, "stage1": 0, "stage2": 0}
    calls = tokens = 0
    seconds = 0.0

    for r in rows:
        y_true.append(r["label"])
        score = r["s0_score"]

        if score >= pol.tau_high:
            y_pred.append(PHISH); exits["stage0"] += 1; continue
        if score < pol.tau_low:
            y_pred.append(BENIGN); exits["stage0"] += 1; continue

        calls   += r["s1_calls"]
        tokens  += r["s1_tokens"]
        seconds += r["s1_seconds"]

        margin = abs(r["s1_p"] - 0.5)
        if r["s1_verdict"] is not None and margin >= pol.delta / 2.0:
            y_pred.append(r["s1_verdict"]); exits["stage1"] += 1; continue

        calls   += r["s2_calls"]
        tokens  += r["s2_tokens"]
        seconds += r["s2_seconds"]
        y_pred.append(r["s2_verdict"] if r["s2_verdict"] is not None else BENIGN)
        exits["stage2"] += 1

    n = len(rows) or 1
    return {"policy": pol.as_dict(), "metrics": metrics(y_true, y_pred),
            "exits": exits,
            "per_1k": {"llm_calls": 1000*calls/n,
                       "tokens": 1000*tokens/n,
                       "inference_seconds": 1000*seconds/n}}


def pareto_front(points: list[dict]) -> list[dict]:
    """Keep points not dominated on (higher F1, lower seconds)."""
    front = []
    for p in points:
        f1, s = p["metrics"]["f1"], p["per_1k"]["inference_seconds"]
        dominated = any(
            q["metrics"]["f1"] >= f1
            and q["per_1k"]["inference_seconds"] <= s
            and (q["metrics"]["f1"] > f1
                 or q["per_1k"]["inference_seconds"] < s)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["per_1k"]["inference_seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = RunConfig(backend=args.backend)
    if cfg.backend == "ollama" and not ollama_available():
        print("❌  Ollama unreachable. Start with `ollama serve`, "
              "or use --backend mock to test the plumbing.")
        sys.exit(1)

    csv_path = DEV_CSV if args.split == "dev" else TEST_CSV
    records = load_emails(str(csv_path))
    if args.limit:
        records = records[:args.limit]
    y = [1 if r.label == "phishing" else 0 for r in records]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oracle_path = RESULTS_DIR / f"oracle_{args.split}_{cfg.backend}.jsonl"

    print("=" * 80)
    print(f"  ORACLE PASS  ·  {len(records)} emails  ·  backend={cfg.backend}")
    print("  Every email visits every stage once. Policies replay from this")
    print("  cache, so the sweep below costs no extra inference.")
    print("=" * 80)
    rows = build_oracle(records, y, cfg, oracle_path)

    points = []
    for tau_low in (0, 1, 6, 11):
        for tau_high in (26, 31, 41, 51, 10**9):
            for delta in (0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.01):
                points.append(replay(rows, RoutingPolicy(tau_low, tau_high, delta)))

    baselines = {
        "heuristics_only":   replay(rows, RoutingPolicy.heuristics_only()),
        "stage1_only":       replay(rows, RoutingPolicy(tau_low=-1,
                                                        tau_high=10**9, delta=0.0)),
        "always_deliberate": replay(rows, RoutingPolicy.always_deliberate()),
    }

    front = pareto_front(points)
    n = len(rows)

    print()
    print("── Baselines ──")
    h = f"{'policy':<20}{'F1%':>8}{'Prec%':>8}{'Rec%':>8}{'calls/1k':>11}{'sec/1k':>10}"
    print(h); print("-" * len(h))
    for name, b in baselines.items():
        print(f"{name:<20}{100*b['metrics']['f1']:>8.2f}"
              f"{100*b['metrics']['precision']:>8.2f}"
              f"{100*b['metrics']['recall']:>8.2f}"
              f"{b['per_1k']['llm_calls']:>11.0f}"
              f"{b['per_1k']['inference_seconds']:>10.1f}")

    print()
    print("── Pareto frontier (accuracy vs compute) ──")
    h2 = (f"{'tau_lo':>7}{'tau_hi':>8}{'delta':>7}{'F1%':>8}{'Prec%':>8}"
          f"{'Rec%':>8}{'s0%':>7}{'s1%':>7}{'s2%':>7}{'sec/1k':>10}")
    print(h2); print("-" * len(h2))
    for p in front:
        pol, m, e = p["policy"], p["metrics"], p["exits"]
        th = pol["tau_high"] if pol["tau_high"] < 10**8 else "inf"
        print(f"{pol['tau_low']:>7}{th:>8}"
              f"{pol['delta']:>7.2f}{100*m['f1']:>8.2f}{100*m['precision']:>8.2f}"
              f"{100*m['recall']:>8.2f}{100*e['stage0']/n:>7.1f}"
              f"{100*e['stage1']/n:>7.1f}{100*e['stage2']/n:>7.1f}"
              f"{p['per_1k']['inference_seconds']:>10.1f}")

    ad = baselines["always_deliberate"]
    target = ad["metrics"]["f1"] - 0.01
    ok = [p for p in points if p["metrics"]["f1"] >= target]
    if ok:
        best = min(ok, key=lambda p: p["per_1k"]["inference_seconds"])
        sp = ad["per_1k"]["inference_seconds"]
        bp = best["per_1k"]["inference_seconds"]
        print()
        print("── Headline result ──")
        print(f"  Always-deliberate:   F1 {100*ad['metrics']['f1']:.2f}%   "
              f"{sp:.1f} s / 1k emails")
        print(f"  Cheapest within 1pt: F1 {100*best['metrics']['f1']:.2f}%   "
              f"{bp:.1f} s / 1k emails")
        print(f"  policy: {best['policy']}")
        if bp > 0:
            print(f"  → {sp/bp:.2f}x less inference for "
                  f"{100*(best['metrics']['f1']-ad['metrics']['f1']):+.2f} F1")

    out = {"split": args.split, "backend": cfg.backend, "n": n,
           "baselines": baselines, "pareto_front": front, "all_points": points}
    (RESULTS_DIR / f"pareto_{args.split}_{cfg.backend}.json").write_text(
        json.dumps(out, indent=2))
    print()
    print(f"✅  pareto_{args.split}_{cfg.backend}.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
