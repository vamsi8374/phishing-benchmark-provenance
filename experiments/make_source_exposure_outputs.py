"""
Build the table and figure for the rewritten pooling section (major gap 1).

    python experiments/pooled_indistribution_sweep.py        # if not already run
    python experiments/source_vs_task_decomposition.py
    python experiments/make_source_exposure_outputs.py

Reads   results/pooled_indistribution_sweep.json
        results/source_vs_task_decomposition.json
Writes  paper/tab_source_exposure.tex
        paper/fig_source_exposure.pdf  (+ .png preview)

Numbers are read from the JSON files only; nothing is typed by hand. The table
caption is left as a placeholder for the author to write.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    sys.path.insert(0, str(HERE.parent))
    from phishcascade.config import RESULTS_DIR          # noqa: E402
    PAPER_DIR = HERE.parent / "paper"
except Exception:
    RESULTS_DIR = HERE / "results"
    PAPER_DIR = HERE / "paper"
PAPER_DIR = Path(os.environ.get("PAPER_DIR", PAPER_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sweep = json.loads((RESULTS_DIR / "pooled_indistribution_sweep.json").read_text())["summary"]
dec = json.loads((RESULTS_DIR / "source_vs_task_decomposition.json").read_text())
ks = dec["k_sweep"]
S = dec["summary"]


def cell(d, metric="f1"):
    m, h = 100 * d[metric]["mean"], 100 * d[metric]["ci95"]
    return f"{m:.2f}\\,$\\pm${h:.2f}"


def arm(a, k):
    return S["A_k0"] if k == 0 else S[f"{a}_k{k}"]


prov = sweep["provenance_only"]

# ── table ─────────────────────────────────────────────────────────────────────
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{% TODO (author): write the caption in your own words.",
    r"% It should say: what is added to the pool in each column, that the test",
    r"% split is the unchanged benchmark test set (n = 2,170), that values are",
    r"% mean F1 (%) with 95% Student-t half-widths over five seeds, and why the",
    r"% dev-only k = 465 cell has no interval (the whole dev partition is used,",
    r"% so the seed does not change the training set).",
    r"}",
    r"\label{tab:source_exposure}",
    r"\begin{tabular}{rccccc}",
    r"\toprule",
    r" & \multicolumn{4}{c}{Pool (CEAS + Ling + Enron) plus $k$ per class of \ldots} & \\",
    r"\cmidrule(lr){2-5}",
    r"$k$ & both dev classes & dev ham only & dev phishing only & Nigerian fraud & dev only \\",
    r"\midrule",
]
for k in ks:
    dev_only = sweep.get(f"dev_only_k{k}")
    lines.append(
        f"{k} & {cell(arm('A', k))} & {cell(arm('H', k)) if k else '---'} & "
        f"{cell(arm('P', k)) if k else '---'} & {cell(arm('N', k)) if k else '---'} & "
        f"{cell(dev_only) if dev_only else '---'} \\\\")
lines += [
    r"\midrule",
    f"\\multicolumn{{6}}{{l}}{{Provenance-only detector (no phishing labels): "
    f"{cell(prov)}}} \\\\",
    r"\midrule",
    r"\multicolumn{6}{l}{\emph{Error split at $k=465$ (recall / false-positive rate, \%):}} \\",
]
for a, name in (("A", "both dev classes"), ("H", "dev ham only"),
                ("P", "dev phishing only"), ("N", "Nigerian fraud")):
    d = S[f"{a}_k465"]
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\quad {name}: "
                 f"{100*d['recall']['mean']:.2f} / {100*d['fpr']['mean']:.2f}}} \\\\")
d0 = S["A_k0"]
lines.append(f"\\multicolumn{{6}}{{l}}{{\\quad pool only ($k=0$): "
             f"{100*d0['recall']['mean']:.2f} / {100*d0['fpr']['mean']:.2f}}} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

PAPER_DIR.mkdir(parents=True, exist_ok=True)
(PAPER_DIR / "tab_source_exposure.tex").write_text("\n".join(lines))

# ── figure ────────────────────────────────────────────────────────────────────
series = [("A", "pool + both dev classes", "#1f4e79", "o"),
          ("H", "pool + dev ham only", "#2e8b57", "s"),
          ("P", "pool + dev phishing only", "#c0504d", "^"),
          ("N", "pool + Nigerian fraud", "#7f7f7f", "D")]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
ax = axes[0]
for a, lab, col, mk in series:
    ys = [100 * arm(a, k)["f1"]["mean"] for k in ks]
    es = [100 * arm(a, k)["f1"]["ci95"] for k in ks]
    ax.errorbar(ks, ys, yerr=es, label=lab, color=col, marker=mk, ms=5,
                capsize=3, lw=1.6)
dks = [k for k in ks if f"dev_only_k{k}" in sweep]
ax.plot(dks, [100 * sweep[f"dev_only_k{k}"]["f1"]["mean"] for k in dks],
        color="black", ls=":", marker="x", label="dev only (no pool)")
ax.axhline(100 * prov["f1"]["mean"], color="#d4a017", ls="--", lw=1.4,
           label="provenance-only (no phishing labels)")
ax.set_xlabel("examples added per class ($k$)")
ax.set_ylabel("benchmark test $F_1$ (%)")
ax.set_title("(a) $F_1$ on the benchmark")
ax.set_ylim(60, 101)
ax.grid(alpha=0.3)
handles, labels = ax.get_legend_handles_labels()

ax = axes[1]
for a, lab, col, mk in series:
    rec = [100 * arm(a, k)["recall"]["mean"] for k in ks]
    fpr = [100 * arm(a, k)["fpr"]["mean"] for k in ks]
    ax.plot(ks, rec, color=col, marker=mk, ms=5, lw=1.6)
    ax.plot(ks, fpr, color=col, marker=mk, ms=5, lw=1.6, ls="--", mfc="white")
ax.set_xlabel("examples added per class ($k$)")
ax.set_ylabel("%")
ax.set_title("(b) recall (solid) and false-positive rate (dashed)")
ax.grid(alpha=0.3)
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(PAPER_DIR / "fig_source_exposure.pdf")
fig.savefig(PAPER_DIR / "fig_source_exposure.png", dpi=160)
print("wrote", PAPER_DIR / "tab_source_exposure.tex")
print("wrote", PAPER_DIR / "fig_source_exposure.pdf")
print("wrote", PAPER_DIR / "fig_source_exposure.png")
