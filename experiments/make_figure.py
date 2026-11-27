"""
Figure 1 for the paper.

    python experiments/make_figure.py

Two panels, both from measured JSON in results/:

  (a) Channel ablation on the confounded pairing versus the single-source
      control. The gap on `date_only` is the whole argument in one picture.
  (b) Provenance transfer against its null, with the supervised reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import RESULTS_DIR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "fig1.pdf"

# Measured values (see results/*.json)
VIEWS = ["full", "body\nonly", "subject\nonly", "sender\nonly",
         "domain\nonly", "date\nonly", "nonword\nshape"]
CONFOUNDED = [99.82, 99.63, 94.87, 90.54, 83.54, 99.91, 96.86]
SINGLE_SRC = [98.50, 98.18, 93.52, 80.54, 79.55, 63.43, 97.39]

TRANSFER = {
    "Provenance\ndetector\n(no phishing label)": (98.42, 0.25),  # 95% Student-t half-widths (results/provenance_transfer.json)
    "Null\n(labels shuffled)": (57.34, 7.53),
}
SUPERVISED = 99.82


def main():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.75, 1]})

    # ── (a) channel ablation ────────────────────────────────────────────────
    x = np.arange(len(VIEWS))
    w = 0.38
    ax1.bar(x - w/2, CONFOUNDED, w, label="Nazario / SpamAssassin (confounded)",
            color="#b2182b", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, SINGLE_SRC, w, label="CEAS 2008 (single-source control)",
            color="#4393c3", edgecolor="black", linewidth=0.5)

    ax1.axhline(50, ls=":", lw=1, color="grey")
    ax1.text(len(VIEWS) - 0.35, 51.5, "chance", fontsize=7.5, color="grey",
             ha="right")

    # Highlight the date_only gap
    i = VIEWS.index("date\nonly")
    ax1.annotate("", xy=(i + w/2, SINGLE_SRC[i]), xytext=(i - w/2, CONFOUNDED[i]),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    ax1.text(i + 0.05, (CONFOUNDED[i] + SINGLE_SRC[i]) / 2,
             f"  {CONFOUNDED[i]-SINGLE_SRC[i]:.1f} pt", fontsize=8.5,
             va="center", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(VIEWS, fontsize=8)
    ax1.set_ylabel("test $F_1$ (%)", fontsize=9)
    ax1.set_ylim(40, 105)
    ax1.set_title("(a) Channel ablation: which views separate the classes?",
                  fontsize=9.5, loc="left")
    ax1.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    # ── (b) provenance transfer ─────────────────────────────────────────────
    labels = list(TRANSFER)
    means = [TRANSFER[k][0] for k in labels]
    errs = [TRANSFER[k][1] for k in labels]
    colors = ["#b2182b", "#bbbbbb"]
    xb = np.arange(len(labels))
    ax2.bar(xb, means, 0.55, yerr=errs, capsize=4, color=colors,
            edgecolor="black", linewidth=0.5)

    ax2.axhline(SUPERVISED, ls="--", lw=1.2, color="#333333")
    ax2.text(len(labels) - 0.5, SUPERVISED + 1.2,
             f"supervised reference  {SUPERVISED:.2f}",
             fontsize=7.5, ha="right", color="#333333")
    ax2.axhline(50, ls=":", lw=1, color="grey")

    for xi, (m, e) in enumerate(zip(means, errs)):
        ax2.text(xi, m + e + 2.0, f"{m:.2f}", ha="center", fontsize=9,
                 fontweight="bold")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("test $F_1$ (%)", fontsize=9)
    ax2.set_ylim(0, 112)
    ax2.set_title("(b) Provenance transfer", fontsize=9.5, loc="left")
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"✅  {OUT.name} and {OUT.with_suffix('.png').name}")


if __name__ == "__main__":
    main()
