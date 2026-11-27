# A Phishing Detector That Never Saw Phishing

Code, results and paper source for *A Phishing Detector That Never Saw
Phishing: Provenance Confounding in a Public Phishing E-mail Benchmark*
(Vamsi Vunnam, University of Arkansas at Little Rock; under review at DMLR).

## The finding

A widely used phishing benchmark pairs Nazario phishing with SpamAssassin ham,
so every phishing message comes from one collection and every legitimate
message from another. Corpus identity then determines the label, and a model
that only recognises the collection scores as well as one that detects
phishing.

| Check | F1 (%) |
|---|---|
| Detector trained only to tell two *legitimate* corpora apart (never sees a phishing label) | 98.42 ± 0.25 |
| ... with its training pool made disjoint from the benchmark (exact and near-duplicates removed) | 97.78 ± 0.37 |
| Same test with a pretrained transformer (DeBERTa-v3-small) | 97.84 ± 0.51 |
| Parameter-free rule: "phishing if the Date year is 2003 or later" | 99.95 |
| Pooled model from three other corpora, then + 50 benchmark e-mails per class | 73.48 → 93.37 |

On the 82,486-e-mail six-corpus Kaggle compilation, provenance does not explain
the overall score, but it governs the Nazario phishing slice inside it.

Intervals are 95% Student-t half-widths over five seeds. The paper's Threats
to Validity section states what the evidence does and does not cover.

## Reproducing

```bash
pip install -r requirements.txt
```

The corpora are public but are **not redistributed here**.
`dataset_builder/SETUP.md` explains how to obtain them and rebuild the exact
splits. Every script writes JSON to `results/`; the committed JSON files are
the outputs reported in the paper, so you can compare against them.

| Script | Paper section |
|---|---|
| `experiments/provenance_transfer.py` | Provenance transfer |
| `experiments/provenance_overlap_audit.py` | Overlap control (Table 2) |
| `experiments/generalize.py` | Composed benchmarks |
| `experiments/model_families.py` | Robustness to model family |
| `colab_transformer/transformer_checks.py` | Pretrained transformer (GPU; see `RUN_ON_COLAB.txt`) |
| `experiments/pooled_indistribution_sweep.py` | Source-exposure sweep |
| `experiments/source_vs_task_decomposition.py` | Decomposition of the sweep |
| `experiments/sweep_near_duplicate_check.py` | Near-duplicate check on the sweep |
| `experiments/label_invariant_control.py` | Label-invariant separability |
| `experiments/independence.py` | Model-free rule; corpus robustness |
| `experiments/leakage_probe.py` | Channel ablation |
| `experiments/sanitize.py` | Attempts at repair |
| `experiments/why_ling_fails.py` | Robustness of the corpus choices |
| `experiments/compilation_audit.py`, `compilation_followups.py` | Six-corpus compilation |
| `experiments/make_figure.py`, `make_source_exposure_outputs.py` | Figures |

The TF–IDF experiments run on a laptop CPU in minutes. Seeds are fixed;
results were reproduced on a second machine.

## The diagnostic tool

`tools/provenance_report.py` runs the paper's diagnostics on any labelled
e-mail collection supplied as one CSV per source: datasheet, label–source
link, provenance transfer, class-held-fixed separability, year rule,
leave-one-source-out and an exposure sweep. See
`tools/README_provenance_report.md`.

## Paper

LaTeX source in `paper/` (DMLR style). Build with
`pdflatex main_dmlr && bibtex main_dmlr && pdflatex main_dmlr && pdflatex main_dmlr`.

## Also in this repository

`phishcascade/` is an earlier cost-aware triage prototype. It is not part of
the paper's claims; see `docs/PHISHCASCADE_PROTOTYPE.md`.

## Licence

Code: MIT (see `LICENSE`). The e-mail corpora keep their own terms, which
the paper's Data Availability section records.
