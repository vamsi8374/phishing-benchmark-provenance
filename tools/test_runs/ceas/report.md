# Provenance report — CEAS 2008 alone

Rows analysed: 6000 (cap 3000 per source × class); seeds: [0, 1, 2].

## Verdict and warnings

- Source alone gives F1 66.67 vs supervised 99.39: provenance does not explain the overall score (sources mostly contain both classes).

## 1. Datasheet

| Source | File | Class | n | Dated | Years (1st–99th pct.; all) | Licence |
|---|---|---|---|---|---|---|
| CEAS | CEAS_08.csv | 0 | 17312 | 100.0% | 2008 (all: 2005–2008) | not stated |
| CEAS | CEAS_08.csv | 1 | 21842 | 99.9% | 2008 (all: 1998–2020) | not stated |

## 2. Label–source link

Purity (share of the majority class, full counts before the cap) per source: CEAS 55.8%

| Predictor | F1 | Balanced accuracy |
|---|---|---|
| Supervised, random 80/20 split | 99.39 ± 1.23 | 99.39 ± 1.21 |
| Source-majority oracle (source → its majority label) | 66.67 ± 0.00 | 50.00 ± 0.00 |

## 4. Source separability with the class held fixed

- class_1: needs at least 2 sources with >= 400 class-1 emails; skipped
- class_0: needs at least 2 sources with >= 400 class-0 emails; skipped

## 5. Date

- Rows with a parseable date: 99.9%
- Year-majority rule, all dated rows: balanced accuracy 50.17 ± 0.42 (50 = chance)
- Within CEAS only: balanced accuracy 50.17 ± 0.42
- Years by class (1st–99th percentile): class 0: 2008–2008; class 1: 2008–2008

## 6. Leave one source out

| Held-out source | n | Result |
|---|---|---|
| CEAS | — | only one source; nothing to hold out |

---
Model: TF-IDF (1–2 grams) + logistic regression. Intervals: 95% Student t over seeds.
This report measures provenance signals; it does not measure phishing-detection ability.
