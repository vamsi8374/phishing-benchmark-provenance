# Provenance report — Nazario phishing + SpamAssassin ham

Rows analysed: 5656 (no cap); seeds: [0, 1, 2].

## Verdict and warnings

- Single-class sources (label is fixed by source): Nazario, SpamAssassin. Any detector that recognises these sources gets their label for free.
- SEVERE: the provenance-transfer detector (SpamAssassin_vs_CEAS) reaches F1 97.61 without any label of the other class, within 5 points of the supervised model (99.73).
- Every source contains one class only, so source and label are the same variable in this dataset: the supervised score cannot be separated from source recognition without an external reference (use --transfer).
- SEVERE: the email year alone gives balanced accuracy 100.00.

## 1. Datasheet

| Source | File | Class | n | Dated | Years (1st–99th pct.; all) | Licence |
|---|---|---|---|---|---|---|
| Nazario | Nazario.csv | 1 | 1565 | 99.9% | 2015–2022 (all: 2015–2022) | CC BY 4.0 |
| SpamAssassin | SpamAssasin.csv | 0 | 4091 | 100.0% | 2002 (all: 2002–2028) | none stated (copyright stays with senders) |

## 2. Label–source link

Purity (share of the majority class, full counts before the cap) per source: Nazario 100.0%, SpamAssassin 100.0%

| Predictor | F1 | Balanced accuracy |
|---|---|---|
| Supervised, random 80/20 split | 99.73 ± 0.23 | 99.73 ± 0.23 |
| Source-majority oracle (source → its majority label) | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Label-free source detector → majority label | 99.30 ± 0.62 | 99.31 ± 0.61 |

Source-identification accuracy: 99.62 ± 0.34

## 3. Provenance transfer

| Detector | F1 | Balanced acc. | Null F1 |
|---|---|---|---|
| is-SpamAssassin vs CEAS (both class 0); looks like SpamAssassin -> class 0 | 97.61 ± 0.58 | 99.06 ± 0.23 | 42.13 ± 15.71 |

## 4. Source separability with the class held fixed

- class_1: needs at least 2 sources with >= 400 class-1 emails; skipped
- class_0: needs at least 2 sources with >= 400 class-0 emails; skipped

## 5. Date

- Rows with a parseable date: 100.0%
- Year-majority rule, all dated rows: balanced accuracy 100.00 ± 0.00 (50 = chance)
- Years by class (1st–99th percentile): class 0: 2002–2002; class 1: 2015–2022

## 6. Leave one source out

| Held-out source | n | Result |
|---|---|---|
| Nazario | — | remaining sources contain one class only; skipped |
| SpamAssassin | — | remaining sources contain one class only; skipped |

---
Model: TF-IDF (1–2 grams) + logistic regression. Intervals: 95% Student t over seeds.
This report measures provenance signals; it does not measure phishing-detection ability.
