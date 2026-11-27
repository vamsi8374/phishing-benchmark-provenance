# Provenance report — Six-corpus compilation

Rows analysed: 24142 (cap 3000 per source × class); seeds: [0, 1, 2].

## Verdict and warnings

- Single-class sources (label is fixed by source): Nazario, Nigerian. Any detector that recognises these sources gets their label for free.
- Source alone gives F1 65.32 vs supervised 98.65: provenance does not explain the overall score (sources mostly contain both classes).
- Sources are recognisable with class 1 held fixed (87.83% vs chance 16.67%): a provenance signal exists.
- Sources are recognisable with class 0 held fixed (95.15% vs chance 25.0%): a provenance signal exists.
- Leave-Nazario-out: 74.57% of Nazario is assigned the positive class by a model that never saw it.
- Leave-Nigerian-out: 99.27% of Nigerian is assigned the positive class by a model that never saw it.

## 1. Datasheet

| Source | File | Class | n | Dated | Years (1st–99th pct.; all) | Licence |
|---|---|---|---|---|---|---|
| CEAS | CEAS_08.csv | 0 | 17312 | 100.0% | 2008 (all: 2005–2008) | not stated |
| CEAS | CEAS_08.csv | 1 | 21842 | 99.9% | 2008 (all: 1998–2020) | not stated |
| Enron | Enron.csv | 0 | 15791 | 0.0% | — | not stated |
| Enron | Enron.csv | 1 | 13976 | 0.0% | — | not stated |
| Ling | Ling.csv | 0 | 2401 | 0.0% | — | not stated |
| Ling | Ling.csv | 1 | 458 | 0.0% | — | not stated |
| SpamAssassin | SpamAssasin.csv | 0 | 4091 | 100.0% | 2002 (all: 2002–2028) | not stated |
| SpamAssassin | SpamAssasin.csv | 1 | 1718 | 96.4% | 2001–2002 (all: 1993–2020) | not stated |
| Nazario | Nazario.csv | 1 | 1565 | 99.9% | 2015–2022 (all: 2015–2022) | CC BY 4.0 |
| Nigerian | Nigerian_Fraud.csv | 1 | 3332 | 85.2% | 2001–2007 (all: 1990–2015) | CC BY-SA 3.0 US (CLAIR source) |

## 2. Label–source link

Purity (share of the majority class, full counts before the cap) per source: CEAS 55.8%, Enron 53.0%, Ling 84.0%, Nazario 100.0%, Nigerian 100.0%, SpamAssassin 70.4%

| Predictor | F1 | Balanced accuracy |
|---|---|---|
| Supervised, random 80/20 split | 98.65 ± 0.36 | 98.57 ± 0.37 |
| Source-majority oracle (source → its majority label) | 65.32 ± 0.71 | 66.96 ± 0.25 |
| Label-free source detector → majority label | 66.10 ± 0.34 | 67.61 ± 0.94 |

Source-identification accuracy: 94.26 ± 0.18

## 4. Source separability with the class held fixed

- class_1: 6-way accuracy 87.83 ± 1.43 (chance 16.67); null (CEAS split in half) 49.62 ± 3.58
- class_0: 4-way accuracy 95.15 ± 1.36 (chance 25.0); null (CEAS split in half) 50.88 ± 7.54

## 5. Date

- Rows with a parseable date: 61.2%
- Year-majority rule, all dated rows: balanced accuracy 70.88 ± 11.44 (50 = chance)
- Within CEAS only: balanced accuracy 50.17 ± 0.42
- Within SpamAssassin only: balanced accuracy 53.62 ± 0.74
- Years by class (1st–99th percentile): class 0: 2002–2008; class 1: 2001–2022

## 6. Leave one source out

| Held-out source | n | Result |
|---|---|---|
| CEAS | 6000 | F1 91.65 (recall 92.97, FPR 9.9) |
| Enron | 6000 | F1 84.34 (recall 90.77, FPR 24.47) |
| Ling | 2859 | F1 87.25 (recall 90.39, FPR 3.21) |
| Nazario | 1565 | 74.57% assigned the positive class |
| Nigerian | 3000 | 99.27% assigned the positive class |
| SpamAssassin | 4718 | F1 90.08 (recall 95.69, FPR 9.6) |

## 7. Exposure sweep — Nazario

Held-out Nazario test emails: 521

| k added | Accuracy on held-out target | F1 on other sources |
|---|---|---|
| 0 | 71.85 ± 18.98 | 98.70 ± 0.41 |
| 10 | 79.21 ± 15.53 | 98.70 ± 0.41 |
| 25 | 90.02 ± 7.20 | 98.67 ± 0.45 |
| 50 | 93.28 ± 6.45 | 98.67 ± 0.47 |
| 100 | 96.42 ± 4.00 | 98.68 ± 0.37 |
| 250 | 97.95 ± 1.53 | 98.63 ± 0.34 |
| 500 | 98.59 ± 1.46 | 98.64 ± 0.33 |

---
Model: TF-IDF (1–2 grams) + logistic regression. Intervals: 95% Student t over seeds.
This report measures provenance signals; it does not measure phishing-detection ability.
