# provenance_report.py

A single-file audit that asks one question of a labelled email dataset: how much of the label can be predicted from where an email came from (its source collection, or its year) rather than from its content?

Requirements: Python 3.9+, scikit-learn, numpy. No GPU.

## Input
One CSV per source collection. Each CSV needs a label column and one or more text columns. A date column is optional; defaults are `label`, `subject,body` and `date`, and flags change them.

## Examples (the three test runs)

```bash
D=path/to/phishguard_dataset_builder     # folder with the six Kaggle CSVs

# 1. benchmark-style pair: Nazario phishing vs SpamAssassin ham, transfer test with CEAS ham
python tools/provenance_report.py \
  --source Nazario=$D/Nazario.csv:1 --source SpamAssassin=$D/SpamAssasin.csv:0 \
  --transfer SpamAssassin:CEAS=$D/CEAS_08.csv:0 \
  --licence "Nazario=CC BY 4.0" --licence "SpamAssassin=none stated (copyright stays with senders)" \
  --title "Nazario phishing + SpamAssassin ham" --out reports/benchmark

# 2. the six-corpus compilation, with a Nazario exposure sweep
python tools/provenance_report.py \
  --source CEAS=$D/CEAS_08.csv --source Enron=$D/Enron.csv --source Ling=$D/Ling.csv \
  --source SpamAssassin=$D/SpamAssasin.csv --source Nazario=$D/Nazario.csv \
  --source Nigerian=$D/Nigerian_Fraud.csv --target Nazario \
  --licence "Nazario=CC BY 4.0" --licence "Nigerian=CC BY-SA 3.0 US (CLAIR source)" \
  --title "Six-corpus compilation" --out reports/compilation

# 3. a single-source corpus (what a clean result looks like)
python tools/provenance_report.py --source CEAS=$D/CEAS_08.csv \
  --title "CEAS 2008 alone" --out reports/ceas
```

`NAME=PATH:LABEL` keeps only rows with that label (e.g. `:0` for ham only).

## Output
- `report.md` contains the verdict and warnings, then seven sections: datasheet, label–source link, transfer, separability with the class held fixed, date, leave-one-source-out, and exposure.
- `report.json` contains every number.

## Notes
- Rows are capped at `--max-per-group` (default 3000) per source × class to bound run time. Purity and the source-majority labels use the full counts.
- Years in the datasheet are shown as the 1st–99th percentile, so a few mis-dated emails do not distort the range.
- Intervals are 95% Student-t over `--seeds` (default 3).
- The report measures provenance signals. It does not measure phishing-detection ability.
- The numbers here come from the Kaggle copies of the corpora with a 3,000 cap. They will not match the paper's tables, which use the Zenodo benchmark splits and uncapped data.
