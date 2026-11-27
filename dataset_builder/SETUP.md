# Rebuilding the corpora and splits

The corpora are not redistributed in this repository. They are public, and the
scripts here rebuild the exact splits used in the paper.

## 1. Obtain the source CSVs

The six corpora are distributed together as the *Phishing Email Dataset* on
Kaggle (Al-Subaiey et al., 2024). The uploader labels it CC BY-SA 4.0, but the
underlying corpora keep their own terms; the paper's Data Availability section
records them. Download and place these files in a directory named
`phishguard_dataset_builder/` **next to this repository** (same parent folder):

```
CEAS_08.csv  Enron.csv  Ling.csv  Nazario.csv  Nigerian_Fraud.csv  SpamAssasin.csv
```

Note the spelling `SpamAssasin.csv` (one `s` in the second word) — this matches
the upstream distribution and the scripts expect it.

Headers as distributed (column order varies; the scripts read by name):

```
CEAS_08, SpamAssasin:           sender,receiver,date,subject,body,label,urls
Nazario, Nigerian_Fraud:        sender,receiver,date,subject,body,urls,label
Enron, Ling:                    subject,body,label
```

## 2. Build the splits

`prepare_dataset.py` reads and writes in its own folder, so copy it next to
the CSVs first. From the repository root:

```bash
cp dataset_builder/prepare_dataset.py ../phishguard_dataset_builder/
cd ../phishguard_dataset_builder
python3 prepare_dataset.py
```

This deduplicates on a hash of sender, subject and body prefix, drops messages
under 30 body characters or without a sender, balances the classes, and writes
a stratified 30/70 split with a fixed seed.

Expected output: `phishguard_dev.csv` (930 rows + header) and
`phishguard_test.csv` (2,170 rows + header).

Verify:

```bash
wc -l phishguard_test.csv    # 2171
```

If that number differs, the upstream CSVs have changed and the reported
figures will not reproduce exactly. Report the row count you get alongside any
numbers you publish.

## 3. Variants

`prepare_dataset.py` honours a `PC_VARIANT` environment variable used to build
the alternative pairings in `experiments/generalize.py`.

## Other scripts here

| Script | Purpose |
|---|---|
| `parse_nazario.py` | Converts raw Nazario `.mbox` archives to the common schema |
| `parse_enron.py` | Extracts legitimate mail from the Enron maildir tree |
| `combine_and_evaluate.py` | Runs the rule-based baseline over a built split |
| `tune_thresholds.py` | Dev-set-only threshold calibration |
| `evaluate_final.py` | Held-out test metrics and PR curve |

The last three belong to the cascade prototype, not to the paper's claims.
