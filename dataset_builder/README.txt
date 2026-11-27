PhishGuard Evaluation Dataset — Build Pipeline
===============================================

RECOMMENDED PATH (use this)
----------------------------
Phishing class : Nazario.csv       (~4,500 real credential-phishing emails, 2015-2021)
Legit class    : SpamAssasin.csv   (label=0 rows — real ham with full headers)

Both use PhishGuard's exact schema: sender,receiver,date,subject,body,label,urls
No column remapping needed. Only the `urls` column needs fixing (it's a 0/1 flag,
not a URL list) — prepare_dataset.py does that automatically.

STEP 1 — Download into THIS folder
-----------------------------------
  https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nazario.csv
  https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/SpamAssasin.csv

  Optional third class (419 scams):
  https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nigerian_Fraud.csv

  Same files also live in the Kaggle bundle you already used for CEAS_08:
  https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
  (Take the PER-SOURCE csv files. Ignore the combined phishing_email.csv —
   that one is preprocessed down to text+label and is unusable.)

STEP 2 — Clean and build
-------------------------
    cd ~/Desktop/phishguard_dataset_builder
    python prepare_dataset.py

  Outputs:
    phishguard_eval_clean.csv   ← deduped, balanced, PhishGuard-ready
    prepare_report.txt          ← every row dropped, with the reason

  READ THE CLASS PROFILE TABLE IT PRINTS. The column that matters is
  "brand-in-name%". If that number is near zero for the phishing class,
  stop — PhishGuard's +30 display-name-mismatch rule cannot fire and your
  F1 will collapse. (Measured on CEAS_08: 0.85% phishing vs 0.50% legit,
  which is exactly why CEAS_08 produced F1 = 0.02%.)

STEP 3 — Evaluate
------------------
    python combine_and_evaluate.py

  Outputs the full metrics table (accuracy / precision / recall / F1 / FPR)
  for PhishGuard at two thresholds plus two baselines, and writes
  real_evaluation_results.json for the paper's tables.


WHAT NOT TO USE
---------------
  CEAS_08.csv            2008 pharma/porn/stock spam. Structurally perfect,
                         semantically wrong. Measured F1 = 0.02% at the shipped
                         Medium threshold; 45.3% at the best possible threshold,
                         against a 55.8% base rate. Essentially chance.
  eduphish_dataset.csv   text + label only. No sender, no headers. Unusable.
  phishing_email.csv     text_combined + label only. Unusable.


OTHER SCRIPTS IN THIS FOLDER
-----------------------------
  parse_nazario.py   Parses raw Nazario .mbox files, if you'd rather build from
                     the original corpus than use the pre-parsed CSV.
                     Also the base for a Phishing Pot .eml parser — swap
                     mailbox.mbox() for email.message_from_bytes() over a
                     glob of .eml files.
  parse_enron.py     Enron ham → CSV. Fallback negative class if you can't get
                     SpamAssassin.
  prepare_dataset.py Cleaner + balancer. START HERE.
  combine_and_evaluate.py  Metrics runner.


MODERN ALTERNATIVE (for the recency objection)
-----------------------------------------------
  Phishing Pot — https://github.com/rf-peixoto/phishing_pot
  ~4,350 raw .eml honeypot samples, last updated June 2026, full unmodified
  headers. Best answer to a reviewer asking why your newest phishing sample
  is from 2021. Use as a second evaluation slice, not a replacement.


CALIBRATION WARNING
-------------------
Even on the right corpus, PhishGuard's shipped bands are miscalibrated.
Excluding the +30 sender rule, a typical phishing email fires urgency (15)
plus a shortener (15) = 30, which is still "Low". Reaching Medium (40) needs
three co-occurring signals. Report a threshold sweep and a PR curve rather
than a single operating point — and widen _BRAND_IMPERSONATION in
phishguard/dataset.py to include the banking/shipping brands that dominate
Nazario (chase, wells fargo, bank of america, hsbc, dhl, fedex, ups,
netflix, linkedin, dropbox, adobe, docusign).
