"""
prepare_dataset.py  —  PhishGuard dataset cleaner + builder
============================================================

Builds a clean, balanced, PhishGuard-ready evaluation CSV from the
"Enron / Ling / CEAS / Nazario / Nigerian / SpamAssassin" collection.

WHY THIS SCRIPT EXISTS
----------------------
The source CSVs already use PhishGuard's exact column names
(sender,receiver,date,subject,body,label,urls) so no remapping is needed.
But three things WILL corrupt your evaluation if you don't fix them:

  1. The `urls` column is a BINARY FLAG (0/1), not a URL list.
     PhishGuard concatenates subject+body+urls and regex-extracts IOCs,
     so a literal "1" contributes nothing — but it also means the column
     is dead weight and misleading in the paper. We rebuild it with the
     real URLs extracted from the body.

  2. Class balance. Nazario is 100% phishing, SpamAssassin is mostly ham.
     Merging them raw gives you a skewed set and inflated accuracy.

  3. Junk rows: empty bodies, empty senders, exact duplicates.
     These silently drag every metric down.

INPUT FILES — download these two and put them in this folder:
  Nazario.csv       https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nazario.csv
  SpamAssasin.csv   https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/SpamAssasin.csv

  (Same files also ship inside the Kaggle bundle you already used for CEAS_08:
   https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset )

OPTIONAL third source for a harder negative class:
  Nigerian_Fraud.csv  (419 scams — phishing-adjacent, no brand impersonation)

RUN:
    python prepare_dataset.py

OUTPUT:
    phishguard_eval_clean.csv     ← feed this to PhishGuard
    prepare_report.txt            ← what was dropped and why
"""

import csv
import os
import re
import sys
import random
import hashlib
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
csv.field_size_limit(10_000_000)

# ── Config ────────────────────────────────────────────────────────────────────
# ── Source selection ──────────────────────────────────────────────────────────
#
# PRIMARY (default): Nazario phishing vs SpamAssassin ham.
#
# Nigerian_Fraud is deliberately NOT in the default. It is advance-fee fraud,
# which is reply-based rather than link-based: including it drops the phishing
# class's URL rate from 12.4% to 25.9% *below* the ham class and inverts every
# URL-derived signal. That is a real phenomenon worth a paragraph in the paper,
# but it makes for a muddled primary result. Run it as a labelled variant:
#
#     PC_VARIANT=mixed python prepare_dataset.py
#
# CEAS_08 is excluded entirely from training/eval splits. It is reserved as an
# out-of-distribution transfer set — see experiments/make_transfer_set.py.
VARIANT = os.environ.get("PC_VARIANT", "primary")

if VARIANT == "mixed":
    PHISH_SOURCES = ["Nazario.csv", "Nigerian_Fraud.csv"]
else:
    PHISH_SOURCES = ["Nazario.csv"]
HAM_SOURCES = ["SpamAssasin.csv"]
OUT_CSV       = HERE / "phishguard_eval_clean.csv"
DEV_CSV       = HERE / "phishguard_dev.csv"
TEST_CSV      = HERE / "phishguard_test.csv"
REPORT_TXT    = HERE / "prepare_report.txt"

MIN_BODY_CHARS = 30       # drop near-empty emails
BALANCE        = True     # downsample the majority class to 50/50
SEED           = 42
DEV_FRACTION   = 0.30     # tune thresholds on this; report on the other 70%

FIELDS = ["sender", "receiver", "date", "subject", "body", "urls", "label"]

URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
DISPLAY_NAME_RE = re.compile(r'\s*"?([^"<>]+)"?\s*<([^>]+)>')
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(s: str) -> str:
    """Strip control chars that break openpyxl and confuse regexes."""
    return CTRL_RE.sub("", s or "").strip()


def rebuild_urls(body: str) -> str:
    """
    Replace the useless binary `urls` flag with the actual URLs.
    Deduplicated, order-preserved, capped at 30.
    """
    found = list(dict.fromkeys(URL_RE.findall(body or "")))
    return ", ".join(found[:30])


def row_fingerprint(row: dict) -> str:
    """Content hash for exact-duplicate removal."""
    blob = (row.get("sender", "") + "|" + row.get("subject", "") + "|" +
            (row.get("body", "") or "")[:2000])
    return hashlib.md5(blob.encode("utf-8", "replace")).hexdigest()


def load_source(path: Path, want_label: str, stats: Counter) -> list[dict]:
    """
    Read one source CSV, keep only rows matching want_label ('1' or '0'),
    clean them, and return PhishGuard-shaped dicts.
    """
    if not path.exists():
        print(f"  · skipped (not found): {path.name}")
        return []

    kept = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        missing = {"sender", "subject", "body", "label"} - cols
        if missing:
            print(f"  ✗ {path.name}: missing required columns {sorted(missing)} — skipped")
            stats[f"{path.name}: unusable schema"] += 1
            return []

        for row in reader:
            stats[f"{path.name}: rows read"] += 1

            label = (row.get("label") or "").strip()
            if label != want_label:
                continue

            sender  = clean_text(row.get("sender", ""))
            subject = clean_text(row.get("subject", ""))
            body    = clean_text(row.get("body", ""))

            if len(body) < MIN_BODY_CHARS:
                stats[f"{path.name}: dropped (empty/short body)"] += 1
                continue
            if not sender:
                stats[f"{path.name}: dropped (no sender)"] += 1
                continue

            kept.append({
                "sender":   sender[:500],
                "receiver": clean_text(row.get("receiver", ""))[:300],
                "date":     clean_text(row.get("date", ""))[:100],
                "subject":  subject[:300],
                "body":     body[:8000],
                "urls":     rebuild_urls(body),      # ← the important fix
                "label":    want_label,
                "_src":     path.name,
            })

    print(f"  ✓ {path.name}: kept {len(kept)} rows (label={want_label})")
    return kept


def profile(rows: list[dict], name: str) -> dict:
    """Measure the properties that actually drive PhishGuard's score."""
    if not rows:
        return {}
    n = len(rows)
    brands = ("microsoft", "google", "office 365", "office365", "outlook",
              "apple", "paypal", "amazon", "onedrive", "sharepoint")
    disp = sum(1 for r in rows if DISPLAY_NAME_RE.match(r["sender"]))
    brand_disp = 0
    for r in rows:
        m = DISPLAY_NAME_RE.match(r["sender"])
        if m and any(b in m.group(1).lower() for b in brands):
            brand_disp += 1
    has_url  = sum(1 for r in rows if r["urls"])
    has_recv = sum(1 for r in rows if r["receiver"])
    return {
        "name": name,
        "rows": n,
        "display_name_sender_pct": round(100 * disp / n, 1),
        "brand_in_display_name_pct": round(100 * brand_disp / n, 2),
        "has_real_urls_pct": round(100 * has_url / n, 1),
        "has_receiver_pct": round(100 * has_recv / n, 1),
    }


def main():
    random.seed(SEED)
    stats: Counter = Counter()
    lines: list[str] = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("  PHISHGUARD DATASET PREPARATION")
    log("=" * 70)
    log()

    log("Loading PHISHING sources (label=1):")
    phish = []
    for fn in PHISH_SOURCES:
        phish += load_source(HERE / fn, "1", stats)

    log()
    log("Loading LEGITIMATE sources (label=0):")
    ham = []
    for fn in HAM_SOURCES:
        ham += load_source(HERE / fn, "0", stats)

    if not phish or not ham:
        log()
        log("❌  Need at least one phishing source AND one legitimate source.")
        log("   Download Nazario.csv and SpamAssasin.csv into this folder.")
        log("   See the header comment in this file for direct links.")
        REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    # ── Deduplicate across the whole pool ─────────────────────────────────────
    log()
    log("Deduplicating …")
    seen: set[str] = set()
    def dedupe(rows):
        out = []
        for r in rows:
            fp = row_fingerprint(r)
            if fp in seen:
                continue
            seen.add(fp)
            out.append(r)
        return out

    before_p, before_h = len(phish), len(ham)
    phish = dedupe(phish)
    ham   = dedupe(ham)
    log(f"  phishing:   {before_p} → {len(phish)}  (-{before_p - len(phish)} dupes)")
    log(f"  legitimate: {before_h} → {len(ham)}  (-{before_h - len(ham)} dupes)")

    # ── Balance ───────────────────────────────────────────────────────────────
    if BALANCE:
        n = min(len(phish), len(ham))
        random.shuffle(phish)
        random.shuffle(ham)
        phish, ham = phish[:n], ham[:n]
        log()
        log(f"Balanced to {n} + {n} = {2*n} emails (seed={SEED})")

    combined = phish + ham
    random.shuffle(combined)

    # ── Profile the two classes ───────────────────────────────────────────────
    log()
    log("── Class profile (these drive PhishGuard's risk score) ──")
    hdr = (f"{'class':<12}{'rows':>7}{'displayname%':>14}"
           f"{'brand-in-name%':>16}{'real URLs%':>12}{'receiver%':>11}")
    log(hdr)
    log("-" * len(hdr))
    for rows, nm in ((phish, "phishing"), (ham, "legitimate")):
        p = profile(rows, nm)
        log(f"{p['name']:<12}{p['rows']:>7}{p['display_name_sender_pct']:>14}"
            f"{p['brand_in_display_name_pct']:>16}"
            f"{p['has_real_urls_pct']:>12}{p['has_receiver_pct']:>11}")
    log()
    log("  NOTE: 'brand-in-name%' is the single most important number here.")
    log("  PhishGuard's +30 display-name-mismatch rule ONLY fires when the")
    log("  display name contains a brand (microsoft/paypal/apple/...) that")
    log("  is absent from the sending domain. If this is near 0% for the")
    log("  phishing class, the dataset cannot exercise PhishGuard's")
    log("  strongest signal and your F1 will collapse. That is exactly")
    log("  what happens with CEAS_08 (0.72%).")

    # ── Source mix ────────────────────────────────────────────────────────────
    log()
    log("── Source contribution ──")
    for src, cnt in Counter(r["_src"] for r in combined).most_common():
        log(f"  {src:<26} {cnt:>7}")

    # ── Write full set ────────────────────────────────────────────────────────
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT_CSV, combined)
    log()
    log(f"✅  Written {len(combined)} emails → {OUT_CSV.name}")
    log(f"    Columns: {FIELDS}")

    # ── Stratified dev / test split ───────────────────────────────────────────
    #
    # Thresholds and the brand list are tuned on DEV only; every number that
    # goes in the paper is measured on TEST, which is never looked at during
    # calibration. Stratified so both splits keep the exact 50/50 class ratio.
    dev, test = [], []
    for lbl in ("1", "0"):
        pool = [r for r in combined if r["label"] == lbl]
        random.Random(SEED).shuffle(pool)
        cut = int(round(len(pool) * DEV_FRACTION))
        dev  += pool[:cut]
        test += pool[cut:]
    random.Random(SEED + 1).shuffle(dev)
    random.Random(SEED + 2).shuffle(test)

    write_csv(DEV_CSV, dev)
    write_csv(TEST_CSV, test)

    log()
    log("── Dev / test split (stratified, seed={}) ──".format(SEED))
    for nm, rows in (("dev  (tuning)", dev), ("test (reporting)", test)):
        np_ = sum(1 for r in rows if r["label"] == "1")
        log(f"  {nm:<20} {len(rows):>6} rows   "
            f"{np_} phishing / {len(rows)-np_} legitimate")
    log(f"  → {DEV_CSV.name} / {TEST_CSV.name}")
    log()
    log("  Tune ONLY on dev. Touching test before the final run is the")
    log("  thing a reviewer will ask about, and there is no way to undo it.")

    # ── Drop reasons ──────────────────────────────────────────────────────────
    log()
    log("── Rows dropped, by reason ──")
    for k, v in sorted(stats.items()):
        log(f"  {k:<48} {v:>8}")

    log()
    log("NEXT:")
    log("  python tune_thresholds.py    # calibrate on dev split only")
    log("  python evaluate_final.py     # report on held-out test split")
    log("=" * 70)

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
