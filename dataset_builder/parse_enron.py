"""
parse_enron.py
──────────────
Converts Enron legitimate emails into a PhishGuard-compatible CSV.

STEP 1 — Download the Enron email dataset (legitimate emails only).

Option A — full corpus (~432 MB tar):
  https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz
  Extract and point ENRON_DIR below to any subfolder (e.g. maildir/lay-k/inbox/)

Option B — small Enron sample on Kaggle (~70 MB):
  https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
  This gives emails.csv — run this script with ENRON_CSV mode (see code below).

STEP 2 — Place extracted files in  phishguard_dataset_builder/enron/
          or update ENRON_DIR below, then run:
    python parse_enron.py

OUTPUT: legitimate_enron.csv  (sender, receiver, date, subject, body, urls, label=0)

NOTE: This script also accepts an emails.csv from Kaggle Enron dataset.
      Just place it as  phishguard_dataset_builder/enron/emails.csv
"""

import re
import csv
import email
from email.header import decode_header, make_header
from pathlib import Path

URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)

MAX_EMAILS = 5000   # cap — we only need ~2,500 to match phishing count


def _decode_header_value(val):
    if not val:
        return ""
    try:
        return str(make_header(decode_header(val)))
    except Exception:
        return val or ""


def _get_body(msg) -> str:
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        parts.append(payload.decode(charset, errors="replace"))
                    except LookupError:
                        parts.append(payload.decode("latin-1", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                parts.append(payload.decode(charset, errors="replace"))
            except LookupError:
                parts.append(payload.decode("latin-1", errors="replace"))
    return "\n".join(parts).strip()


def _extract_urls(text: str) -> str:
    urls = list(dict.fromkeys(URL_RE.findall(text)))
    return ", ".join(urls[:20])


def parse_raw_email_file(path: Path) -> dict | None:
    """Parse a single raw email file (Enron maildir format)."""
    try:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw)
        sender   = _decode_header_value(msg.get("From", ""))
        receiver = _decode_header_value(msg.get("To", ""))
        date     = _decode_header_value(msg.get("Date", ""))
        subject  = _decode_header_value(msg.get("Subject", ""))
        body     = _get_body(msg)
        urls     = _extract_urls(body)

        if not body and not subject:
            return None

        return {
            "sender":   sender[:500],
            "receiver": receiver[:300],
            "date":     date[:100],
            "subject":  subject[:300],
            "body":     body[:8000],
            "urls":     urls,
            "label":    "0"   # legitimate
        }
    except Exception:
        return None


def parse_kaggle_csv(csv_path: Path) -> list[dict]:
    """Parse Kaggle Enron emails.csv (columns: file, message)."""
    rows = []
    print(f"  Reading Kaggle CSV: {csv_path.name} …")
    import csv as _csv

    # Kaggle Enron CSV can have very large fields
    _csv.field_size_limit(10_000_000)

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(rows) >= MAX_EMAILS:
                break
            raw_msg = row.get("message", "")
            if not raw_msg:
                continue
            try:
                msg      = email.message_from_string(raw_msg)
                sender   = _decode_header_value(msg.get("From", ""))
                receiver = _decode_header_value(msg.get("To", ""))
                date     = _decode_header_value(msg.get("Date", ""))
                subject  = _decode_header_value(msg.get("Subject", ""))
                body     = _get_body(msg)
                urls     = _extract_urls(body)

                if not body and not subject:
                    continue

                rows.append({
                    "sender":   sender[:500],
                    "receiver": receiver[:300],
                    "date":     date[:100],
                    "subject":  subject[:300],
                    "body":     body[:8000],
                    "urls":     urls,
                    "label":    "0"
                })
            except Exception:
                continue

    print(f"  ✓ Kaggle CSV: {len(rows)} legitimate emails parsed")
    return rows


def parse_enron_directory(enron_dir: Path) -> list[dict]:
    """Walk a directory of raw email files (Enron maildir)."""
    rows = []
    # Recursively find all files with no extension or .txt extension
    candidates = list(enron_dir.rglob("*"))
    candidates = [p for p in candidates if p.is_file() and p.suffix in ("", ".txt")]

    print(f"  Found {len(candidates)} email files in {enron_dir}")

    for path in candidates:
        if len(rows) >= MAX_EMAILS:
            break
        row = parse_raw_email_file(path)
        if row:
            rows.append(row)

    print(f"  ✓ Directory: {len(rows)} legitimate emails parsed")
    return rows


def main():
    here = Path(__file__).parent
    enron_dir = here / "enron"

    rows = []

    # Priority 1: emails.csv from Kaggle
    kaggle_csv = enron_dir / "emails.csv"
    if kaggle_csv.exists():
        rows = parse_kaggle_csv(kaggle_csv)
    # Priority 2: maildir directory tree
    elif enron_dir.exists() and any(enron_dir.iterdir()):
        rows = parse_enron_directory(enron_dir)
    else:
        print("❌  No Enron data found.")
        print("   Create folder:  phishguard_dataset_builder/enron/")
        print("   Then place one of:")
        print("   • emails.csv   (from Kaggle Enron dataset)")
        print("   • Extracted maildir folders (from CMU Enron corpus)")
        print()
        print("   Download links:")
        print("   Kaggle: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
        print("   CMU:    https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz")
        return

    if not rows:
        print("❌  No emails parsed.")
        return

    # Cap at 3000 to have some buffer
    rows = rows[:3000]

    out_path = here / "legitimate_enron.csv"
    fieldnames = ["sender", "receiver", "date", "subject", "body", "urls", "label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"✅  Written {len(rows)} legitimate emails → {out_path.name}")
    print()
    print("Next step: run  python combine_and_evaluate.py")


if __name__ == "__main__":
    main()
