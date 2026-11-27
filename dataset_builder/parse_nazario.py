"""
parse_nazario.py
────────────────
Converts Nazario phishing .mbox files into a PhishGuard-compatible CSV.

STEP 1 — Download these files in your browser and place them in this folder
(phishguard_dataset_builder/):

    http://monkey.org/~jose/phishing/phishing3.mbox
    http://monkey.org/~jose/phishing/phishing2.mbox
    http://monkey.org/~jose/phishing/phishing0.mbox

STEP 2 — Run:
    python parse_nazario.py

OUTPUT: phishing_nazario.csv  (sender, receiver, date, subject, body, urls, label)
"""

import mailbox
import email
import re
import csv
import os
import quopri
import base64
from pathlib import Path
from email.header import decode_header, make_header

# ── Regex helpers ────────────────────────────────────────────────
URL_RE    = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)
EMAIL_RE  = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)


def _decode_header_value(val: str | None) -> str:
    if not val:
        return ""
    try:
        return str(make_header(decode_header(val)))
    except Exception:
        return val or ""


def _get_body(msg) -> str:
    """Extract plain-text body from an email.Message, handling MIME."""
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        body_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        body_parts.append(payload.decode("latin-1", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_parts.append(payload.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                body_parts.append(payload.decode("latin-1", errors="replace"))
    return "\n".join(body_parts).strip()


def _extract_urls(text: str) -> str:
    """Return comma-separated URLs found in text."""
    urls = list(dict.fromkeys(URL_RE.findall(text)))  # deduplicated, order preserved
    return ", ".join(urls[:30])  # cap at 30 to keep CSV manageable


def parse_mbox(mbox_path: str) -> list[dict]:
    """Parse one .mbox file and return list of email dicts (label=1)."""
    rows = []
    try:
        box = mailbox.mbox(mbox_path)
    except Exception as e:
        print(f"  ✗ Could not open {mbox_path}: {e}")
        return rows

    for i, msg in enumerate(box):
        try:
            sender   = _decode_header_value(msg.get("From", ""))
            receiver = _decode_header_value(msg.get("To", ""))
            date     = _decode_header_value(msg.get("Date", ""))
            subject  = _decode_header_value(msg.get("Subject", ""))
            body     = _get_body(msg)
            urls     = _extract_urls(body)

            # Skip empty emails
            if not body and not subject:
                continue

            rows.append({
                "sender":   sender[:500],
                "receiver": receiver[:300],
                "date":     date[:100],
                "subject":  subject[:300],
                "body":     body[:8000],    # cap body at 8K chars
                "urls":     urls,
                "label":    "1"             # phishing
            })
        except Exception as e:
            print(f"  ⚠ Skipping message {i} in {mbox_path}: {e}")
            continue

    print(f"  ✓ {mbox_path}: {len(rows)} phishing emails parsed")
    return rows


def main():
    here = Path(__file__).parent

    # Find all .mbox files in this folder
    mbox_files = sorted(here.glob("*.mbox"))
    if not mbox_files:
        print("❌  No .mbox files found in this folder.")
        print("   Download from http://monkey.org/~jose/phishing/ and place here.")
        return

    print(f"Found {len(mbox_files)} mbox file(s): {[f.name for f in mbox_files]}")
    print()

    all_rows = []
    for mbox_path in mbox_files:
        rows = parse_mbox(str(mbox_path))
        all_rows.extend(rows)

    if not all_rows:
        print("❌  No emails parsed. Check the .mbox files.")
        return

    out_path = here / "phishing_nazario.csv"
    fieldnames = ["sender", "receiver", "date", "subject", "body", "urls", "label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"✅  Written {len(all_rows)} phishing emails → {out_path.name}")
    print(f"   Columns: {fieldnames}")
    print()
    print("Next step: run  python parse_enron.py  to build the legitimate email CSV.")


if __name__ == "__main__":
    main()
