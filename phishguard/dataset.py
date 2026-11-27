"""Minimal CSV loader used by the paper's experiments.

This is a standalone copy of ``load_emails`` and its helpers from the author's
PhishGuard project (``phishguard/dataset.py``), included so that the
experiments in this repository run without the rest of PhishGuard. Only the
loading and label-normalisation logic is copied; it is functionally identical
to the original, minus logging. PhishGuard's heuristic scoring is not needed
by any result in the paper.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Map normalised field names -> column names seen in the wild.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sender":   ("sender", "from", "from_email", "from_address", "sender_email",
                 "fromaddress", "From"),
    "receiver": ("receiver", "to", "to_email", "to_address", "recipient",
                 "recipient_email", "To"),
    "subject":  ("subject", "subject_line", "email_subject", "Subject"),
    "body":     ("body", "content", "text", "message", "email_text",
                 "email_body", "html", "raw_body", "Body"),
    "urls":     ("urls", "url", "links", "link", "extracted_urls"),
    "date":     ("date", "timestamp", "date_received", "received_at", "Date"),
    "label":    ("label", "class", "is_phishing", "category", "type",
                 "target", "Label"),
}


@dataclass
class EmailRecord:
    sender: str = ""
    receiver: str = ""
    subject: str = ""
    body: str = ""
    urls: str = ""
    date: str = ""
    label: str = ""
    raw_row: dict[str, str] = field(default_factory=dict)


def _build_column_map(fieldnames: Iterable[str]) -> dict[str, str]:
    """Return a {our_field: actual_column_name} map for this CSV."""
    actual = list(fieldnames)
    lower = {col.lower(): col for col in actual}
    mapping: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in actual:
                mapping[canonical] = alias
                break
            if alias.lower() in lower:
                mapping[canonical] = lower[alias.lower()]
                break
    return mapping


_PHISHING_LABELS = {"1", "phishing", "phish", "spam", "malicious", "true",
                    "yes", "phishing email"}
_LEGIT_LABELS = {"0", "legitimate", "ham", "benign", "safe", "false",
                 "no", "legit"}


def _normalise_label(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in _PHISHING_LABELS:
        return "phishing"
    if v in _LEGIT_LABELS:
        return "legitimate"
    return v or "unknown"


def load_emails(path: str | Path, limit: int | None = None) -> list[EmailRecord]:
    """Load and normalise a phishing-email CSV: one EmailRecord per row."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p.resolve()}")

    records: list[EmailRecord] = []
    csv.field_size_limit(10_000_000)

    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {p}")
        col_map = _build_column_map(reader.fieldnames)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            records.append(
                EmailRecord(
                    sender=row.get(col_map.get("sender", ""), "") or "",
                    receiver=row.get(col_map.get("receiver", ""), "") or "",
                    subject=row.get(col_map.get("subject", ""), "") or "",
                    body=row.get(col_map.get("body", ""), "") or "",
                    urls=row.get(col_map.get("urls", ""), "") or "",
                    date=row.get(col_map.get("date", ""), "") or "",
                    label=_normalise_label(row.get(col_map.get("label", ""), "")),
                    raw_row=row,
                )
            )
    return records
