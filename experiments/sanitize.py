"""
Build a de-leaked evaluation corpus.

    python experiments/sanitize.py

WHY
---
The leakage probe shows the standard Nazario-vs-SpamAssassin pairing is
separable to F1 99.91% by the Date header alone, and to 96.9% by the
punctuation/whitespace shape of the body with every letter and digit
destroyed. Neither view contains any persuasive content. A model scoring well
on this pairing has, in the general case, learned which corpus a message came
from.

That makes headline accuracies on this benchmark uninterpretable — including
those of the rule engine, the supervised baselines, and the cascade. All of
them have to be re-measured on a corpus where the shortcut is closed.

WHAT IS REMOVED, AND WHY EACH ONE
---------------------------------
1. Date field, entirely.
   Collection epochs do not overlap. Nothing recoverable here is about
   phishing.

2. List-serv and MTA boilerplate.
   SpamAssassin ham is largely mailing-list traffic and carries footers
   ("To unsubscribe...", "Sponsored by...", "_______" rules, Yahoo! Groups
   blocks). These are a property of how the ham was collected.

3. Whitespace and punctuation shape, normalised.
   Line-wrap width, tab conventions, repeated-dash rules and quoted-printable
   residue differ systematically between the two collection pipelines. This is
   what nonword_shape was reading.

4. Charset/MIME residue and soft-break artefacts (=20, =\n, =3D).
   Encoding conventions changed between 2004 and 2015.

WHAT IS DELIBERATELY KEPT
-------------------------
The sender line is kept. Display-name-versus-domain disagreement is a genuine
phishing signal, not an artefact, and removing it would sanitise away the
phenomenon under study. The domain_only probe (83.5%) shows some provenance
does ride along with it; that is reported as a residual limitation rather than
hidden, because deleting the sender would make the task unrepresentative.

Sanitisation is destructive and deliberately conservative in one direction: it
is better to remove a real signal than to leave a shortcut open. Expect every
method's score to fall. That fall is the finding.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishcascade.config import DATA_DIR  # noqa: E402

csv.field_size_limit(10_000_000)

FIELDS = ["sender", "receiver", "date", "subject", "body", "urls", "label"]

IN_OUT = [
    ("phishguard_dev.csv",   "san_dev.csv"),
    ("phishguard_test.csv",  "san_test.csv"),
    ("ceas_transfer.csv",    "san_ceas_transfer.csv"),
]

# ── 2. Collection boilerplate ────────────────────────────────────────────────
_BOILERPLATE = [
    # Mailing-list footers, the dominant SpamAssassin-ham artefact
    re.compile(r"To unsubscribe.*?$", re.I | re.S),
    re.compile(r"You are currently subscribed to.*?$", re.I | re.S),
    re.compile(r"Yahoo!\s*Groups.*?$", re.I | re.S),
    re.compile(r"-{4,}\s*Yahoo!.*?$", re.I | re.S),
    re.compile(r"Sponsored by.*?$", re.I | re.S),
    re.compile(r"_{5,}.*?$", re.S),
    re.compile(r"-{5,}.*?$", re.S),
    re.compile(r"={5,}.*?$", re.S),
    re.compile(r"^\s*--\s*$.*", re.M | re.S),           # sig delimiter onward
    re.compile(r"This (?:e-?mail|message) (?:was )?(?:sent|scanned).*?$", re.I | re.S),
    re.compile(r"https?://\S*(?:unsubscribe|listinfo|mailman)\S*", re.I),
    re.compile(r"\b(?:ilug|exmh|razor|spamassassin|zzzz+)[\w.-]*@[\w.-]+", re.I),
]

# ── 4. Encoding residue ──────────────────────────────────────────────────────
_QP_SOFTBREAK = re.compile(r"=\r?\n")
_QP_HEX       = re.compile(r"=[0-9A-F]{2}")
_CTRL         = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CHARSET      = re.compile(r"(?:charset|content-type|content-transfer-encoding)"
                           r"\s*[:=]\s*\S+", re.I)

# ── 3. Shape normalisation ───────────────────────────────────────────────────
_WS      = re.compile(r"[ \t ]+")
_NL      = re.compile(r"\s*\n\s*")
_PUNCT   = re.compile(r"([^\w\s])\1{2,}")     # collapse !!!! ---- ....


def sanitize_body(body: str) -> str:
    s = body or ""
    s = _QP_SOFTBREAK.sub("", s)
    s = _QP_HEX.sub(" ", s)
    s = _CTRL.sub(" ", s)
    s = _CHARSET.sub(" ", s)
    for pat in _BOILERPLATE:
        s = pat.sub(" ", s)
    # Shape normalisation: kill line-wrap width, tab style, repeated rules.
    s = _PUNCT.sub(r"\1", s)
    s = _WS.sub(" ", s)
    s = _NL.sub("\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def sanitize_subject(subj: str) -> str:
    s = _QP_HEX.sub(" ", subj or "")
    # Strip list tags like "[ILUG]" / "[SAtalk]" — pure provenance.
    s = re.sub(r"^\s*(?:\[[^\]]{1,20}\]\s*)+", "", s)
    s = re.sub(r"^\s*(?:re|fwd|fw)\s*:\s*", "", s, flags=re.I)
    s = _PUNCT.sub(r"\1", s)
    return _WS.sub(" ", s).strip()


def process(src: Path, dst: Path) -> tuple[int, int]:
    if not src.exists():
        return 0, 0
    kept = dropped = 0
    with src.open("r", encoding="utf-8", errors="replace", newline="") as fh, \
         dst.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in csv.DictReader(fh):
            body = sanitize_body(row.get("body", ""))
            subj = sanitize_subject(row.get("subject", ""))
            # Sanitising can empty a message that was pure boilerplate.
            if len(body) < 30 and len(subj) < 5:
                dropped += 1
                continue
            w.writerow({
                "sender":   (row.get("sender") or "").strip()[:500],
                "receiver": (row.get("receiver") or "").strip()[:300],
                "date":     "",                    # (1) removed entirely
                "subject":  subj[:300],
                "body":     body[:8000],
                "urls":     (row.get("urls") or "")[:2000],
                "label":    (row.get("label") or "").strip(),
            })
            kept += 1
    return kept, dropped


def main():
    print("=" * 74)
    print("  SANITISING — closing the provenance shortcut")
    print("=" * 74)
    total_k = total_d = 0
    for src_name, dst_name in IN_OUT:
        src, dst = DATA_DIR / src_name, DATA_DIR / dst_name
        k, d = process(src, dst)
        if k == 0 and d == 0:
            print(f"  · {src_name}: not found, skipped")
            continue
        print(f"  ✓ {src_name:<24} → {dst_name:<24} {k} kept, {d} emptied")
        total_k += k
        total_d += d
    print()
    print(f"  {total_k} emails retained, {total_d} dropped as pure boilerplate.")
    print()
    print("  Re-run the probe against the sanitised split to confirm the")
    print("  shortcut is closed:")
    print("    python experiments/leakage_probe.py --sanitized")
    print("=" * 74)


if __name__ == "__main__":
    main()
