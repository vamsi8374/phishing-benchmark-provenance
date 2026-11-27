"""
SIEM output: Elastic Common Schema (ECS) events that Wazuh ingests directly.

Why this is in the paper rather than an appendix: the deployment claim is that
a cascade can run on-premises inside an existing SOC pipeline. An artefact
that only prints a metric table does not support that claim. Emitting real ECS
documents, with the routing decision and per-email compute cost attached, is
what lets an operator verify the alert volume and budget numbers against their
own instance.

Wazuh ingestion:
    Write JSONL to a file the agent tails, e.g.
      /var/ossec/logs/alerts/phishcascade.json
    and add to ossec.conf:
      <localfile>
        <log_format>json</log_format>
        <location>/var/ossec/logs/alerts/phishcascade.json</location>
      </localfile>
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import CascadeResult, PHISH

# Wazuh rule levels: 0-15. Mapping chosen so that only adjudicated or
# rule-certain phishing reaches level 12+, which is the usual paging
# threshold. A cascade that pages on every stage-1 guess is worse than no
# cascade.
_LEVEL = {
    ("stage0", PHISH): 12,   # high-precision deterministic band
    ("stage1", PHISH): 10,
    ("stage2", PHISH): 13,   # full panel deliberated
}


def to_ecs(r: CascadeResult, record, rule_id_base: int = 100200) -> dict:
    """Render one CascadeResult as an ECS document."""
    now = datetime.now(timezone.utc).isoformat()
    is_phish = r.verdict == PHISH
    level = _LEVEL.get((r.exit_stage, r.verdict), 3)

    doc = {
        "@timestamp": now,
        "event": {
            "kind": "alert" if is_phish else "event",
            "category": ["email", "threat"],
            "type": ["info"],
            "action": "phishing-triage",
            "outcome": "success",
            "severity": level,
            "risk_score": r.heuristic_score,
            "reason": f"decided at {r.exit_stage}",
        },
        "rule": {
            "id": str(rule_id_base + (1 if is_phish else 0)),
            "name": "PhishCascade: phishing detected" if is_phish
                    else "PhishCascade: email cleared",
            "level": level,
            "description": (r.stages[-1].detail.get("rationale") or "")[:512],
        },
        "email": {
            "from": {"address": [_addr(record.sender)]},
            "to": {"address": [_addr(record.receiver)]} if record.receiver else {},
            "subject": (record.subject or "")[:300],
            "message_id": f"pc-{r.email_id}",
        },
        "threat": {
            "framework": "MITRE ATT&CK",
            "technique": {"id": ["T1566.002"], "name": ["Phishing: Spearphishing Link"]},
            "indicator": {
                "url": {"full": r.iocs.get("urls", [])[:20]},
                "ip": r.iocs.get("ipv4", [])[:20],
                "email": {"address": r.iocs.get("emails", [])[:20]},
            },
        },
        # Namespaced extension — everything the paper measures, per email.
        "phishcascade": {
            "verdict": "phishing" if is_phish else "benign",
            "confidence": round(r.confidence, 3),
            "exit_stage": r.exit_stage,
            "heuristic_score": r.heuristic_score,
            "heuristic_signals": r.heuristic_signals,
            "llm_calls": r.n_llm_calls,
            "large_model_calls": r.large_model_calls,
            "tokens_total": r.total_tokens,
            "tokens_generated": r.generated_tokens,
            "compute_seconds": round(r.compute_s, 4),
            "agent_votes": {o.role: o.verdict for o in r.opinions},
        },
    }
    return doc


def _addr(raw: str | None) -> str:
    s = (raw or "").strip()
    if "<" in s and ">" in s:
        return s[s.rfind("<") + 1: s.rfind(">")].strip().lower()
    return s.lower()


def write_jsonl(results: list[CascadeResult], records: list,
                path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r, rec in zip(results, records):
            fh.write(json.dumps(to_ecs(r, rec), ensure_ascii=False) + "\n")
    return p
