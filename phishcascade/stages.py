"""
The three cascade stages.

  Stage 0  deterministic rule engine (PhishGuard). Zero LLM calls.
  Stage 1  one call to the small model. Accept if confident.
  Stage 2  four specialists + adjudicator on the large model. Five calls.

The routing decisions live here and are the substance of the contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import PHISHGUARD_DIR, AGENT_ROLES, ModelSpec, RoutingPolicy, RunConfig
from .agents import (
    AGENT_SYSTEMS, TRIAGE_SYSTEM,
    render_specialist_prompt, render_adjudicator_prompt,
)
from .llm import generate, parse_verdict, probability_of_phish
from .schema import AgentOpinion, LLMCall, StageResult, BENIGN, PHISH

# PhishGuard is imported, not reimplemented — the rule engine under test must
# be the same code that was evaluated standalone, or the comparison is void.
if str(PHISHGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(PHISHGUARD_DIR))

try:
    from phishguard.dataset import analyze_email  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        f"Cannot import PhishGuard from {PHISHGUARD_DIR}. "
        "The cascade's stage 0 is PhishGuard itself."
    ) from e


# ── Stage 0 ───────────────────────────────────────────────────────────────────
def stage0(record, policy: RoutingPolicy) -> StageResult:
    """Deterministic gate.

    Uses the two operating points measured in the standalone evaluation: a
    high-precision upper band that can auto-confirm, and a floor below which
    nothing fired at all. Everything between is escalated.

    The upper band is the interesting one. On the Nazario/SpamAssassin split
    a threshold of 41 gave 95.4% precision at 17.9% recall. As a standalone
    detector that recall is useless. As the first stage of a cascade it is
    free capacity: roughly a fifth of the phishing is resolved at zero
    inference cost, and the remaining budget concentrates on the residue.
    """
    a = analyze_email(record)
    score, signals = a.risk_score, list(a.signals)

    note = ""
    if a.flags:
        note = "\n".join(f"  - {f}" for f in a.flags)

    detail = {
        "risk_score": score,
        "risk_band": a.risk_band,
        "signals": signals,
        "flags": list(a.flags),
        "note": note,
        "iocs": {
            "urls": list(a.iocs.urls)[:20],
            "ipv4": list(a.iocs.ipv4)[:20],
            "emails": list(a.iocs.emails)[:20],
            "domains": list(getattr(a.iocs, "domains", []))[:20],
        },
    }

    if score >= policy.tau_high:
        return StageResult("stage0", PHISH, 0.95, True,
                           {**detail, "reason": "score >= tau_high"})
    if score < policy.tau_low:
        return StageResult("stage0", BENIGN, 0.9, True,
                           {**detail, "reason": "score < tau_low"})
    return StageResult("stage0", None, 0.0, False,
                       {**detail, "reason": "ambiguous -> escalate"})


# ── Stage 1 ───────────────────────────────────────────────────────────────────
def stage1(record, cfg: RunConfig, heuristic_note: str = "") -> StageResult:
    """Single pass on the small model.

    The confidence margin delta decides whether this verdict stands. A model
    that is badly calibrated will push everything to stage 2 and the cascade
    degenerates to always-on deliberation — which is exactly what the Pareto
    sweep will reveal if it happens, rather than hiding it.
    """
    prompt = render_specialist_prompt(record, heuristic_note, cfg.max_body_chars)
    text, call = generate(cfg.small, TRIAGE_SYSTEM, prompt, role="triage",
                          backend=cfg.backend, json_mode=True,
                          use_cache=cfg.cache)
    verdict, conf, rationale, evidence = parse_verdict(text)
    p = probability_of_phish(verdict, conf)
    margin = abs(p - 0.5)

    decided = margin >= cfg.policy.delta / 2.0 and verdict is not None
    return StageResult(
        "stage1", verdict, conf, decided,
        {"p_phish": p, "margin": margin, "rationale": rationale,
         "evidence": evidence,
         "reason": "confident" if decided else "uncertain -> deliberate"},
        [call],
    )


# ── Stage 2 ───────────────────────────────────────────────────────────────────
def stage2(record, cfg: RunConfig, heuristic_note: str = ""
           ) -> tuple[StageResult, list[AgentOpinion]]:
    """Four specialists in parallel domains, then an adjudicator.

    Specialists run against the large model because the whole point of
    reaching this stage is that the cheap model could not resolve the email.
    Running the panel on the small model would save budget but defeat the
    escalation.
    """
    calls: list[LLMCall] = []
    opinions: list[AgentOpinion] = []

    specialists = [r for r in AGENT_ROLES if r != "adjudicator"]
    prompt = render_specialist_prompt(record, heuristic_note, cfg.max_body_chars)

    for role in specialists:
        text, call = generate(cfg.large, AGENT_SYSTEMS[role], prompt, role=role,
                              backend=cfg.backend, json_mode=True,
                              use_cache=cfg.cache)
        calls.append(call)
        v, c, rat, ev = parse_verdict(text)
        opinions.append(AgentOpinion(role, v, c, rat, ev))

    adj_prompt = render_adjudicator_prompt(record, opinions, heuristic_note,
                                           max_body=cfg.max_body_chars // 2)
    text, call = generate(cfg.large, AGENT_SYSTEMS["adjudicator"], adj_prompt,
                          role="adjudicator", backend=cfg.backend,
                          json_mode=True, use_cache=cfg.cache)
    calls.append(call)
    v, c, rat, ev = parse_verdict(text)

    # If the adjudicator itself fails to answer, fall back to a majority of
    # non-abstaining specialists. Recorded explicitly so the fallback rate can
    # be reported rather than quietly absorbed into accuracy.
    fallback = False
    if v is None:
        votes = [o.verdict for o in opinions if o.verdict is not None]
        if votes:
            v = PHISH if sum(votes) * 2 > len(votes) else BENIGN
            c = abs(sum(votes) / len(votes) - 0.5) * 2
            fallback = True
        else:
            v, c, fallback = BENIGN, 0.0, True

    return (
        StageResult("stage2", v, c, True,
                    {"rationale": rat, "evidence": ev,
                     "adjudicator_fallback": fallback,
                     "specialist_votes": {o.role: o.verdict for o in opinions},
                     "reason": "deliberated"},
                    calls),
        opinions,
    )
