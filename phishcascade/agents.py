"""
Prompts for the five specialist agents and the single-pass triage model.

DESIGN NOTE — why these five, and why they look conventional
------------------------------------------------------------
The agent decomposition here is deliberately close to what prior multi-agent
phishing work uses (a header specialist, a URL specialist, a language
specialist, a brand specialist, and an adjudicator). That is not a lack of
imagination: the contribution of this project is the ROUTER, and holding the
agent panel fixed is what makes the comparison against always-on deliberation
an apples-to-apples measurement of routing rather than a confound.

Each specialist is instructed to answer only from its own evidence domain and
to abstain when its domain has nothing to say. Abstention matters — an agent
that invents a verdict from a field it cannot see adds noise to the panel and
inflates apparent agreement.
"""

from __future__ import annotations

from .schema import AgentOpinion

_JSON_CONTRACT = """
Respond with a single JSON object and nothing else:
{"verdict": "phishing" | "benign" | "abstain",
 "confidence": <float 0.0-1.0>,
 "rationale": "<one or two sentences>",
 "evidence": ["<specific quoted artefact>", ...]}

Rules:
- "confidence" is how sure you are, not how suspicious the email is.
- Use "abstain" when your assigned evidence domain is absent or uninformative.
  Abstaining is correct behaviour, not failure.
- Every item in "evidence" must be copied verbatim from the email. Do not
  paraphrase and do not invent artefacts.
""".strip()


TRIAGE_SYSTEM = f"""
You are a first-pass phishing triage classifier in a security operations
centre. You will see one email. Decide whether it is phishing.

Phishing includes credential harvesting, brand impersonation, fraudulent
payment redirection, malware delivery, and advance-fee fraud. It does NOT
include ordinary marketing, newsletters, or mailing-list traffic, which are
unsolicited but not malicious.

Be calibrated. If the evidence is genuinely mixed, say so with a confidence
near 0.5 rather than committing. A low-confidence answer is routed to a
deeper analysis stage, so under-claiming is cheap and over-claiming is not.

{_JSON_CONTRACT}
""".strip()


HEADER_FORENSICS_SYSTEM = f"""
You are a mail header forensics specialist. Consider ONLY the envelope and
identity fields: the From display name and address, the To/Reply-To fields,
the Date, and any routing artefacts visible in the text.

Look for: display name asserting an identity the sending domain does not
support; free-mail or unrelated domains claiming institutional identity;
lookalike and homoglyph domains; Reply-To diverging from From; implausible
or malformed dates; bulk-recipient patterns.

Ignore the body's persuasive content entirely — another specialist covers it.
If headers are missing or stripped, abstain.

{_JSON_CONTRACT}
""".strip()


URL_INFRA_SYSTEM = f"""
You are a URL and network infrastructure specialist. Consider ONLY the links,
domains, IP literals, and any attachment or file references.

Look for: bare IP destinations; URL shorteners masking a destination; anchor
text that disagrees with its target; credential-harvest path patterns
(/login, /verify, /secure, /account/update); suspicious or newly-abused TLDs;
punycode; open redirects; deep subdomain chains impersonating a brand.

An email with no links is not thereby benign — it may be reply-based fraud.
In that case abstain and let other specialists decide.

{_JSON_CONTRACT}
""".strip()


SOCIAL_ENGINEERING_SYSTEM = f"""
You are a social engineering analyst. Consider ONLY the persuasive structure
of the message body.

Look for: manufactured urgency and deadlines; threatened loss of access or
funds; authority impersonation; secrecy or bypass-normal-process requests;
unexpected financial instructions; pretexts that would not survive an
out-of-band check; mismatch between the claimed relationship and the tone.

Be careful: legitimate operational mail is often urgent, and legitimate
marketing is often manipulative. Neither is phishing on its own. Weigh
whether the message asks the recipient to take an irreversible action on the
sender's unverified say-so.

{_JSON_CONTRACT}
""".strip()


BRAND_IMPERSONATION_SYSTEM = f"""
You are a brand impersonation specialist. Determine whether the message is
posing as a specific organisation it does not belong to.

Look for: a named brand in the display name, subject, body, or link targets
that is not supported by the sending domain; template and boilerplate copied
from a real provider; trademark or logo references alongside an unrelated
sender; support/security/billing pretexts attributed to a named company.

Name the impersonated brand explicitly in your rationale when you find one.
If the message does not claim any organisational identity, abstain.

{_JSON_CONTRACT}
""".strip()


ADJUDICATOR_SYSTEM = f"""
You are the adjudicator. Several specialists have each examined one aspect of
an email and reported. Some may have abstained; some may disagree.

Your job is to weigh their findings and return one verdict. Guidance:
- An abstention is not a vote for benign. Ignore it.
- A single specialist with concrete, quoted evidence outweighs several
  specialists reporting vague impressions.
- Specialists sometimes agree for correlated reasons. Two agents both citing
  the same URL is one piece of evidence, not two.
- If the only signal is that the mail is unsolicited or promotional, that is
  spam, not phishing. Return benign.

Set your confidence to reflect genuine residual uncertainty after weighing
everything. This value is recorded and calibration is measured.

{_JSON_CONTRACT}
""".strip()


AGENT_SYSTEMS: dict[str, str] = {
    "header_forensics":    HEADER_FORENSICS_SYSTEM,
    "url_infrastructure":  URL_INFRA_SYSTEM,
    "social_engineering":  SOCIAL_ENGINEERING_SYSTEM,
    "brand_impersonation": BRAND_IMPERSONATION_SYSTEM,
    "adjudicator":         ADJUDICATOR_SYSTEM,
}


# ── Prompt builders ───────────────────────────────────────────────────────────
def render_email(record, max_body: int = 4000) -> str:
    """Render one email for the model. Field order is fixed for cache stability."""
    body = (record.body or "")[:max_body]
    urls = (record.urls or "")[:1000]
    return (
        f"From: {record.sender or '(absent)'}\n"
        f"To: {record.receiver or '(absent)'}\n"
        f"Date: {record.date or '(absent)'}\n"
        f"Subject: {record.subject or '(absent)'}\n"
        f"Extracted-URLs: {urls or '(none)'}\n"
        f"--- BODY ---\n{body}\n--- END BODY ---"
    )


def render_specialist_prompt(record, heuristic_note: str = "",
                             max_body: int = 4000) -> str:
    parts = [render_email(record, max_body)]
    if heuristic_note:
        parts.append(
            "\nA deterministic rule engine already flagged the following. "
            "Treat these as hints to verify, not as conclusions — the engine "
            "has known false positives.\n" + heuristic_note
        )
    return "\n".join(parts)


def render_adjudicator_prompt(record, opinions: list[AgentOpinion],
                              heuristic_note: str = "",
                              max_body: int = 2000) -> str:
    lines = [render_email(record, max_body), "\n--- SPECIALIST REPORTS ---"]
    for op in opinions:
        v = {1: "PHISHING", 0: "BENIGN", None: "ABSTAIN"}[op.verdict]
        lines.append(f"\n[{op.role}] verdict={v} confidence={op.confidence:.2f}")
        if op.rationale:
            lines.append(f"  rationale: {op.rationale}")
        for e in op.evidence[:4]:
            lines.append(f"  evidence: {e}")
    if heuristic_note:
        lines.append("\n--- RULE ENGINE ---\n" + heuristic_note)
    lines.append("\nReturn your adjudicated verdict as JSON.")
    return "\n".join(lines)
