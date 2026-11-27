"""
Configuration for the cost-aware cascade.

DESIGN NOTE — why local models
------------------------------
The motivating constraint is not price, it is data residency. A SOC handling
customer mailboxes generally cannot ship message bodies to a third-party API
(GDPR Art. 44 transfers, client confidentiality clauses, FedRAMP boundaries).
That forces on-premises inference, which in turn makes *compute* the binding
budget rather than dollars. Every cost figure in this project is therefore
reported in wall-clock seconds and generated tokens per 1,000 emails, which
are the quantities an on-prem operator actually has to provision for.

This is also what separates the work from MultiPhishGuard (CCS '25) and
PhishDebate, which assume cloud APIs and optimise accuracy alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT.parent / "phishguard_dataset_builder"
PHISHGUARD_DIR = PROJECT_ROOT.parent / "phishguard"

DEV_CSV   = DATA_DIR / "phishguard_dev.csv"
TEST_CSV  = DATA_DIR / "phishguard_test.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR   = PROJECT_ROOT / ".llm_cache"


# ── Model tiers ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    name: str            # ollama model tag, e.g. "qwen2.5:3b"
    tier: str            # "small" | "large"
    ctx: int = 8192
    temperature: float = 0.0     # deterministic — required for reproducibility


# Defaults chosen to run on a single consumer GPU / Apple Silicon box.
# Override with env vars so the same code runs on a bigger machine unchanged.
SMALL_MODEL = ModelSpec(
    name=os.environ.get("PC_SMALL_MODEL", "qwen2.5:3b"),
    tier="small",
)
LARGE_MODEL = ModelSpec(
    name=os.environ.get("PC_LARGE_MODEL", "qwen2.5:14b"),
    tier="large",
)

BACKEND = os.environ.get("PC_BACKEND", "ollama")   # ollama | mock
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Hard cap so one pathological email cannot stall a batch run.
REQUEST_TIMEOUT_S = int(os.environ.get("PC_TIMEOUT", "180"))


# ── Routing policy ────────────────────────────────────────────────────────────
@dataclass
class RoutingPolicy:
    """The object this paper is actually about.

    Three thresholds decide how much compute each email is worth:

      tau_low   Stage-0 score below this  → auto-BENIGN, zero LLM calls.
      tau_high  Stage-0 score at/above    → auto-PHISH,  zero LLM calls.
      delta     Stage-1 confidence margin. |p - 0.5| >= delta  → accept the
                small model's verdict. Otherwise escalate to Stage-2
                deliberation.

    Setting tau_low=0, tau_high=inf, delta=1.0 forces every email through the
    full 5-agent pipeline, which reproduces the always-on baseline that prior
    work evaluates. That degenerate setting is the comparison point.
    """
    tau_low: int = 1          # score 0 → nothing fired at all
    tau_high: int = 41        # measured: 95.4% precision on the Nazario split
    delta: float = 0.25       # accept stage-1 if p<=0.25 or p>=0.75

    def as_dict(self) -> dict:
        return {"tau_low": self.tau_low, "tau_high": self.tau_high,
                "delta": self.delta}

    @staticmethod
    def always_deliberate() -> "RoutingPolicy":
        """Baseline: every email gets the full agent panel (prior-work setting)."""
        return RoutingPolicy(tau_low=-1, tau_high=10**9, delta=1.01)

    @staticmethod
    def heuristics_only() -> "RoutingPolicy":
        """Baseline: no LLM at all. Stage-0 decides everything."""
        return RoutingPolicy(tau_low=10**9, tau_high=-1, delta=0.0)


DEFAULT_POLICY = RoutingPolicy()


# ── Agent roster ──────────────────────────────────────────────────────────────
# Kept deliberately close to the standard decomposition used by prior
# multi-agent phishing work, so that the comparison isolates the router
# rather than confounding it with a different agent design.
AGENT_ROLES: tuple[str, ...] = (
    "header_forensics",
    "url_infrastructure",
    "social_engineering",
    "brand_impersonation",
    "adjudicator",
)


@dataclass
class RunConfig:
    backend: str = BACKEND
    small: ModelSpec = SMALL_MODEL
    large: ModelSpec = LARGE_MODEL
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    cache: bool = True
    max_body_chars: int = 4000     # truncate for context budget + determinism
    seed: int = 42
