"""Data structures carried through the cascade.

Everything the paper reports is derived from CascadeResult, so this module is
the single place to look when you want to know what was actually measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


BENIGN, PHISH = 0, 1


@dataclass
class LLMCall:
    """One inference. Cost accounting lives here and nowhere else."""
    model: str
    tier: str                 # small | large
    role: str                 # which agent/stage issued it
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billable_latency_s(self) -> float:
        """Cache hits cost nothing — excluded from the compute budget."""
        return 0.0 if self.cached else self.latency_s


@dataclass
class AgentOpinion:
    role: str
    verdict: int | None       # 0/1, or None if the agent abstained
    confidence: float         # 0..1, agent's own estimate
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    stage: str                # stage0 | stage1 | stage2
    verdict: int | None
    confidence: float
    decided: bool             # did the cascade stop here?
    detail: dict[str, Any] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)


@dataclass
class CascadeResult:
    email_id: int
    label: int | None                 # ground truth, 0/1, None if unlabelled
    verdict: int                      # final decision
    confidence: float
    exit_stage: str                   # where it stopped: stage0|stage1|stage2
    stages: list[StageResult] = field(default_factory=list)
    opinions: list[AgentOpinion] = field(default_factory=list)
    heuristic_score: int = 0
    heuristic_signals: list[str] = field(default_factory=list)
    iocs: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None

    # ── Cost accounting ──────────────────────────────────────────────────────
    @property
    def all_calls(self) -> list[LLMCall]:
        return [c for s in self.stages for c in s.calls]

    @property
    def n_llm_calls(self) -> int:
        return len(self.all_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.all_calls)

    @property
    def generated_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.all_calls)

    @property
    def compute_s(self) -> float:
        """Wall-clock inference seconds actually spent (excludes cache hits)."""
        return sum(c.billable_latency_s for c in self.all_calls)

    @property
    def large_model_calls(self) -> int:
        return sum(1 for c in self.all_calls if c.tier == "large")

    def to_json(self) -> dict:
        d = asdict(self)
        d["n_llm_calls"] = self.n_llm_calls
        d["total_tokens"] = self.total_tokens
        d["generated_tokens"] = self.generated_tokens
        d["compute_s"] = round(self.compute_s, 4)
        d["large_model_calls"] = self.large_model_calls
        return d


def dumps(results: list[CascadeResult]) -> str:
    return json.dumps([r.to_json() for r in results], indent=2)
