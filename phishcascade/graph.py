"""
Cascade orchestration.

Implemented twice on purpose:

  · LangGraph state machine, when langgraph is installed. This is what the
    paper describes and what you would deploy.
  · A plain sequential fallback with identical semantics, used when langgraph
    is unavailable.

Both paths are exercised by the test suite and must produce identical
CascadeResults for the same input. If they ever diverge, the LangGraph wiring
is wrong — an equivalence test is cheaper than debugging a state machine.
"""

from __future__ import annotations

from typing import Any, List, Optional, TypedDict

from .config import RunConfig
from .schema import CascadeResult, StageResult, AgentOpinion, BENIGN, PHISH
from .stages import stage0, stage1, stage2


def run_cascade(record, cfg: RunConfig, email_id: int = 0,
                label: int | None = None) -> CascadeResult:
    """Sequential reference implementation. Semantics of record here."""
    stages: list[StageResult] = []
    opinions: list[AgentOpinion] = []

    s0 = stage0(record, cfg.policy)
    stages.append(s0)
    note = s0.detail.get("note", "")

    if s0.decided:
        return _finish(email_id, label, s0.verdict, s0.confidence,
                       "stage0", stages, opinions, s0)

    s1 = stage1(record, cfg, note)
    stages.append(s1)
    if s1.decided:
        return _finish(email_id, label, s1.verdict, s1.confidence,
                       "stage1", stages, opinions, s0)

    s2, opinions = stage2(record, cfg, note)
    stages.append(s2)
    return _finish(email_id, label, s2.verdict, s2.confidence,
                   "stage2", stages, opinions, s0)


def _finish(email_id, label, verdict, confidence, exit_stage,
            stages, opinions, s0) -> CascadeResult:
    return CascadeResult(
        email_id=email_id,
        label=label,
        verdict=int(verdict) if verdict is not None else BENIGN,
        confidence=float(confidence),
        exit_stage=exit_stage,
        stages=stages,
        opinions=opinions,
        heuristic_score=s0.detail.get("risk_score", 0),
        heuristic_signals=s0.detail.get("signals", []),
        iocs=s0.detail.get("iocs", {}),
    )


# ── LangGraph variant ─────────────────────────────────────────────────────────
# Declared at module scope: LangGraph resolves the TypedDict's annotations
# against module globals, and `from __future__ import annotations` turns them
# into strings. A function-local TypedDict therefore fails to resolve.
class CascadeState(TypedDict, total=False):
    record: Any
    email_id: int
    label: Optional[int]
    stages: List[StageResult]
    opinions: List[AgentOpinion]
    note: str
    verdict: Optional[int]
    confidence: float
    exit_stage: str


def build_langgraph(cfg: RunConfig):
    """Return a compiled LangGraph app, or None if langgraph isn't installed."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None

    

    def n_stage0(s: CascadeState) -> CascadeState:
        r = stage0(s["record"], cfg.policy)
        return {**s, "stages": [r], "note": r.detail.get("note", ""),
                "verdict": r.verdict, "confidence": r.confidence,
                "exit_stage": "stage0"}

    def n_stage1(s: CascadeState) -> CascadeState:
        r = stage1(s["record"], cfg, s.get("note", ""))
        return {**s, "stages": s["stages"] + [r], "verdict": r.verdict,
                "confidence": r.confidence, "exit_stage": "stage1"}

    def n_stage2(s: CascadeState) -> CascadeState:
        r, ops = stage2(s["record"], cfg, s.get("note", ""))
        return {**s, "stages": s["stages"] + [r], "opinions": ops,
                "verdict": r.verdict, "confidence": r.confidence,
                "exit_stage": "stage2"}

    def route_after_0(s: CascadeState) -> str:
        return END if s["stages"][-1].decided else "stage1"

    def route_after_1(s: CascadeState) -> str:
        return END if s["stages"][-1].decided else "stage2"

    g = StateGraph(CascadeState)
    g.add_node("stage0", n_stage0)
    g.add_node("stage1", n_stage1)
    g.add_node("stage2", n_stage2)
    g.set_entry_point("stage0")
    g.add_conditional_edges("stage0", route_after_0, {"stage1": "stage1", END: END})
    g.add_conditional_edges("stage1", route_after_1, {"stage2": "stage2", END: END})
    g.add_edge("stage2", END)
    return g.compile()


def run_cascade_langgraph(app, record, cfg: RunConfig, email_id: int = 0,
                          label: int | None = None) -> CascadeResult:
    out = app.invoke({"record": record, "email_id": email_id, "label": label,
                      "stages": [], "opinions": []})
    stages = out["stages"]
    return _finish(email_id, label, out.get("verdict"), out.get("confidence", 0.0),
                   out.get("exit_stage", "stage0"), stages,
                   out.get("opinions", []), stages[0])
