"""PhishCascade — cost-aware cascaded multi-agent phishing triage.

The paper's experiments import only ``phishcascade.config``. The cascade
itself (``graph``/``stages``) needs PhishGuard's heuristic scorer, so it is
loaded lazily, on first use, rather than whenever this package is imported.
"""

__version__ = "0.1.0"

from .config import RunConfig, RoutingPolicy, SMALL_MODEL, LARGE_MODEL
from .schema import CascadeResult, StageResult, AgentOpinion, BENIGN, PHISH

_LAZY = {"run_cascade", "build_langgraph", "run_cascade_langgraph"}


def __getattr__(name):
    if name in _LAZY:
        from . import graph
        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RunConfig", "RoutingPolicy", "SMALL_MODEL", "LARGE_MODEL",
    "run_cascade", "build_langgraph", "run_cascade_langgraph",
    "CascadeResult", "StageResult", "AgentOpinion", "BENIGN", "PHISH",
]
