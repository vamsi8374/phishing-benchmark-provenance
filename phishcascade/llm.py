"""
LLM access layer: Ollama backend, deterministic offline mock, disk cache,
and per-call compute accounting.

Three things here matter for the paper.

1. ACCOUNTING. Every call records prompt/completion tokens and wall-clock
   latency. Cache hits are recorded but billed at zero, so the reported
   compute budget is the cost of the run as if it were the first run.

2. DETERMINISM. temperature=0 and a fixed seed. Re-running the experiment
   must reproduce the tables, or the numbers are not defensible.

3. THE MOCK. `PC_BACKEND=mock` gives a deterministic pseudo-model derived from
   a hash of the prompt. It is NOT a model of phishing and its accuracy is
   meaningless — it exists so the routing/plumbing can be exercised and
   unit-tested without occupying a GPU. Never report mock numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import CACHE_DIR, OLLAMA_HOST, REQUEST_TIMEOUT_S, ModelSpec
from .schema import LLMCall


class LLMError(RuntimeError):
    pass


# ── Cache ─────────────────────────────────────────────────────────────────────
def _cache_key(model: str, system: str, user: str, fmt: str) -> str:
    h = hashlib.sha256()
    for part in (model, system, user, fmt):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _cache_get(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_put(key: str, payload: dict) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


# ── Backends ──────────────────────────────────────────────────────────────────
def _ollama_generate(model: ModelSpec, system: str, user: str,
                     json_mode: bool) -> dict:
    body = {
        "model": model.name,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {
            "temperature": model.temperature,
            "num_ctx": model.ctx,
            "seed": 42,
        },
    }
    if json_mode:
        body["format"] = "json"

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LLMError(
            f"Cannot reach Ollama at {OLLAMA_HOST}: {e}\n"
            f"  Start it with:  ollama serve\n"
            f"  Pull the model: ollama pull {model.name}"
        ) from e

    return {
        "text": data.get("response", ""),
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "completion_tokens": int(data.get("eval_count", 0)),
    }


def _mock_generate(model: ModelSpec, system: str, user: str,
                   json_mode: bool) -> dict:
    """Deterministic stand-in. Accuracy is meaningless by construction."""
    h = int(hashlib.sha256((model.name + user).encode()).hexdigest()[:8], 16)
    p = (h % 1000) / 1000.0
    verdict = 1 if p >= 0.5 else 0
    payload = {
        "verdict": "phishing" if verdict else "benign",
        "confidence": round(abs(p - 0.5) * 2, 3),
        "rationale": "MOCK BACKEND — not a real assessment.",
        "evidence": [],
    }
    text = json.dumps(payload) if json_mode else payload["rationale"]
    # Simulate a plausible latency so budget plumbing is exercised.
    time.sleep(0.001)
    return {
        "text": text,
        "prompt_tokens": max(1, len(user) // 4),
        "completion_tokens": max(1, len(text) // 4),
    }


# ── Public API ────────────────────────────────────────────────────────────────
def generate(model: ModelSpec, system: str, user: str, role: str,
             backend: str = "ollama", json_mode: bool = True,
             use_cache: bool = True) -> tuple[str, LLMCall]:
    """Run one inference. Returns (text, LLMCall) — always both."""
    key = _cache_key(model.name, system, user, "json" if json_mode else "text")

    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit["text"], LLMCall(
                model=model.name, tier=model.tier, role=role,
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                latency_s=hit.get("latency_s", 0.0),
                cached=True,
            )

    fn = _mock_generate if backend == "mock" else _ollama_generate
    t0 = time.perf_counter()
    out = fn(model, system, user, json_mode)
    dt = time.perf_counter() - t0

    payload = {
        "text": out["text"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        "latency_s": dt,
    }
    if use_cache:
        _cache_put(key, payload)

    return out["text"], LLMCall(
        model=model.name, tier=model.tier, role=role,
        prompt_tokens=out["prompt_tokens"],
        completion_tokens=out["completion_tokens"],
        latency_s=dt, cached=False,
    )


# ── Robust JSON parsing ───────────────────────────────────────────────────────
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[int | None, float, str, list[str]]:
    """Extract (verdict, confidence, rationale, evidence) from model output.

    Small local models ignore format instructions more often than API models,
    so this degrades gracefully: strict JSON → embedded JSON → keyword scan →
    abstain. An abstention is recorded as None rather than silently guessed,
    because a guessed verdict would contaminate the accuracy numbers.
    """
    raw = (text or "").strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_BLOCK.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict):
        v = obj.get("verdict", obj.get("label", obj.get("classification")))
        verdict = None
        if isinstance(v, str):
            vl = v.strip().lower()
            if vl in ("phishing", "phish", "malicious", "1", "true", "yes"):
                verdict = 1
            elif vl in ("benign", "legitimate", "ham", "safe", "0", "false", "no"):
                verdict = 0
        elif isinstance(v, (int, float)):
            verdict = int(bool(v))
        elif isinstance(v, bool):
            verdict = int(v)

        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(1.0, max(0.0, conf))

        ev = obj.get("evidence", [])
        if isinstance(ev, str):
            ev = [ev]
        elif not isinstance(ev, list):
            ev = []

        return verdict, conf, str(obj.get("rationale", ""))[:800], [str(e)[:200] for e in ev[:6]]

    # Last resort: keyword scan over free text.
    low = raw.lower()
    if "phish" in low and "not phish" not in low and "benign" not in low:
        return 1, 0.5, raw[:400], []
    if "benign" in low or "legitimate" in low:
        return 0, 0.5, raw[:400], []
    return None, 0.0, raw[:400], []


def probability_of_phish(verdict: int | None, confidence: float) -> float:
    """Map (verdict, confidence) onto a single p(phishing) in [0,1].

    The router needs one scalar. An abstention maps to 0.5, i.e. maximally
    ambiguous, which routes it onward rather than resolving it arbitrarily.
    """
    if verdict is None:
        return 0.5
    return 0.5 + (confidence / 2.0) if verdict == 1 else 0.5 - (confidence / 2.0)


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def installed_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []
