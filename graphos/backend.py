"""Single port onto the environment's language model.

Architect finding R: graphos reached into `maximillian.synthesizer.detect_backend`
from five scattered sites with divergent caching (only live_engine memoised).
This is the one legitimate cross-package dependency — give it one door, cached
once, so detection cost + log noise happen a single time.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("graphos.backend")

_backend = None
_detected = False
_lock = threading.Lock()


class _DspyBackend:
    """A dspy.LM-backed backend with NATIVE async via dspy's ``acall``.

    Activated by ``GRAPHOS_DSPY_MODEL`` (e.g. 'anthropic/claude-3-5-sonnet-latest',
    'openai/gpt-4o-mini', 'ollama/qwen2.5-coder'). Concurrency is bounded by
    ``async_max_workers`` (GRAPHOS_DSPY_ASYNC_WORKERS, default 8)."""

    def __init__(self, model: str, lm) -> None:
        self.name = f"dspy:{model}"
        self._lm = lm

    @staticmethod
    def _text(resp) -> str:
        # dspy 3.5 returns an lm15 Response with `.text`; be tolerant of older forms.
        t = getattr(resp, "text", None)
        if t is not None:
            return t
        return resp[0] if isinstance(resp, list) else str(resp)

    def complete(self, prompt: str, system: str = "", timeout: int = 150) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self._text(self._lm(full))  # dspy 3.5: call with a prompt string

    async def acomplete(self, prompt: str, system: str = "", timeout: int = 150) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self._text(await self._lm.acall(full))  # native async

    def ping(self) -> bool:
        try:
            return "pong" in self.complete("Reply with one word: pong").lower()
        except Exception:  # pragma: no cover - env dependent
            return False


def _dspy_backend():
    model = os.environ.get("GRAPHOS_DSPY_MODEL")
    if not model:
        return None
    try:
        import dspy
        workers = int(os.environ.get("GRAPHOS_DSPY_ASYNC_WORKERS", "8") or 8)
        lm = dspy.LM(model)
        dspy.configure(lm=lm, async_max_workers=workers)
        return _DspyBackend(model, lm)
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("dspy backend %s unavailable: %s", model, exc)
        return None


def get_backend():
    """Return the cached model backend, or None. Detection runs at most once.

    Order: a dspy.LM (native async) when GRAPHOS_DSPY_MODEL is set, else the
    environment-detected backend (the session `claude` CLI, provider keys)."""
    global _backend, _detected
    if _detected:
        return _backend
    with _lock:
        if _detected:
            return _backend
        _detected = True
        if os.environ.get("GRAPHOS_DISABLE_LIVE"):
            _backend = None
            return None
        dsb = _dspy_backend()
        if dsb is not None and dsb.ping():
            logger.info("backend: %s (native async)", dsb.name)
            _backend = dsb
            return _backend
        try:
            from maximillian.synthesizer import detect_backend
            _backend = detect_backend()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("backend detection failed: %s", exc)
            _backend = None
        return _backend


async def acomplete(prompt: str, system: str = "", timeout: int = 150) -> Optional[str]:
    """Async completion. Uses the backend's native ``acall`` when it has one
    (dspy.LM); otherwise offloads the sync call to a worker thread so the event
    loop is never blocked."""
    b = get_backend()
    if b is None:
        return None
    acall = getattr(b, "acomplete", None)
    if acall is not None:
        return await acall(prompt, system, timeout=timeout)
    import anyio
    return await anyio.to_thread.run_sync(lambda: b.complete(prompt, system, timeout=timeout))


def is_live() -> bool:
    return get_backend() is not None


def backend_name() -> str:
    b = get_backend()
    return b.name if b else "none"


def complete(prompt: str, system: str = "", timeout: int = 150) -> Optional[str]:
    """Run one completion through the cached backend, or None if unavailable."""
    b = get_backend()
    if b is None:
        return None
    return b.complete(prompt, system, timeout=timeout)


def reset_cache() -> None:
    """Testing hook: forget the detected backend."""
    global _backend, _detected
    with _lock:
        _backend, _detected = None, False
