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


def get_backend():
    """Return the cached model backend, or None. Detection runs at most once."""
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
        try:
            from maximillian.synthesizer import detect_backend
            _backend = detect_backend()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("backend detection failed: %s", exc)
            _backend = None
        return _backend


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
