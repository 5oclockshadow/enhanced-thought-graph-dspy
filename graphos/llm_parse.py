"""Shared helpers for pulling structured text out of a model reply.

Craftsman finding #3: `extract_json` / code-fence stripping was reimplemented
four times (live_engine, foundry_api, self_improve, maximillian.synthesizer),
each with its own failure handling. One source of truth now.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-z]*\n|\n```$", re.MULTILINE)


def strip_fence(text: str) -> str:
    """Remove a leading/trailing ```lang fence if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t).strip()
    return t


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object in the text, or None."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
