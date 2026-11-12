"""Cognitive telemetry for Graphos.

Implements the dual-format observability mandate:

1. A structured, timestamped (ISO 8601) telemetry JSON written twice per task
   run — a ``pre`` record (planning state) and a ``post`` record (validation
   state).
2. A human-readable Markdown execution log appended for every run.

Telemetry is written with resilient fault isolation: a telemetry failure must
never take down a generation request, so every sink is wrapped and errors are
swallowed after being logged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("graphos.telemetry")

# Resolve output roots relative to the repository, overridable via env so the
# container and tests can redirect them to a writable/ephemeral location.
_REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_DIR = Path(os.environ.get("GRAPHOS_TELEMETRY_DIR", _REPO_ROOT / "telemetry"))
EXECUTION_LOG = Path(os.environ.get("GRAPHOS_EXECUTION_LOG", _REPO_ROOT / "GRAPHOS_EXECUTION_LOG.md"))

_write_lock = threading.Lock()


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, limit: int = 40) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:limit] or "task"


@dataclass
class TelemetrySpan:
    """A single traced generation task, emitted pre- and post-execution.

    In production this is where Graphos wires spans to Arize Phoenix over a
    Redis-backed store (``graph-redis`` with raw-redis fallback). Here we emit
    the same structured records to disk so the contract is observable offline.
    """

    task: str
    parameters: Dict[str, Any]
    span_id: str = field(default_factory=lambda: _now_iso())
    started_at: str = field(default_factory=_now_iso)
    _paths: Dict[str, str] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------
    def pre(self, plan: Dict[str, Any]) -> None:
        """Emit the planning-state record before execution begins."""
        record = {
            "phase": "pre",
            "span_id": self.span_id,
            "task": self.task,
            "timestamp": self.started_at,
            "planned_parameters": self.parameters,
            "strategy": plan,
        }
        self._emit("pre", record)

    def post(
        self,
        *,
        success: bool,
        metrics: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """Emit the validation-state record after execution completes."""
        finished = _now_iso()
        record = {
            "phase": "post",
            "span_id": self.span_id,
            "task": self.task,
            "timestamp": finished,
            "started_at": self.started_at,
            "success": success,
            "metrics": metrics,
            "error": error,
        }
        self._emit("post", record)
        self._append_log(success=success, metrics=metrics, error=error, finished=finished)

    # -- sinks -------------------------------------------------------------
    def _emit(self, phase: str, record: Dict[str, Any]) -> None:
        try:
            with _write_lock:
                TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
                # Immutable naming: unique per phase + task + ISO timestamp.
                stamp = _now_iso().replace(":", "").replace("+00:00", "Z")
                path = TELEMETRY_DIR / f"telemetry_{phase}_{_slug(self.task)}_{stamp}.json"
                path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
                self._paths[phase] = str(path)
        except Exception as exc:  # pragma: no cover - telemetry must not crash task
            logger.warning("Telemetry %s sink failed: %s", phase, exc)

    def _append_log(self, *, success: bool, metrics: Dict[str, Any], error: Optional[str], finished: str) -> None:
        try:
            with _write_lock:
                new_file = not EXECUTION_LOG.exists()
                with EXECUTION_LOG.open("a", encoding="utf-8") as fh:
                    if new_file:
                        fh.write("# Graphos Execution Log\n\n")
                        fh.write("Human-readable trace of every graph-intelligence task.\n\n")
                    status = "✅ success" if success else "❌ failure"
                    fh.write(f"## {finished} — {self.task}\n\n")
                    fh.write(f"- **Status:** {status}\n")
                    fh.write(f"- **Span:** `{self.span_id}`\n")
                    if error:
                        fh.write(f"- **Error:** {error}\n")
                    for key, value in metrics.items():
                        fh.write(f"- **{key}:** {value}\n")
                    fh.write("\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("Execution-log sink failed: %s", exc)

    @property
    def paths(self) -> Dict[str, str]:
        return dict(self._paths)
