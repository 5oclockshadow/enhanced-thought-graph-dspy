"""LLM synthesis with environment auto-detection ("autopilot").

Autopilot turns itself ON whenever the environment already provides a model —
no flag required. Detection order (first that answers a ping wins):

1. ``MAXIMILLIAN_MODEL``            -> DSPy LM (explicit override)
2. standard provider env keys       -> DSPy LM (OpenAI / Anthropic / Ollama / …)
3. the ``claude`` CLI on PATH        -> session-authenticated CLI backend

Every backend is verified with a live ping before autopilot is declared on, so
the loop never claims a model it cannot actually call. When nothing answers,
the synthesis stages degrade to recorded no-ops and the rest of the loop still
runs deterministically.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("maximillian.synthesizer")

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_PING = "Reply with exactly one word: pong"


def _strip_code(text: str) -> str:
    m = _CODE_FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class Backend:
    name = "none"

    def complete(self, prompt: str, system: str = "", timeout: int = 120) -> str:
        raise NotImplementedError

    def ping(self) -> bool:
        try:
            return "pong" in self.complete(_PING, timeout=60).lower()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("%s ping failed: %s", self.name, exc)
            return False


class ClaudeCLIBackend(Backend):
    """Uses the session-authenticated `claude` CLI in non-interactive mode."""

    name = "claude-cli"

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def complete(self, prompt: str, system: str = "", timeout: int = 120) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        proc = subprocess.run(
            [self.binary, "-p", full, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[:200] or f"exit {proc.returncode}")
        return proc.stdout.strip()


class DSPyBackend(Backend):
    """Uses a DSPy-configured LM (litellm under the hood)."""

    def __init__(self, model: str) -> None:
        import dspy
        self.name = f"dspy:{model}"
        self._lm = dspy.LM(model)
        dspy.settings.configure(lm=self._lm)

    def complete(self, prompt: str, system: str = "", timeout: int = 120) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = self._lm(messages=messages)
        return out[0] if isinstance(out, list) else str(out)


def _env_model() -> Optional[str]:
    """Map standard provider env vars to a litellm model id."""
    if os.environ.get("MAXIMILLIAN_MODEL"):
        return os.environ["MAXIMILLIAN_MODEL"]
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "openai/" + os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/" + os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    if os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST"):
        return "ollama/" + os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    if os.environ.get("GROQ_API_KEY"):
        return "groq/" + os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return None


def detect_backend() -> Optional[Backend]:
    """Return the first environment-provided model that answers a ping."""
    if os.environ.get("MAXIMILLIAN_NO_AUTOPILOT"):
        return None
    # 1 & 2: a DSPy-addressable model id from env.
    model = _env_model()
    if model:
        try:
            b = DSPyBackend(model)
            if b.ping():
                logger.info("Autopilot backend: %s", b.name)
                return b
        except Exception as exc:  # pragma: no cover
            logger.warning("DSPy backend %s unavailable: %s", model, exc)
    # 3: the session's own claude CLI.
    cli = shutil.which("claude")
    if cli:
        b = ClaudeCLIBackend(cli)
        if b.ping():
            logger.info("Autopilot backend: %s (%s)", b.name, cli)
            return b
    logger.info("No model detected in environment — deterministic mode.")
    return None


def configure_lm() -> bool:
    """Back-compat shim: True when any backend is detected."""
    return detect_backend() is not None


# --------------------------------------------------------------------------
# Synthesizer
# --------------------------------------------------------------------------
@dataclass
class SynthResult:
    applied: bool
    output: str
    reason: str = ""
    backend: str = "none"


_BUILD_SYS = ("You are a principal Python engineer. Output ONLY a single complete, "
              "importable Python module in one ```python code block. Strict type hints, "
              "Pydantic v2, docstrings, no side effects at import, no TODOs.")
_REFACTOR_SYS = ("You are a refactoring engine. Given findings, return concrete, minimal "
                 "refactor recommendations as a short numbered list (no code blocks). "
                 "Each item: the symbol, the fix, and why. Be specific and terse.")
_PATCH_SYS = ("You are a bug-fix engine. Return ONLY the corrected source for the given "
              "file in one ```python code block. Make the minimal change that fixes the "
              "failing test without breaking the listed dependents.")


class Synthesizer:
    def __init__(self) -> None:
        self.backend = detect_backend()

    @property
    def live(self) -> bool:
        return self.backend is not None

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "none"

    def build(self, spec: str, constraints: str) -> SynthResult:
        if not self.backend:
            return SynthResult(False, "", "no model in environment (deterministic mode)")
        try:
            out = self.backend.complete(f"Spec:\n{spec}\n\nConstraints: {constraints}", _BUILD_SYS)
            return SynthResult(True, _strip_code(out), backend=self.backend.name)
        except Exception as exc:
            return SynthResult(False, "", f"build failed: {exc}", self.backend.name)

    def refactor(self, source_or_findings: str) -> SynthResult:
        if not self.backend:
            return SynthResult(False, "", "no model in environment (deterministic mode)")
        try:
            out = self.backend.complete(f"Findings:\n{source_or_findings}", _REFACTOR_SYS)
            return SynthResult(True, out.strip(), backend=self.backend.name)
        except Exception as exc:
            return SynthResult(False, "", f"refactor failed: {exc}", self.backend.name)

    def patch(self, source: str, traceback: str, blast_radius: str) -> SynthResult:
        if not self.backend:
            return SynthResult(False, source, "no model in environment (deterministic mode)")
        try:
            prompt = f"FILE SOURCE:\n```python\n{source}\n```\n\nTRACEBACK:\n{traceback}\n\nDEPENDENTS: {blast_radius}"
            out = self.backend.complete(prompt, _PATCH_SYS)
            patched = _strip_code(out)
            return SynthResult(bool(patched), patched or source, backend=self.backend.name)
        except Exception as exc:
            return SynthResult(False, source, f"patch failed: {exc}", self.backend.name)

    def advise(self, system: str, prompt: str) -> SynthResult:
        if not self.backend:
            return SynthResult(False, "", "no model in environment (deterministic mode)")
        try:
            return SynthResult(True, self.backend.complete(prompt, system).strip(),
                               backend=self.backend.name)
        except Exception as exc:
            return SynthResult(False, "", f"advise failed: {exc}", self.backend.name)
