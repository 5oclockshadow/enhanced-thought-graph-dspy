"""LLM synthesis stages (Build / Refactor / Patch) as DSPy modules.

Every LLM call is a ``dspy.Module`` with explicit Input/Output fields (FR-02),
so prompts, completions and rewards can be logged to the optimizer ledger and
later compiled with MIPROv2 / BootstrapFewShot.

When no language model is configured (``configure_lm`` returns False — e.g. an
air-gapped sandbox with no API key), the synthesizer runs in **deterministic
no-op mode**: it returns the input unchanged with ``applied=False`` and a clear
reason, so the loop stays sound and never fabricates code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import dspy

logger = logging.getLogger("maximillian.synthesizer")


def configure_lm() -> bool:
    """Configure a DSPy LM from env; return True if a real model is wired.

    Honors MAXIMILLIAN_MODEL (e.g. "ollama/qwen2.5-coder:32b",
    "anthropic/claude-sonnet-5-5"). Returns False when unset or unreachable.
    """
    model = os.environ.get("MAXIMILLIAN_MODEL")
    if not model:
        return False
    try:
        lm = dspy.LM(model)
        dspy.settings.configure(lm=lm)
        return True
    except Exception as exc:  # pragma: no cover - depends on local env
        logger.warning("LM configuration failed (%s); deterministic mode.", exc)
        return False


class BuildSignature(dspy.Signature):
    """Synthesize static Python source for a module spec. Do NOT execute code."""

    spec = dspy.InputField(desc="Natural-language spec of the module to build")
    constraints = dspy.InputField(desc="Type, style and architectural constraints")
    source = dspy.OutputField(desc="Complete, importable Python source (no side effects)")


class RefactorSignature(dspy.Signature):
    """Refactor Python source for SOLID structure without changing behavior."""

    source = dspy.InputField(desc="Current Python source")
    findings = dspy.InputField(desc="Audit findings to resolve")
    refactored = dspy.OutputField(desc="Refactored Python source")


class PatchSignature(dspy.Signature):
    """Produce a minimal patch that fixes a failing test given its traceback."""

    source = dspy.InputField(desc="Source of the faulting symbol")
    traceback = dspy.InputField(desc="Pytest traceback and failing assertion")
    blast_radius = dspy.InputField(desc="Upstream dependents that must keep working")
    patched = dspy.OutputField(desc="Minimal corrected source")


@dataclass
class SynthResult:
    applied: bool
    output: str
    reason: str = ""
    tokens: int = 0


class Synthesizer:
    """Zero-exec Build + Refactor + Patch, LLM-backed when a model is present."""

    def __init__(self) -> None:
        self.live = configure_lm()
        if self.live:
            self._build = dspy.Predict(BuildSignature)
            self._refactor = dspy.Predict(RefactorSignature)
            self._patch = dspy.Predict(PatchSignature)

    def build(self, spec: str, constraints: str) -> SynthResult:
        if not self.live:
            return SynthResult(False, "", "no language model configured (deterministic mode)")
        try:
            out = self._build(spec=spec, constraints=constraints)
            return SynthResult(True, out.source)
        except Exception as exc:  # pragma: no cover - needs live model
            return SynthResult(False, "", f"build failed: {exc}")

    def refactor(self, source: str, findings: str) -> SynthResult:
        if not self.live:
            return SynthResult(False, source, "no language model configured (deterministic mode)")
        try:
            out = self._refactor(source=source, findings=findings)
            return SynthResult(True, out.refactored)
        except Exception as exc:  # pragma: no cover
            return SynthResult(False, source, f"refactor failed: {exc}")

    def patch(self, source: str, traceback: str, blast_radius: str) -> SynthResult:
        if not self.live:
            return SynthResult(False, source, "no language model configured (deterministic mode)")
        try:
            out = self._patch(source=source, traceback=traceback, blast_radius=blast_radius)
            return SynthResult(True, out.patched)
        except Exception as exc:  # pragma: no cover
            return SynthResult(False, source, f"patch failed: {exc}")
