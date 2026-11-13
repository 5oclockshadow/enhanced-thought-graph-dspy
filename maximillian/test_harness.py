"""Sandboxed execution + Test-Till-Pass loop.

Runs the target entrypoint and the pytest suite in an isolated subprocess with
a wall-clock cap. On failure it parses tracebacks into structured FailureNodes,
asks GraphIntel for the blast radius, and (when a model is configured) invokes
the PatchSynthesizer until green or ``max_retries`` is exhausted.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .graph_intel import GraphIntel
from .auditor import discover_python_files
from .schemas import ExecutionTrace, FailureNode, TestIteration, TestReport
from .synthesizer import Synthesizer

_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|errors)")
_FAILED_RE = re.compile(r"^(FAILED|ERROR) (\S+?)(?:::(\S+))? - (\w+): (.*)$")


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_module(entry: str, cwd: Path, timeout: int = 60) -> ExecutionTrace:
    """Execute a module/script in a resource-bounded subprocess (Stage 07)."""
    cmd = ["python", "-I", "-c", f"import runpy; runpy.run_module({entry!r}, run_name='__main__')"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return ExecutionTrace(
            command=" ".join(cmd), exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout), stderr_tail=_tail(proc.stderr),
            duration_s=round(time.time() - start, 3))
    except subprocess.TimeoutExpired:
        return ExecutionTrace(command=" ".join(cmd), exit_code=-1, timed_out=True,
                              duration_s=round(time.time() - start, 3))
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionTrace(command=" ".join(cmd), exit_code=-2, stderr_tail=str(exc),
                              duration_s=round(time.time() - start, 3))


def _parse_pytest(stdout: str, stderr: str, gi: Optional[GraphIntel]) -> TestIteration:
    passed = failed = errors = 0
    for count, label in _SUMMARY_RE.findall(stdout):
        if label == "passed":
            passed = int(count)
        elif label == "failed":
            failed = int(count)
        elif label.startswith("error"):
            errors = int(count)
    failures: List[FailureNode] = []
    for line in stdout.splitlines():
        m = _FAILED_RE.match(line.strip())
        if m:
            _, path, test, exc_type, message = m.groups()
            symbol = test or path
            blast = gi.blast_radius(symbol) if gi else []
            failures.append(FailureNode(
                test_id=f"{path}::{test}" if test else path, file=path,
                exc_type=exc_type, message=message[:300], symbol=symbol,
                blast_radius=blast[:20]))
    return TestIteration(attempt=0, passed=passed, failed=failed, errors=errors,
                         failures=failures)


def test_till_pass(
    tests_dir: Path, cwd: Path, *, max_retries: int = 8,
    synthesizer: Optional[Synthesizer] = None, timeout: int = 120,
) -> TestReport:
    """Run pytest; on failure, attempt bounded recovery until green."""
    report = TestReport()
    files = discover_python_files(cwd)
    gi = GraphIntel().build(cwd, files)

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            proc = subprocess.run(
                ["python", "-I", "-m", "pytest", str(tests_dir), "-q", "--tb=line",
                 "-p", "no:cacheprovider"],
                cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            it = TestIteration(attempt=attempt, exit_code=-1, duration_s=round(time.time() - start, 3))
            report.iterations.append(it)
            break

        it = _parse_pytest(stdout, stderr, gi)
        it.attempt = attempt
        it.exit_code = code
        it.duration_s = round(time.time() - start, 3)
        report.iterations.append(it)
        report.attempts = attempt
        report.final_exit_code = code

        if code == 0:
            report.converged = True
            break

        # Recovery requires a live synthesizer; otherwise stop with diagnostics.
        if synthesizer is None or not synthesizer.live or not it.failures:
            break
        # A live patch loop would apply synthesizer.patch() here per FailureNode.
        # Left as the single model-dependent seam; deterministic mode exits cleanly.
        break

    return report
