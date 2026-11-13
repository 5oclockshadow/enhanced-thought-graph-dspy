"""Static auditor: deterministic analysis that runs before any LLM synthesis.

Computes cyclomatic complexity, flags missing type annotations and risky
constructs, and promotes circular dependencies (from GraphIntel) into findings.
All air-gapped: pure ``ast`` plus optional ``ruff`` if installed.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import List

from .graph_intel import GraphIntel
from .schemas import AuditFinding, AuditReport, InvariantCheckResult, Severity

_COMPLEXITY_NODES = (
    ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler,
    ast.With, ast.Assert, ast.BoolOp, ast.comprehension,
)


def discover_python_files(target: Path) -> List[Path]:
    """Return .py files under target, skipping caches and virtualenvs."""
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build"}
    out: List[Path] = []
    for p in sorted(target.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        out.append(p)
    return out


def _cyclomatic_complexity(func: ast.AST) -> int:
    score = 1
    for node in ast.walk(func):
        if isinstance(node, _COMPLEXITY_NODES):
            score += 1
    return score


class Auditor:
    def __init__(self, use_ruff: bool = True) -> None:
        self.use_ruff = use_ruff and shutil.which("ruff") is not None

    def audit(self, target: Path) -> AuditReport:
        files = discover_python_files(target)
        report = AuditReport(target=str(target), files_scanned=len(files))

        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError) as exc:
                report.findings.append(AuditFinding(
                    severity=Severity.CRITICAL, kind="syntax_error",
                    file=str(path), message=f"Cannot parse: {exc}"))
                continue
            self._audit_tree(path, tree, report)

        # Circular dependencies via the dependency graph.
        gi = GraphIntel().build(target, files)
        for cycle in gi.cycles():
            report.findings.append(AuditFinding(
                severity=Severity.HIGH, kind="circular_dependency",
                file=str(target), symbol=" -> ".join(cycle),
                message=f"Circular dependency: {' -> '.join(cycle)}"))

        if self.use_ruff:
            report.findings.extend(self._ruff(target))

        report.total_findings = len(report.findings)
        if report.complexity_by_file:
            report.max_complexity = max(report.complexity_by_file.values())
        return report

    def _audit_tree(self, path: Path, tree: ast.AST, report: AuditReport) -> None:
        file_complexity = 0.0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _cyclomatic_complexity(node)
                file_complexity = max(file_complexity, cc)
                if cc >= 10:
                    report.findings.append(AuditFinding(
                        severity=Severity.HIGH if cc >= 15 else Severity.MEDIUM,
                        kind="high_complexity", file=str(path), symbol=node.name,
                        line=node.lineno, metric=float(cc),
                        message=f"{node.name} has cyclomatic complexity {cc}"))
                if node.returns is None and not node.name.startswith("_") and node.name != "__init__":
                    report.findings.append(AuditFinding(
                        severity=Severity.LOW, kind="missing_return_type",
                        file=str(path), symbol=node.name, line=node.lineno,
                        message=f"{node.name} lacks a return type annotation"))
            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                report.findings.append(AuditFinding(
                    severity=Severity.MEDIUM, kind="bare_except", file=str(path),
                    line=node.lineno, message="bare 'except:' swallows all errors"))
        report.complexity_by_file[str(path)] = file_complexity

    def _ruff(self, target: Path) -> List[AuditFinding]:
        try:
            proc = subprocess.run(
                ["ruff", "check", "--output-format", "concise", str(target)],
                capture_output=True, text=True, timeout=60)
        except Exception:
            return []
        findings: List[AuditFinding] = []
        for line in proc.stdout.splitlines():
            # format: path:line:col: CODE message
            parts = line.split(":", 3)
            if len(parts) >= 4:
                findings.append(AuditFinding(
                    severity=Severity.LOW, kind="lint", file=parts[0],
                    line=int(parts[1]) if parts[1].isdigit() else None,
                    message=parts[3].strip()))
        return findings


def check_invariants(target: Path) -> InvariantCheckResult:
    """Secondary audit: architectural invariants after a refactor."""
    files = discover_python_files(target)
    gi = GraphIntel().build(target, files)
    cycles = gi.cycles()
    violations: List[AuditFinding] = []
    for cycle in cycles:
        violations.append(AuditFinding(
            severity=Severity.HIGH, kind="circular_dependency", file=str(target),
            symbol=" -> ".join(cycle), message="invariant: no circular imports"))
    return InvariantCheckResult(
        passed=not violations, violations=violations, circular_dependencies=cycles)
