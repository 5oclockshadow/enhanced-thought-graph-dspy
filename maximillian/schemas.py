"""Pydantic v2 schemas for every Maximillian stage input/output.

These models are the typed contracts that flow through the 14-stage loop. They
are deliberately serialisable (``model_dump(mode="json")``) so the telemetry
ledger and HTML reporter can persist them verbatim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Stage(str, Enum):
    AUDIT_INIT = "01_audit_init"
    BUILD_NO_EXEC = "02_build_no_exec"
    READ_ALL = "03_read_all"
    REFACTOR_PRIMARY = "04_refactor_primary"
    AUDIT_POST_REFACTOR = "05_audit_post_refactor"
    REFACTOR_PRECISION = "06_refactor_precision"
    EXEC_SANDBOX = "07_exec_sandbox"
    TEST_TILL_PASS = "08_test_till_pass"
    INGEST_ALL_LOGS = "09_ingest_all_logs"
    TRIAGE_ROUND = "10_triage_round"
    AUDIT_AND_PLAN = "11_audit_and_plan"
    SYNC_LEDGERS = "12_sync_ledgers"
    PREFLIGHT_SNAPSHOT = "13_preflight_snapshot"
    REPORT_HTML = "14_report_html"


class AuditFinding(BaseModel):
    severity: Severity
    kind: str  # e.g. "missing_type_hint", "high_complexity", "circular_dependency"
    file: str
    symbol: Optional[str] = None
    line: Optional[int] = None
    message: str
    metric: Optional[float] = None


class AuditReport(BaseModel):
    target: str
    files_scanned: int = 0
    total_findings: int = 0
    findings: List[AuditFinding] = Field(default_factory=list)
    complexity_by_file: Dict[str, float] = Field(default_factory=dict)
    max_complexity: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)

    def by_severity(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts


class GraphSummary(BaseModel):
    nodes: int = 0
    edges: int = 0
    node_kinds: Dict[str, int] = Field(default_factory=dict)
    edge_kinds: Dict[str, int] = Field(default_factory=dict)
    top_central: List[Dict[str, Any]] = Field(default_factory=list)
    cycles: List[List[str]] = Field(default_factory=list)


class InvariantCheckResult(BaseModel):
    passed: bool = True
    violations: List[AuditFinding] = Field(default_factory=list)
    circular_dependencies: List[List[str]] = Field(default_factory=list)


class FailureNode(BaseModel):
    test_id: str
    file: Optional[str] = None
    line: Optional[int] = None
    exc_type: str = ""
    message: str = ""
    symbol: Optional[str] = None
    blast_radius: List[str] = Field(default_factory=list)


class TestIteration(BaseModel):
    attempt: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int = 1
    failures: List[FailureNode] = Field(default_factory=list)
    duration_s: float = 0.0


class TestReport(BaseModel):
    converged: bool = False
    iterations: List[TestIteration] = Field(default_factory=list)
    attempts: int = 0
    final_exit_code: int = 1


class ExecutionTrace(BaseModel):
    command: str
    exit_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0
    timed_out: bool = False


class SystemHealthSnapshot(BaseModel):
    files_tracked: int = 0
    total_loc: int = 0
    audit_findings: int = 0
    tests_passing: bool = False
    complexity_delta: float = 0.0


class TriageDecision(BaseModel):
    should_continue: bool
    roi_score: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""
    target: Optional[str] = None


class PlanStep(BaseModel):
    order: int
    stage: str
    action: str
    target: Optional[str] = None


class RoundExecutionPlan(BaseModel):
    round_id: int
    converged: bool = False
    steps: List[PlanStep] = Field(default_factory=list)


class StageTiming(BaseModel):
    stage: str
    duration_s: float
    ok: bool = True
    note: str = ""


class RewardBreakdown(BaseModel):
    test_pass_speed: float = 0.0
    diff_minimality: float = 0.0
    complexity_reduction: float = 0.0
    reward_score: float = Field(0.0, ge=0.0, le=1.0)


class RoundResult(BaseModel):
    round_id: int
    timestamp: datetime = Field(default_factory=_utcnow)
    target: str
    converged: bool = False
    audit: Optional[AuditReport] = None
    graph: Optional[GraphSummary] = None
    invariants: Optional[InvariantCheckResult] = None
    tests: Optional[TestReport] = None
    execution: Optional[ExecutionTrace] = None
    health: Optional[SystemHealthSnapshot] = None
    triage: Optional[TriageDecision] = None
    plan: Optional[RoundExecutionPlan] = None
    reward: Optional[RewardBreakdown] = None
    timeline: List[StageTiming] = Field(default_factory=list)
    git_sha: Optional[str] = None
    snapshot_tag: Optional[str] = None
    report_path: Optional[str] = None
    llm_available: bool = False
