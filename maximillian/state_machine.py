"""The Maximillian loop: a 14-stage DAG orchestrator with bounded rounds.

Deterministic stages always run; the three LLM-synthesis stages degrade to
no-ops with a recorded reason when no model is configured. Each stage is timed
into the round's telemetry, and every round emits a self-contained HTML report.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .auditor import Auditor, check_invariants, discover_python_files
from .graph_intel import GraphIntel
from .reporter import render_report
from .schemas import (
    PlanStep, RoundExecutionPlan, RoundResult, StageTiming, Stage,
    SystemHealthSnapshot, TriageDecision,
)
from .snapshotter import git_commit, git_sha, preflight, snapshot_tag
from .synthesizer import Synthesizer
from .telemetry import LedgerWriter, compute_reward
from .test_harness import run_module, test_till_pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("maximillian")


@dataclass
class Config:
    target: Path
    tests: Optional[Path] = None
    repo: Optional[Path] = None
    tracking_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None
    entry_module: Optional[str] = None
    max_rounds: int = 1
    max_test_retries: int = 8
    enable_git: bool = False


class StateMachine:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.repo = config.repo or Path.cwd()
        self.tracking = config.tracking_dir or (self.repo / "tracking")
        self.reports = config.reports_dir or (self.repo / "reports")
        self.synth = Synthesizer()
        self.ledger = LedgerWriter(self.tracking)
        self._prev_complexity: Optional[float] = None

    # -- stage timing helper ----------------------------------------------
    def _timed(self, result: RoundResult, stage: Stage, fn, note: str = ""):
        start = time.time()
        ok = True
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive per-stage isolation
            ok = False
            note = f"error: {exc}"
            logger.exception("Stage %s failed", stage.value)
            return None
        finally:
            result.timeline.append(StageTiming(
                stage=stage.value, duration_s=round(time.time() - start, 4), ok=ok, note=note))

    # -- one round ---------------------------------------------------------
    def run_round(self, round_id: int) -> RoundResult:
        cfg = self.cfg
        result = RoundResult(round_id=round_id, target=str(cfg.target),
                             llm_available=self.synth.live, backend=self.synth.backend_name)
        logger.info("── Round %d · target=%s · autopilot=%s (%s)",
                    round_id, cfg.target, self.synth.live, self.synth.backend_name)

        # 01 Initial audit
        result.audit = self._timed(result, Stage.AUDIT_INIT, lambda: Auditor().audit(cfg.target))
        # 02 Build (no-exec) — only synthesizes when a build spec is supplied.
        self._timed(result, Stage.BUILD_NO_EXEC, lambda: None, note="no build spec (audit run)")
        # 03 Full-context read -> graph
        files = discover_python_files(cfg.target)
        result.graph = self._timed(result, Stage.READ_ALL,
                                   lambda: GraphIntel().build(cfg.target, files).summary())
        # 04 Structural refactor (autopilot: real model proposals on top findings)
        self._timed(result, Stage.REFACTOR_PRIMARY, lambda: self._refactor_stage(result),
                    note="" if self.synth.live else "deterministic no-op")
        # 05 Secondary audit — invariants
        result.invariants = self._timed(result, Stage.AUDIT_POST_REFACTOR,
                                        lambda: check_invariants(cfg.target))
        # 06 Precision refactor — resolution recorded via Stage 04 proposals.
        self._timed(result, Stage.REFACTOR_PRECISION, lambda: None,
                    note=f"{len(result.refactor_proposals)} proposals from stage 04"
                    if self.synth.live else "deterministic no-op")
        # 07 Sandboxed execution (optional entry)
        if cfg.entry_module:
            result.execution = self._timed(result, Stage.EXEC_SANDBOX,
                                          lambda: run_module(cfg.entry_module, self.repo))
        else:
            self._timed(result, Stage.EXEC_SANDBOX, lambda: None, note="no entry module")
        # 08 Test-till-pass
        if cfg.tests and cfg.tests.exists():
            result.tests = self._timed(result, Stage.TEST_TILL_PASS,
                                      lambda: test_till_pass(cfg.tests, self.repo,
                                                             max_retries=cfg.max_test_retries,
                                                             synthesizer=self.synth))
        else:
            self._timed(result, Stage.TEST_TILL_PASS, lambda: None, note="no tests dir")

        # 09 Ingest logs/files -> health
        result.health = self._timed(result, Stage.INGEST_ALL_LOGS, lambda: self._health(result))
        # 10 Triage
        result.triage = self._timed(result, Stage.TRIAGE_ROUND,
                                   lambda: self._triage(result, round_id))
        # 11 Plan
        result.plan = self._timed(result, Stage.AUDIT_AND_PLAN,
                                 lambda: self._plan(result, round_id))
        result.converged = not (result.triage and result.triage.should_continue)

        # 12 Sync ledgers (+ optional git)
        self._timed(result, Stage.SYNC_LEDGERS, lambda: self._sync(result))
        # 13 Preflight & snapshot
        self._timed(result, Stage.PREFLIGHT_SNAPSHOT, lambda: self._snapshot(result))
        # 14 HTML report
        result.report_path = self._timed(result, Stage.REPORT_HTML,
                                        lambda: str(render_report(result, self.reports)))
        logger.info("Round %d complete · converged=%s · report=%s",
                    round_id, result.converged, result.report_path)
        return result

    # -- stage implementations --------------------------------------------
    def _refactor_stage(self, result: RoundResult) -> None:
        """Autopilot Stage 04: ask the live model for concrete refactor proposals
        targeting the highest-severity findings. Advisory only — never edits
        source unless the test-till-pass patch loop is explicitly enabled."""
        if not self.synth.live or not result.audit or not result.audit.findings:
            return
        ranked = sorted(result.audit.findings,
                        key=lambda f: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(f.severity.value),
                        reverse=True)[:5]
        summary = "\n".join(
            f"- [{f.severity.value}] {f.kind} in {Path(f.file).name}"
            f"{(':' + f.symbol) if f.symbol else ''} — {f.message}" for f in ranked)
        res = self.synth.refactor(summary)
        if res.applied and res.output:
            lines = [ln.strip(" -•\t") for ln in res.output.splitlines() if ln.strip()]
            result.refactor_proposals = [ln for ln in lines if len(ln) > 8][:8]

    def _health(self, result: RoundResult) -> SystemHealthSnapshot:
        files = discover_python_files(self.cfg.target)
        loc = 0
        for p in files:
            try:
                loc += p.read_text(encoding="utf-8").count("\n") + 1
            except Exception:
                pass
        cur = result.audit.max_complexity if result.audit else 0.0
        delta = 0.0 if self._prev_complexity is None else cur - self._prev_complexity
        self._prev_complexity = cur
        return SystemHealthSnapshot(
            files_tracked=len(files), total_loc=loc,
            audit_findings=result.audit.total_findings if result.audit else 0,
            tests_passing=bool(result.tests and result.tests.converged),
            complexity_delta=round(delta, 3))

    def _triage(self, result: RoundResult, round_id: int) -> TriageDecision:
        findings = result.audit.total_findings if result.audit else 0
        crit = 0
        if result.audit:
            sev = result.audit.by_severity()
            crit = sev.get("CRITICAL", 0) + sev.get("HIGH", 0)
        tests_green = bool(result.tests and result.tests.converged) or not self.cfg.tests
        # ROI rises with unresolved high-severity findings.
        roi = min(1.0, crit / 10.0 + (0.2 if not tests_green else 0.0))
        more_rounds = round_id < self.cfg.max_rounds
        should = more_rounds and (crit > 0 or not tests_green)
        rationale = (f"{crit} high/critical findings, tests_green={tests_green}, "
                     f"round {round_id}/{self.cfg.max_rounds}")
        return TriageDecision(should_continue=should, roi_score=round(roi, 3),
                              rationale=rationale,
                              target=str(self.cfg.target) if should else None)

    def _plan(self, result: RoundResult, round_id: int) -> RoundExecutionPlan:
        steps: List[PlanStep] = []
        if result.triage and result.triage.should_continue and result.audit:
            top = sorted(result.audit.findings,
                         key=lambda f: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(f.severity.value),
                         reverse=True)[:3]
            for i, f in enumerate(top, 1):
                steps.append(PlanStep(order=i, stage=Stage.REFACTOR_PRECISION.value,
                                      action=f"resolve {f.kind}: {f.message}",
                                      target=f.symbol or f.file))
        # Autopilot: a concise model-authored strategic narrative for next round.
        if self.synth.live and result.audit:
            sev = result.audit.by_severity()
            ctx = (f"Target {Path(self.cfg.target).name}: {result.audit.total_findings} findings "
                   f"({sev}), {result.graph.nodes if result.graph else 0} graph nodes, "
                   f"tests_passing={bool(result.tests and result.tests.converged)}.")
            adv = self.synth.advise(
                "You are a staff engineer. In 3-4 terse sentences, state the single "
                "highest-leverage improvement for the next round and why.", ctx)
            if adv.applied:
                result.strategic_plan_text = adv.output[:1200]
        return RoundExecutionPlan(round_id=round_id + 1,
                                  converged=result.converged, steps=steps)

    def _sync(self, result: RoundResult) -> None:
        delta = result.health.complexity_delta if result.health else 0.0
        result.reward = compute_reward(
            test_attempts=result.tests.attempts if result.tests else 1,
            converged=bool(result.tests and result.tests.converged) or not self.cfg.tests,
            complexity_delta=delta)
        if self.cfg.enable_git:
            result.git_sha = git_commit(self.repo, f"chore(maximillian): round {result.round_id}")
        else:
            result.git_sha = git_sha(self.repo)
        self.ledger.append(result)

    def _snapshot(self, result: RoundResult) -> None:
        checks = preflight(self.repo, import_names=["maximillian"])
        if self.cfg.enable_git and result.converged:
            result.snapshot_tag = snapshot_tag(self.repo, result.round_id)
        logger.info("Preflight: %s", checks)

    # -- multi-round loop --------------------------------------------------
    def run(self) -> List[RoundResult]:
        results: List[RoundResult] = []
        for r in range(1, self.cfg.max_rounds + 1):
            res = self.run_round(r)
            results.append(res)
            if res.converged:
                break
        return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="maximillian", description="Autonomous 14-stage engineering loop")
    ap.add_argument("--target", required=True, help="directory to audit/build")
    ap.add_argument("--tests", help="pytest directory")
    ap.add_argument("--repo", help="repo root (default: cwd)")
    ap.add_argument("--entry-module", help="module to execute in Stage 07")
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--max-test-retries", type=int, default=8)
    ap.add_argument("--enable-git", action="store_true", help="opt in to commits/tags")
    args = ap.parse_args(argv)

    cfg = Config(
        target=Path(args.target).resolve(),
        tests=Path(args.tests).resolve() if args.tests else None,
        repo=Path(args.repo).resolve() if args.repo else Path.cwd(),
        entry_module=args.entry_module,
        max_rounds=args.max_rounds,
        max_test_retries=args.max_test_retries,
        enable_git=args.enable_git,
    )
    results = StateMachine(cfg).run()
    last = results[-1]
    print(f"\n✅ {len(results)} round(s) · converged={last.converged}")
    print(f"   report: {last.report_path}")
    print(f"   ledger: {cfg.repo / 'tracking'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
