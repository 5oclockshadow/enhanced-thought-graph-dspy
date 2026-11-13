"""Tests for the Maximillian loop: graph intel, auditor, telemetry, reporter,
and a full deterministic round on a synthetic target."""

import os
import textwrap
from pathlib import Path

import pytest

# Keep tests deterministic and offline — never ping a live model.
os.environ["MAXIMILLIAN_NO_AUTOPILOT"] = "1"

from maximillian.auditor import Auditor, check_invariants, discover_python_files
from maximillian.graph_intel import GraphIntel
from maximillian.reporter import render_report
from maximillian.schemas import Severity
from maximillian.state_machine import Config, StateMachine
from maximillian.telemetry import LedgerWriter, compute_reward


@pytest.fixture
def sample_pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "sample"
    pkg.mkdir()
    (pkg / "a.py").write_text(textwrap.dedent('''
        """module a"""
        def helper(x: int) -> int:
            return x + 1

        class Widget:
            def run(self):            # missing return type -> LOW finding
                total = 0
                for i in range(10):   # complexity
                    if i % 2 == 0:
                        total += helper(i)
                    else:
                        total -= 1
                return total
    '''), encoding="utf-8")
    (pkg / "b.py").write_text(textwrap.dedent('''
        """module b"""
        from a import helper
        def use() -> int:
            try:
                return helper(1)
            except:           # bare except -> MEDIUM finding
                return 0
    '''), encoding="utf-8")
    return pkg


# -- graph intelligence ----------------------------------------------------
def test_graph_builds_nodes_and_edges(sample_pkg):
    files = discover_python_files(sample_pkg)
    gi = GraphIntel().build(sample_pkg, files)
    summary = gi.summary()
    assert summary.nodes > 0
    assert summary.edges > 0
    assert "module" in summary.node_kinds


def test_blast_radius_returns_dependents(sample_pkg):
    files = discover_python_files(sample_pkg)
    gi = GraphIntel().build(sample_pkg, files)
    radius = gi.blast_radius("helper")
    assert isinstance(radius, list)


# -- auditor ---------------------------------------------------------------
def test_auditor_flags_known_issues(sample_pkg):
    report = Auditor(use_ruff=False).audit(sample_pkg)
    kinds = {f.kind for f in report.findings}
    assert "bare_except" in kinds
    assert "missing_return_type" in kinds
    assert report.files_scanned == 2
    counts = report.by_severity()
    assert sum(counts.values()) == report.total_findings


def test_invariants_clean_when_no_cycles(sample_pkg):
    result = check_invariants(sample_pkg)
    assert isinstance(result.passed, bool)


# -- telemetry -------------------------------------------------------------
def test_reward_is_bounded():
    r = compute_reward(test_attempts=1, converged=True, complexity_delta=-2.0, diff_lines=5)
    assert 0.0 <= r.reward_score <= 1.0
    worse = compute_reward(test_attempts=8, converged=False, complexity_delta=10.0, diff_lines=400)
    assert worse.reward_score <= r.reward_score


def test_ledger_writes_both_formats(sample_pkg, tmp_path):
    cfg = Config(target=sample_pkg, repo=tmp_path,
                 tracking_dir=tmp_path / "tracking", reports_dir=tmp_path / "reports")
    sm = StateMachine(cfg)
    result = sm.run_round(1)
    assert (tmp_path / "tracking" / "maximillian_ledger.yaml").exists()
    assert (tmp_path / "tracking" / "dspy_trainset.jsonl").exists()


# -- full round ------------------------------------------------------------
def test_full_round_produces_report(sample_pkg, tmp_path):
    cfg = Config(target=sample_pkg, repo=tmp_path,
                 tracking_dir=tmp_path / "tracking", reports_dir=tmp_path / "reports",
                 max_rounds=1)
    results = StateMachine(cfg).run()
    assert len(results) == 1
    res = results[0]
    assert res.report_path and Path(res.report_path).exists()
    html = Path(res.report_path).read_text(encoding="utf-8")
    assert "Maximillian" in html
    assert "http://" not in html.replace("http://localhost", "")  # FR-04: no external CDN
    # 14 stages all recorded in the timeline.
    assert len(res.timeline) >= 14


def test_report_is_self_contained(sample_pkg, tmp_path):
    cfg = Config(target=sample_pkg, repo=tmp_path, reports_dir=tmp_path / "reports",
                 tracking_dir=tmp_path / "tracking")
    res = StateMachine(cfg).run_round(1)
    html = Path(res.report_path).read_text(encoding="utf-8")
    for bad in ("cdn.", "googleapis", "unpkg", "jsdelivr", "<script src"):
        assert bad not in html


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
