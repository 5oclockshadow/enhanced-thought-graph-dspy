"""Dual-ledger persistence (Optimizer Telemetry Ledger).

Writes verbose, high-signal round metadata to ``tracking/`` — kept OUT of Git —
in two shapes:

* ``maximillian_ledger.yaml`` — human-readable hierarchical round history.
* ``dspy_trainset.jsonl`` — one JSON record per round, directly loadable as
  ``dspy.Example`` inputs for MIPROv2 / BootstrapFewShot.

Reward (R in [0,1]) blends test-pass speed, diff minimality, and complexity
reduction — the scalar signal downstream optimizers maximise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from .schemas import RewardBreakdown, RoundResult

logger = logging.getLogger("maximillian.telemetry")


def compute_reward(
    *, test_attempts: int, converged: bool, complexity_delta: float,
    diff_lines: int = 0,
) -> RewardBreakdown:
    """Scalar reward for DSPy optimization. Higher is better."""
    # Fewer attempts to green => higher speed score.
    speed = 1.0 / test_attempts if test_attempts > 0 else 0.0
    if not converged:
        speed *= 0.3
    # Smaller diffs are preferred (minimality).
    minimality = 1.0 / (1.0 + diff_lines / 50.0)
    # Complexity reduction: negative delta (simpler) rewarded.
    reduction = max(0.0, min(1.0, 0.5 - complexity_delta / 20.0))
    score = round(min(1.0, 0.5 * speed + 0.25 * minimality + 0.25 * reduction), 4)
    return RewardBreakdown(
        test_pass_speed=round(speed, 4), diff_minimality=round(minimality, 4),
        complexity_reduction=round(reduction, 4), reward_score=score)


class LedgerWriter:
    def __init__(self, tracking_dir: Path) -> None:
        self.dir = tracking_dir
        self.yaml_path = tracking_dir / "maximillian_ledger.yaml"
        self.jsonl_path = tracking_dir / "dspy_trainset.jsonl"

    def append(self, result: RoundResult) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._append_yaml(result)
        self._append_jsonl(result)

    def _append_yaml(self, result: RoundResult) -> None:
        try:
            history = []
            if self.yaml_path.exists():
                history = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or []
            entry = {
                "round_id": result.round_id,
                "timestamp": result.timestamp.isoformat(),
                "target": result.target,
                "converged": result.converged,
                "llm_available": result.llm_available,
                "audit_findings": result.audit.total_findings if result.audit else 0,
                "graph": result.graph.model_dump() if result.graph else {},
                "test_iterations_to_pass": result.tests.attempts if result.tests else 0,
                "ast_complexity_delta": result.health.complexity_delta if result.health else 0.0,
                "reward_score": result.reward.reward_score if result.reward else 0.0,
                "git_sha": result.git_sha,
                "snapshot_tag": result.snapshot_tag,
                "timeline": [t.model_dump() for t in result.timeline],
            }
            history.append(entry)
            self.yaml_path.write_text(yaml.safe_dump(history, sort_keys=False), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("YAML ledger append failed: %s", exc)

    def _append_jsonl(self, result: RoundResult) -> None:
        """One DSPy-ready Example record per round (inputs -> reward)."""
        try:
            record = {
                "round_id": result.round_id,
                "inputs": {
                    "target": result.target,
                    "audit_findings": result.audit.by_severity() if result.audit else {},
                    "graph_nodes": result.graph.nodes if result.graph else 0,
                    "graph_edges": result.graph.edges if result.graph else 0,
                },
                "outputs": {
                    "converged": result.converged,
                    "test_attempts": result.tests.attempts if result.tests else 0,
                },
                "reward": result.reward.reward_score if result.reward else 0.0,
                "dspy_signatures_used": ["BuildSignature", "RefactorSignature", "PatchSignature"],
            }
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("JSONL ledger append failed: %s", exc)
