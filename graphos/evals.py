"""Evaluation harness for Graphos graph generation.

The research is blunt: an evaluation framework built before shipping is the
strongest predictor of production success. This module scores generated graphs
against structural and validity metrics across a representative suite of the
use cases Graphos targets (customer-360 / churn, fraud, supply chain, incident
postmortem, competitive intel, research synthesis).

Engine-agnostic: it evaluates whichever engine is active (deterministic or
live) and reports which one produced each graph, so the same suite doubles as
a regression gate and a model-quality check.

    python -m graphos.evals            # run suite, print table
    python -m graphos.evals --json out.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .service import GraphRequest, generate_graph


@dataclass
class EvalCase:
    name: str
    use_case: str
    request: Dict[str, Any]


SUITE: List[EvalCase] = [
    EvalCase("churn_drivers", "customer-360 / churn",
             {"input_text": "why enterprise SaaS accounts churn in Q4",
              "generation_mode": "analytical", "domain_focus": "business", "max_nodes": 14}),
    EvalCase("fraud_ring", "fraud / financial crime",
             {"input_text": "detecting coordinated fraud across accounts and devices",
              "generation_mode": "critical", "domain_focus": "business", "max_nodes": 14}),
    EvalCase("supply_chain", "supply chain risk",
             {"input_text": "how a supplier outage propagates through fulfillment",
              "generation_mode": "analytical", "domain_focus": "technology", "max_nodes": 14}),
    EvalCase("incident_postmortem", "incident postmortem",
             {"input_text": "root causes of a cascading production outage",
              "generation_mode": "critical", "domain_focus": "technology", "max_nodes": 14}),
    EvalCase("market_intel", "competitive intelligence",
             {"input_text": "competitive threats to our analytics platform",
              "generation_mode": "exploratory", "domain_focus": "business", "max_nodes": 14}),
    EvalCase("research_synthesis", "research synthesis",
             {"input_text": "synthesis of approaches to graph retrieval augmented generation",
              "generation_mode": "exploratory", "max_nodes": 14}),
]


# -- metrics (each returns 0..1) ------------------------------------------
def m_nodes_present(p: Dict[str, Any]) -> float:
    return 1.0 if p["metrics"]["total_nodes"] >= 5 else p["metrics"]["total_nodes"] / 5.0


def m_connectivity(p: Dict[str, Any]) -> float:
    n = p["metrics"]["total_nodes"] or 1
    return min(1.0, (p["metrics"]["total_edges"] / n) / 2.0)


def m_connected_fraction(p: Dict[str, Any]) -> float:
    ids = {t["id"] for t in p["thoughts"]}
    if not ids:
        return 0.0
    touched = set()
    for c in p["connections"]:
        touched.add(c["source"])
        touched.add(c["target"])
    return len(touched & ids) / len(ids)


def m_score_validity(p: Dict[str, Any]) -> float:
    for t in p["thoughts"]:
        for k in ("confidence", "relevance", "novelty", "practicality"):
            if not (0.0 <= t[k] <= 1.0):
                return 0.0
    for c in p["connections"]:
        if not (0.0 <= c["strength"] <= 1.0):
            return 0.0
    return 1.0


def m_clustered_fraction(p: Dict[str, Any]) -> float:
    ids = {t["id"] for t in p["thoughts"]}
    if not ids:
        return 0.0
    in_cluster = set()
    for cl in p["clusters"]:
        in_cluster.update(cl["thought_ids"])
    return len(in_cluster & ids) / len(ids)


METRICS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "nodes_present": m_nodes_present,
    "connectivity": m_connectivity,
    "connected_fraction": m_connected_fraction,
    "score_validity": m_score_validity,
    "clustered_fraction": m_clustered_fraction,
}

PASS_THRESHOLD = 0.6


@dataclass
class CaseResult:
    name: str
    use_case: str
    engine: str
    scores: Dict[str, float]
    aggregate: float
    passed: bool
    error: str = ""


@dataclass
class EvalReport:
    results: List[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"pass_rate": round(self.pass_rate, 3),
                "cases": [r.__dict__ for r in self.results]}


def run_suite(suite: List[EvalCase] = SUITE) -> EvalReport:
    report = EvalReport()
    for case in suite:
        try:
            payload = generate_graph(GraphRequest(**case.request))
            scores = {name: round(fn(payload), 4) for name, fn in METRICS.items()}
            agg = round(sum(scores.values()) / len(scores), 4)
            passed = agg >= PASS_THRESHOLD and scores["score_validity"] == 1.0 and scores["nodes_present"] == 1.0
            report.results.append(CaseResult(
                case.name, case.use_case, payload.get("engine", "?"), scores, agg, passed))
        except Exception as exc:  # pragma: no cover - surfaced as a failed case
            report.results.append(CaseResult(
                case.name, case.use_case, "error", {}, 0.0, False, str(exc)))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="graphos.evals")
    ap.add_argument("--json", help="write the full report to this path")
    args = ap.parse_args(argv)

    report = run_suite()
    print(f"\nGraphos evaluation — {len(report.results)} cases · "
          f"pass rate {report.pass_rate:.0%}\n")
    print(f"{'case':<22}{'use case':<26}{'engine':<18}{'agg':>6}  result")
    print("-" * 80)
    for r in report.results:
        print(f"{r.name:<22}{r.use_case:<26}{r.engine:<18}{r.aggregate:>6.2f}  "
              f"{'PASS' if r.passed else 'FAIL'}{('  ' + r.error) if r.error else ''}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nReport written to {args.json}")
    return 0 if report.pass_rate >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
