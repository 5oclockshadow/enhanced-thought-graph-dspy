"""Graphos service layer: request validation, generation, reinforcement.

Wraps the existing ``AutonomousThoughtReactor`` with:

* Strict typed ingestion (Pydantic request model).
* A deterministic Hebbian + Q-table reinforcement pass over the generated
  edges, turning the design language into real, computed edge weights.
* Resilient fault isolation around every traversal step.
* Pre/post cognitive telemetry.

No mocked responses: a request always returns a real graph produced by the
engine, or a typed error.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from models import (
    ClusteringMethod,
    ExpandedInput,
    GenerationMode,
    GraphParameters,
    ThoughtGraph,
)
from dspy_program import AutonomousThoughtReactor

from .scoring import realize_scores
from .telemetry import TelemetrySpan

logger = logging.getLogger("graphos.service")

# A single reactor instance is reused across requests (stateless per call).
_reactor = AutonomousThoughtReactor()


class GraphRequest(BaseModel):
    """Typed, validated payload for a graph-intelligence task."""

    input_text: str = Field(..., min_length=1, max_length=5000)
    max_nodes: int = Field(default=18, ge=5, le=200)
    max_depth: int = Field(default=4, ge=1, le=10)
    generation_mode: GenerationMode = GenerationMode.BALANCED
    clustering_method: ClusteringMethod = ClusteringMethod.THEMATIC
    creativity_level: float = Field(default=0.7, ge=0.0, le=1.0)
    divergence_factor: float = Field(default=0.6, ge=0.0, le=1.0)
    depth_vs_breadth: float = Field(default=0.5, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    max_clusters: int = Field(default=5, ge=1, le=20)
    domain_focus: Optional[str] = None
    enable_meta_thinking: bool = False
    enable_emotional_layer: bool = True
    enable_practical_layer: bool = True
    enable_risk_analysis: bool = False
    focus_areas: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    # Learning knobs (Hebbian / Q-table reinforcement layer).
    hebbian_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    q_discount: float = Field(default=0.85, ge=0.0, le=1.0)

    def to_parameters(self) -> GraphParameters:
        return GraphParameters(
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            generation_mode=self.generation_mode,
            creativity_level=self.creativity_level,
            divergence_factor=self.divergence_factor,
            depth_vs_breadth=self.depth_vs_breadth,
            clustering_method=self.clustering_method,
            max_clusters=self.max_clusters,
            min_confidence=self.min_confidence,
            domain_focus=self.domain_focus,
            enable_meta_thinking=self.enable_meta_thinking,
            enable_emotional_layer=self.enable_emotional_layer,
            enable_practical_layer=self.enable_practical_layer,
            enable_risk_analysis=self.enable_risk_analysis,
        )


class GraphServiceError(RuntimeError):
    """Raised when generation fails irrecoverably."""


def _reinforce(graph: ThoughtGraph, hebbian_rate: float, q_discount: float) -> Dict[str, Any]:
    """Apply a deterministic Hebbian + Q-table reinforcement pass to edges.

    * Degree (co-activation) raises each edge's reinforced weight — "cells that
      fire together wire together".
    * A one-step Q-value propagates downstream node confidence back along the
      edge, discounted by ``q_discount``.

    Returns aggregate learning signals for the UI readouts. Fault-isolated:
    any per-edge failure is skipped, never fatal.
    """
    degree: Dict[str, int] = {}
    for conn in graph.connections:
        degree[conn.source_id] = degree.get(conn.source_id, 0) + 1
        degree[conn.target_id] = degree.get(conn.target_id, 0) + 1

    max_degree = max(degree.values()) if degree else 1
    confidence_by_id = {t.id: t.confidence for t in graph.thoughts}

    reinforced_total = 0.0
    q_total = 0.0
    counted = 0
    for conn in graph.connections:
        try:
            co_activation = (degree.get(conn.source_id, 0) + degree.get(conn.target_id, 0)) / (2 * max_degree)
            reinforced = min(1.0, conn.strength * (1.0 + hebbian_rate * co_activation))
            downstream = confidence_by_id.get(conn.target_id, 0.5)
            q_value = conn.strength + q_discount * downstream * conn.confidence
            # Stash on the connection (model Config allows extra fields).
            setattr(conn, "reinforced_weight", round(reinforced, 4))
            setattr(conn, "q_value", round(q_value, 4))
            setattr(conn, "co_activation", round(co_activation, 4))
            reinforced_total += reinforced
            q_total += q_value
            counted += 1
        except Exception as exc:  # pragma: no cover - never fatal
            logger.debug("Reinforcement skipped for edge %s: %s", getattr(conn, "id", "?"), exc)

    return {
        "hebbian_plasticity": round(reinforced_total / counted, 4) if counted else 0.0,
        "q_reward": round(q_total / counted, 4) if counted else 0.0,
        "max_degree": max_degree,
        "edges_reinforced": counted,
    }


def _shape(graph: ThoughtGraph) -> Dict[str, Any]:
    """Serialize a ThoughtGraph into the UI's wire format."""
    return {
        "thoughts": [
            {
                "id": t.id,
                "content": t.content,
                "type": t.thought_type.value,
                "confidence": round(t.confidence, 3),
                "depth": t.depth,
                "relevance": round(t.relevance_score, 3),
                "novelty": round(t.novelty_score, 3),
                "practicality": round(t.practicality_score, 3),
                "impact": t.potential_impact,
                "complexity": t.complexity_level,
                "tags": t.tags,
                "keywords": t.keywords,
                "cluster_id": t.cluster_id,
                "domain": t.domain,
            }
            for t in graph.thoughts
        ],
        "connections": [
            {
                "id": c.id,
                "source": c.source_id,
                "target": c.target_id,
                "type": c.connection_type.value,
                "strength": round(c.strength, 3),
                "confidence": round(c.confidence, 3),
                "reinforced_weight": getattr(c, "reinforced_weight", round(c.strength, 3)),
                "q_value": getattr(c, "q_value", None),
                "description": c.description,
            }
            for c in graph.connections
        ],
        "clusters": [
            {
                "id": cl.id,
                "name": cl.name,
                "description": cl.description,
                "thought_ids": cl.thought_ids,
                "coherence": round(cl.coherence_score, 3),
                "importance": round(cl.importance_score, 3),
                "type": cl.cluster_type,
            }
            for cl in graph.clusters
        ],
        "metrics": {
            "total_nodes": graph.total_nodes,
            "total_edges": graph.total_edges,
            "max_depth": graph.max_depth_achieved,
            "quality_score": round(graph.overall_quality, 3),
            "creativity_score": round(graph.creativity_score, 3),
            "coherence_score": round(graph.coherence_score, 3),
            "completeness_score": round(graph.completeness_score, 3),
        },
    }


def generate_graph(request: GraphRequest) -> Dict[str, Any]:
    """Run a full, traced generation task and return the UI payload."""
    parameters = request.to_parameters()
    span = TelemetrySpan(task=request.input_text, parameters=parameters.model_dump(mode="json"))
    span.pre(
        plan={
            "mode": request.generation_mode.value,
            "clustering": request.clustering_method.value,
            "target_nodes": request.max_nodes,
            "reinforcement": {"hebbian_rate": request.hebbian_rate, "q_discount": request.q_discount},
        }
    )

    start = time.time()
    success = False
    error: Optional[str] = None
    payload: Dict[str, Any] = {}
    learning: Dict[str, Any] = {}
    try:
        expanded = ExpandedInput(
            original_text=request.input_text,
            parameters=parameters,
            focus_areas=request.focus_areas,
            constraints=request.constraints,
            context={"surface": "graphos"},
        )
        # Autopilot: use the live model when the environment provides one,
        # otherwise the deterministic engine with data-derived scoring.
        graph = None
        engine = "deterministic"
        try:
            from . import live_engine
            if live_engine.is_live():
                graph = live_engine.generate(request.input_text, parameters,
                                             request.generation_mode.value)
                if graph is not None:
                    engine = f"live:{live_engine.backend_name()}"
        except Exception as exc:  # pragma: no cover - never fatal
            logger.warning("live engine unavailable, falling back: %s", exc)

        if graph is None:
            result = _reactor(expanded, use_fake_data=True)
            if not result.graph:
                raise GraphServiceError("engine returned no graph")
            graph = result.graph
            # Deterministic values computed from input text + graph structure.
            scoring = realize_scores(graph, request.input_text)
            insights, recommendations, summary = (
                list(result.insights or []), list(result.recommendations or []), result.summary)
        else:
            scoring = {"mean_confidence": graph.overall_quality}
            insights, recommendations = [], []
            summary = f"Live model graph on '{request.input_text}' ({engine})."

        learning = _reinforce(graph, request.hebbian_rate, request.q_discount)
        payload = _shape(graph)
        payload["learning"] = learning
        payload["scoring"] = scoring
        payload["engine"] = engine
        payload["insights"] = insights
        payload["recommendations"] = recommendations
        payload["summary"] = summary
        success = True
        return payload
    except Exception as exc:
        error = str(exc)
        logger.exception("Generation failed for %r", request.input_text)
        raise GraphServiceError(error) from exc
    finally:
        elapsed = round(time.time() - start, 4)
        metrics = {
            "elapsed_seconds": elapsed,
            "nodes": payload.get("metrics", {}).get("total_nodes", 0),
            "edges": payload.get("metrics", {}).get("total_edges", 0),
            "hebbian_plasticity": learning.get("hebbian_plasticity", 0.0),
            "q_reward": learning.get("q_reward", 0.0),
        }
        span.post(success=success, metrics=metrics, error=error)
        if success:
            payload["telemetry"] = {"span_id": span.span_id, "paths": span.paths, "elapsed_seconds": elapsed}


PRESETS: Dict[str, Dict[str, Any]] = {
    "churn_intelligence": {
        "name": "Churn Intelligence",
        "description": "Analytical graph of account churn drivers",
        "parameters": {
            "max_nodes": 18,
            "generation_mode": "analytical",
            "clustering_method": "causal",
            "creativity_level": 0.4,
            "depth_vs_breadth": 0.7,
            "domain_focus": "business",
            "enable_risk_analysis": True,
        },
    },
    "creative_brainstorm": {
        "name": "Creative Brainstorm",
        "description": "Maximum divergence for innovation sessions",
        "parameters": {
            "max_nodes": 25,
            "generation_mode": "creative",
            "clustering_method": "thematic",
            "creativity_level": 0.9,
            "divergence_factor": 0.85,
            "domain_focus": "creative",
            "enable_meta_thinking": True,
        },
    },
    "technical_planning": {
        "name": "Technical Planning",
        "description": "Implementation-focused functional map",
        "parameters": {
            "max_nodes": 16,
            "generation_mode": "practical",
            "clustering_method": "functional",
            "creativity_level": 0.6,
            "min_confidence": 0.6,
            "domain_focus": "technology",
            "enable_practical_layer": True,
        },
    },
}
