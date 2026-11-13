"""Live model-backed graph generation for Graphos.

Reuses Maximillian's environment auto-detected backend (the session `claude`
CLI, or any DSPy-addressable model) to produce *real* thought content and
scores — not templated placeholders. Returns a ``models.ThoughtGraph`` so the
service can run the same reinforcement + shaping path, or ``None`` to fall back
to the deterministic engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Optional

from models import (
    Connection, ConnectionType, GraphParameters, Thought, ThoughtCluster,
    ThoughtGraph, ThoughtType,
)

logger = logging.getLogger("graphos.live_engine")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_backend = None
_detected = False

_SYS = (
    "You are a knowledge-graph engine. Output ONLY one JSON object, no prose, no "
    "code fence. Schema: {\"thoughts\":[{\"content\":str,\"type\":str,\"confidence\":0..1,"
    "\"relevance\":0..1,\"novelty\":0..1,\"practicality\":0..1}],\"connections\":"
    "[{\"source\":int,\"target\":int,\"type\":str,\"strength\":0..1}],\"clusters\":"
    "[{\"name\":str,\"members\":[int]}]}. type is one of core_concept, sub_concept, "
    "question, hypothesis, solution, challenge, opportunity, insight, pattern, cause, "
    "effect, synthesis, critique. connection type is one of leads_to, caused_by, "
    "supports, depends_on, enables, prevents, similar_to, part_of, conflicts_with. "
    "source/target are integer indices into thoughts."
)


def _get_backend():
    global _backend, _detected
    if _detected:
        return _backend
    _detected = True
    if os.environ.get("GRAPHOS_DISABLE_LIVE"):
        return None
    try:
        from maximillian.synthesizer import detect_backend
        _backend = detect_backend()
    except Exception as exc:  # pragma: no cover
        logger.warning("backend detection failed: %s", exc)
        _backend = None
    return _backend


def is_live() -> bool:
    return _get_backend() is not None


def backend_name() -> str:
    b = _get_backend()
    return b.name if b else "none"


def _enum(value: str, enum_cls, default):
    try:
        return enum_cls(str(value).strip().lower())
    except Exception:
        return default


def _clamp(x, lo=0.05, hi=0.99) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.5


def generate(input_text: str, params: GraphParameters, mode: str) -> Optional[ThoughtGraph]:
    """Generate a real graph via the live model; None on any failure."""
    backend = _get_backend()
    if backend is None:
        return None
    n = max(5, min(params.max_nodes, 18))
    prompt = (f"Topic: {input_text}\nGeneration mode: {mode}\n"
              f"Produce exactly {n} thoughts and a connected graph.")
    try:
        raw = backend.complete(prompt, _SYS, timeout=150)
    except Exception as exc:
        logger.warning("live generation call failed: %s", exc)
        return None

    m = _JSON_RE.search(raw or "")
    if not m:
        logger.warning("no JSON in model output")
        return None
    try:
        data = json.loads(m.group(0))
    except Exception as exc:
        logger.warning("JSON parse failed: %s", exc)
        return None

    raw_thoughts = data.get("thoughts") or []
    if not raw_thoughts:
        return None

    thoughts = []
    for i, t in enumerate(raw_thoughts[:n]):
        ttype = _enum(t.get("type"), ThoughtType, ThoughtType.CORE_CONCEPT)
        depth = 1 if ttype in (ThoughtType.CORE_CONCEPT,) else 2
        thoughts.append(Thought(
            id=str(uuid.uuid4()),
            content=str(t.get("content", "")).strip()[:1000] or f"Idea {i+1}",
            thought_type=ttype,
            confidence=_clamp(t.get("confidence", 0.6)),
            depth=depth,
            relevance_score=_clamp(t.get("relevance", 0.6)),
            novelty_score=_clamp(t.get("novelty", 0.5)),
            practicality_score=_clamp(t.get("practicality", 0.5)),
        ))
    idx = {i: th.id for i, th in enumerate(thoughts)}

    connections = []
    for c in (data.get("connections") or []):
        s, t = c.get("source"), c.get("target")
        if s not in idx or t not in idx or s == t:
            continue
        connections.append(Connection(
            id=str(uuid.uuid4()), source_id=idx[s], target_id=idx[t],
            connection_type=_enum(c.get("type"), ConnectionType, ConnectionType.LEADS_TO),
            strength=_clamp(c.get("strength", 0.6)), confidence=_clamp(c.get("strength", 0.6)),
            description=f"model edge: {c.get('type', 'leads_to')}"))

    clusters = []
    for cl in (data.get("clusters") or []):
        members = [idx[i] for i in (cl.get("members") or []) if i in idx]
        if members:
            clusters.append(ThoughtCluster(
                id=str(uuid.uuid4()), name=str(cl.get("name", "cluster"))[:120],
                description="model-grouped", thought_ids=members,
                cluster_type=params.clustering_method.value,
                coherence_score=0.7, importance_score=0.6))

    total_nodes = len(thoughts)
    total_edges = len(connections)
    max_depth = max((t.depth for t in thoughts), default=1)
    confs = [t.confidence for t in thoughts] or [0.5]
    return ThoughtGraph(
        thoughts=thoughts, connections=connections, clusters=clusters,
        generation_parameters=params, total_nodes=total_nodes, total_edges=total_edges,
        max_depth_achieved=max_depth,
        average_connectivity=total_edges / total_nodes if total_nodes else 0.0,
        graph_density=total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0,
        overall_quality=round(sum(confs) / len(confs), 4),
        creativity_score=round(sum(t.novelty_score for t in thoughts) / total_nodes, 4) if total_nodes else 0.0,
        coherence_score=round(sum(c.strength for c in connections) / total_edges, 4) if total_edges else 0.0,
        completeness_score=round(min(1.0, total_nodes / max(1, params.max_nodes)), 4))
