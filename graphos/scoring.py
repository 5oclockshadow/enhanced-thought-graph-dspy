"""Deterministic, content-derived scoring for Graphos graphs.

The upstream engine seeds thoughts with ``random.uniform`` scores. In a sandbox
without a live LLM we replace those with real values computed from the actual
input text and the graph's own structure, so every number in the UI is a
reproducible function of the data — not a dice roll.

Replace this module's heuristics with a live-model path (``live_engine``) when
an API key is available; the wire format is identical.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set

from models import ThoughtGraph

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "by", "as", "at", "it", "this", "that", "from", "into",
    "our", "your", "their", "its", "we", "they", "can", "could", "will", "would",
    "about", "across", "using", "use", "via", "more", "less", "than", "then",
}
_ACTION = {
    "implement", "build", "deploy", "automate", "optimize", "reduce", "launch",
    "scale", "integrate", "migrate", "monitor", "test", "ship", "plan", "design",
    "strategy", "process", "workflow", "feasible", "cost", "roadmap", "execute",
}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _clamp(x: float, lo: float = 0.05, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


def realize_scores(graph: ThoughtGraph, query: str) -> Dict[str, float]:
    """Overwrite random scores with values derived from text + structure.

    Returns the recomputed graph-level metrics for convenience.
    """
    query_tokens = _tokens(query)
    params = graph.generation_parameters
    max_depth = max(1, getattr(params, "max_depth", 1) or 1)
    target_nodes = max(1, getattr(params, "max_nodes", len(graph.thoughts)) or 1)

    # Token sets per thought + document frequency for novelty.
    tok_by_id: Dict[str, Set[str]] = {}
    doc_freq: Dict[str, int] = {}
    for t in graph.thoughts:
        tt = _tokens(t.content) | {k.lower() for k in (t.keywords or [])}
        tok_by_id[t.id] = tt
        for term in tt:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_docs = max(1, len(graph.thoughts))

    # --- per-thought scores ------------------------------------------------
    for t in graph.thoughts:
        tt = tok_by_id[t.id]
        relevance = _jaccard(tt, query_tokens) if query_tokens else 0.4
        # Novelty: average rarity (idf) of this thought's terms.
        if tt:
            idf = sum(math.log(1 + n_docs / doc_freq[term]) for term in tt) / len(tt)
            novelty = idf / math.log(1 + n_docs)  # normalize to ~0..1
        else:
            novelty = 0.3
        practicality = _clamp(len(tt & _ACTION) / 3.0 + 0.2, 0.1, 1.0)
        depth_factor = 1.0 - (t.depth - 1) / max_depth
        confidence = 0.45 * (0.3 + 0.7 * relevance) + 0.25 * depth_factor + 0.30 * practicality

        t.relevance_score = round(_clamp(0.15 + 0.85 * relevance), 4)
        t.novelty_score = round(_clamp(novelty), 4)
        t.practicality_score = round(_clamp(practicality), 4)
        t.confidence = round(_clamp(confidence), 4)

    conf_by_id = {t.id: t.confidence for t in graph.thoughts}

    # --- per-edge scores ---------------------------------------------------
    for c in graph.connections:
        overlap = _jaccard(tok_by_id.get(c.source_id, set()), tok_by_id.get(c.target_id, set()))
        c.strength = round(_clamp(0.2 + 0.8 * overlap), 4)
        c.confidence = round(_clamp(min(conf_by_id.get(c.source_id, 0.5),
                                        conf_by_id.get(c.target_id, 0.5))), 4)

    # --- graph-level metrics ----------------------------------------------
    confidences = [t.confidence for t in graph.thoughts] or [0.0]
    novelties = [t.novelty_score for t in graph.thoughts] or [0.0]
    strengths = [c.strength for c in graph.connections] or [0.0]

    coherence = sum(strengths) / len(strengths)
    creativity = sum(novelties) / len(novelties)
    depth_coverage = (graph.max_depth_achieved or 0) / max_depth
    completeness = min(1.0, len(graph.thoughts) / target_nodes) * _clamp(depth_coverage, 0.2, 1.0)
    mean_conf = sum(confidences) / len(confidences)
    overall = 0.40 * coherence + 0.30 * mean_conf + 0.30 * completeness

    graph.coherence_score = round(_clamp(coherence, 0.0, 1.0), 4)
    graph.creativity_score = round(_clamp(creativity, 0.0, 1.0), 4)
    graph.completeness_score = round(_clamp(completeness, 0.0, 1.0), 4)
    graph.overall_quality = round(_clamp(overall, 0.0, 1.0), 4)

    # Cluster coherence from shared vocabulary of member thoughts.
    for cl in graph.clusters:
        member_tokens: List[Set[str]] = [tok_by_id.get(tid, set()) for tid in cl.thought_ids]
        pairs = [
            _jaccard(member_tokens[i], member_tokens[j])
            for i in range(len(member_tokens))
            for j in range(i + 1, len(member_tokens))
        ]
        cl.coherence_score = round(_clamp(sum(pairs) / len(pairs), 0.0, 1.0), 4) if pairs else 0.3

    return {
        "overall_quality": graph.overall_quality,
        "coherence": graph.coherence_score,
        "creativity": graph.creativity_score,
        "completeness": graph.completeness_score,
        "mean_confidence": round(mean_conf, 4),
    }
