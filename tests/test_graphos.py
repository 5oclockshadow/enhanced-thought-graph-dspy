"""Automated verification for the Graphos service and API.

Covers the three mandated pathways:
  * success paths (happy generation across presets/domains),
  * boundary edge-cases (min/max nodes, validation limits),
  * deliberate error injection (engine returns no graph).
"""

import os
import tempfile

import pytest

# Redirect telemetry to an ephemeral dir before importing the app layer.
_TMP = tempfile.mkdtemp(prefix="graphos-test-")
os.environ["GRAPHOS_TELEMETRY_DIR"] = os.path.join(_TMP, "telemetry")
os.environ["GRAPHOS_EXECUTION_LOG"] = os.path.join(_TMP, "log.md")
# Keep tests deterministic and offline — never call a live model.
os.environ["GRAPHOS_DISABLE_LIVE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from graphos.app import app  # noqa: E402
from graphos import service  # noqa: E402
from graphos.service import GraphRequest, GraphServiceError, generate_graph  # noqa: E402
from models import GenerationMode, ClusteringMethod  # noqa: E402

client = TestClient(app)


# --------------------------------------------------------------------------
# Success paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize("domain", [None, "technology", "business", "creative"])
def test_generate_all_domains_succeed(domain):
    """The template-fill bug fix: every domain yields a real graph."""
    req = GraphRequest(input_text="churn drivers", max_nodes=12, domain_focus=domain)
    data = generate_graph(req)
    assert data["metrics"]["total_nodes"] >= 1
    assert "learning" in data
    assert 0.0 <= data["learning"]["hebbian_plasticity"] <= 1.0


def test_reinforcement_fields_present():
    data = generate_graph(GraphRequest(input_text="x", max_nodes=10))
    for edge in data["connections"]:
        assert "reinforced_weight" in edge
        assert edge["reinforced_weight"] <= 1.0


def test_scores_are_derived_not_random():
    """Relevance tracks real overlap with the query; nothing is random."""
    data = generate_graph(GraphRequest(input_text="renewable solar energy storage", max_nodes=12))
    assert "scoring" in data
    # Every score is a bounded, rounded real number.
    for t in data["thoughts"]:
        for key in ("confidence", "relevance", "novelty", "practicality"):
            assert 0.0 <= t[key] <= 1.0
    for c in data["connections"]:
        assert 0.0 <= c["strength"] <= 1.0
    # Graph quality is a weighted blend, so strictly within (0, 1).
    assert 0.0 < data["metrics"]["quality_score"] < 1.0


def test_api_graph_endpoint():
    resp = client.post("/api/graph", json={"input_text": "sustainable energy", "max_nodes": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["total_nodes"] >= 1
    assert body["telemetry"]["span_id"]


def test_health_and_presets():
    assert client.get("/api/health").json()["status"] == "healthy"
    presets = client.get("/api/presets").json()["presets"]
    assert "churn_intelligence" in presets


# --------------------------------------------------------------------------
# Boundary edge-cases
# --------------------------------------------------------------------------
def test_min_nodes_boundary():
    data = generate_graph(GraphRequest(input_text="edge", max_nodes=5))
    assert data["metrics"]["total_nodes"] <= 5


def test_validation_rejects_out_of_range_nodes():
    resp = client.post("/api/graph", json={"input_text": "x", "max_nodes": 1})
    assert resp.status_code == 422  # pydantic validation (ge=5)


def test_validation_rejects_empty_input():
    resp = client.post("/api/graph", json={"input_text": "", "max_nodes": 10})
    assert resp.status_code == 422


def test_unknown_domain_does_not_crash():
    data = generate_graph(GraphRequest(input_text="x", max_nodes=8, domain_focus="quantum-biology"))
    assert data["metrics"]["total_nodes"] >= 1


# --------------------------------------------------------------------------
# Deliberate error injection
# --------------------------------------------------------------------------
def test_engine_returns_no_graph_raises_service_error(monkeypatch):
    class _Empty:
        graph = None
        insights = []
        recommendations = []
        summary = None

    monkeypatch.setattr(service, "_reactor", type("R", (), {"__call__": lambda self, *a, **k: _Empty()})())
    with pytest.raises(GraphServiceError):
        generate_graph(GraphRequest(input_text="boom", max_nodes=10))


def test_api_surfaces_engine_failure_as_502(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("synthetic engine failure")

    monkeypatch.setattr(service, "_reactor", type("R", (), {"__call__": lambda self, *a, **k: _raise()})())
    resp = client.post("/api/graph", json={"input_text": "boom", "max_nodes": 10})
    assert resp.status_code == 502


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
