"""Tests for the production layer: store, jobs, sessions, async runs, errors."""

import os
import tempfile
import time

import pytest

_TMP = tempfile.mkdtemp(prefix="graphos-plat-")
os.environ["GRAPHOS_DISABLE_LIVE"] = "1"
os.environ["GRAPHOS_DB_PATH"] = os.path.join(_TMP, "plat.db")
os.environ["GRAPHOS_TELEMETRY_DIR"] = os.path.join(_TMP, "telemetry")
os.environ["GRAPHOS_EXECUTION_LOG"] = os.path.join(_TMP, "log.md")

from fastapi.testclient import TestClient  # noqa: E402

from graphos.app import app  # noqa: E402
from graphos.store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


# -- store ----------------------------------------------------------------
def test_store_session_roundtrip(store):
    s = store.create_session("Churn analysis")
    assert store.get_session(s["id"])["title"] == "Churn analysis"
    assert any(x["id"] == s["id"] for x in store.list_sessions())


def test_store_run_lifecycle_and_graph(store):
    s = store.create_session("x")
    rid = store.create_run(s["id"], "topic", {"max_nodes": 10})
    assert store.get_run(rid)["status"] == "pending"
    store.start_run(rid)
    gid = store.save_graph(s["id"], rid, {"metrics": {"total_nodes": 7, "total_edges": 12,
                                                      "quality_score": 0.6}, "engine": "deterministic"})
    store.finish_run(rid, status="succeeded", engine="deterministic", graph_id=gid, latency_ms=42)
    run = store.get_run(rid)
    assert run["status"] == "succeeded" and run["graph_id"] == gid and run["latency_ms"] == 42
    assert store.get_graph(gid)["nodes"] == 7


def test_store_artifacts(store):
    s = store.create_session("x")
    a = store.create_artifact(s["id"], None, "graph", "Snapshot", {"k": 1})
    assert a["meta"]["k"] == 1
    assert len(store.list_artifacts(s["id"])) == 1


def test_store_ping(store):
    assert store.ping() is True


# -- API ------------------------------------------------------------------
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_store_and_backend(client):
    body = client.get("/api/health").json()
    assert body["status"] == "healthy"
    assert body["store"] == "ok"
    assert "model_backend" in body


def test_session_crud_and_detail(client):
    created = client.post("/api/sessions", json={"title": "Fraud ring"}).json()
    sid = created["id"]
    assert client.get("/api/sessions").json()["sessions"]
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["session"]["title"] == "Fraud ring"
    assert detail["graphs"] == [] and detail["artifacts"] == []


def test_missing_session_returns_structured_404(client):
    r = client.get("/api/sessions/sess_does_not_exist")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "http_error"
    assert body["error"]["request_id"]


def test_validation_error_envelope(client):
    r = client.post("/api/graph", json={"input_text": "", "max_nodes": 1})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_async_run_completes_and_persists(client):
    sid = client.post("/api/sessions", json={"title": "Async"}).json()["id"]
    enq = client.post(f"/api/sessions/{sid}/graph",
                      json={"input_text": "supply chain disruption", "max_nodes": 10}).json()
    run_id = enq["run_id"]
    assert enq["status"] == "pending"

    deadline = time.time() + 20
    status = "pending"
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        status = run["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    assert status == "succeeded", f"run ended {status}"

    run = client.get(f"/api/runs/{run_id}").json()
    graph = client.get(f"/api/graphs/{run['graph_id']}").json()
    assert graph["payload"]["metrics"]["total_nodes"] >= 1
    # Artifact auto-created on success.
    assert client.get(f"/api/sessions/{sid}/artifacts").json()["artifacts"]


def test_node_cap_enforced(client):
    r = client.post("/api/graph", json={"input_text": "cap test", "max_nodes": 200}).json()
    # Capped to GRAPHOS_MAX_NODES default (40); engine may return fewer.
    assert r["metrics"]["total_nodes"] <= 40


def test_sync_graph_persists_run_and_graph(client):
    r = client.post("/api/graph", json={"input_text": "incident postmortem", "max_nodes": 10}).json()
    assert r["graph_id"] and r["run_id"] and r["session_id"]
    graph = client.get(f"/api/graphs/{r['graph_id']}").json()
    assert graph["payload"]["metrics"]["total_nodes"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# -- evaluation harness ---------------------------------------------------
def test_eval_suite_passes():
    from graphos.evals import run_suite
    report = run_suite()
    assert report.pass_rate >= 0.8, f"eval pass rate {report.pass_rate}"
    for r in report.results:
        assert r.scores.get("score_validity") == 1.0
