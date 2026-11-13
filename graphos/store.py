"""SQLite persistence for Graphos: sessions, runs, graphs, artifacts.

Two jobs here:

* **State** — sessions and their graphs/artifacts survive restarts, so the
  workspace is real rather than ephemeral.
* **Audit & observability** — every generation is a ``run`` row capturing the
  exact input, parameters, engine, latency, token/cost estimate and error.
  That record is the reasoning-provenance trail that makes a graph defensible,
  and the per-call log the research flags as the production minimum.

Thread-safe via a single connection guarded by a lock (sufficient for the
in-process scale; swap for a pool/Postgres when horizontal scaling is needed).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    status TEXT NOT NULL, engine TEXT, input_text TEXT NOT NULL,
    params_json TEXT NOT NULL, graph_id TEXT, error TEXT,
    latency_ms INTEGER, tokens_est INTEGER,
    created_at TEXT NOT NULL, finished_at TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS graphs (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, run_id TEXT,
    payload_json TEXT NOT NULL, nodes INTEGER, edges INTEGER,
    quality REAL, engine TEXT, created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, graph_id TEXT,
    kind TEXT NOT NULL, name TEXT NOT NULL, meta_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_graphs_session ON graphs(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def ping(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # -- sessions ----------------------------------------------------------
    def create_session(self, title: str) -> Dict[str, Any]:
        sid, now = _id("sess"), _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (sid, title[:200] or "Untitled session", now, now))
            self._conn.commit()
        return self.get_session(sid)

    def get_session(self, sid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM graphs g WHERE g.session_id=s.id) AS graph_count "
                "FROM sessions s ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def touch_session(self, sid: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), sid))
            self._conn.commit()

    # -- runs (audit / observability) -------------------------------------
    def create_run(self, session_id: str, input_text: str, params: Dict[str, Any]) -> str:
        rid = _id("run")
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs(id,session_id,status,input_text,params_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (rid, session_id, "pending", input_text, json.dumps(params), _now()))
            self._conn.commit()
        return rid

    def start_run(self, rid: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status='running' WHERE id=?", (rid,))
            self._conn.commit()

    def finish_run(self, rid: str, *, status: str, engine: Optional[str] = None,
                   graph_id: Optional[str] = None, error: Optional[str] = None,
                   latency_ms: Optional[int] = None, tokens_est: Optional[int] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?,engine=?,graph_id=?,error=?,latency_ms=?,"
                "tokens_est=?,finished_at=? WHERE id=?",
                (status, engine, graph_id, error, latency_ms, tokens_est, _now(), rid))
            self._conn.commit()

    def get_run(self, rid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,session_id,status,engine,input_text,graph_id,error,latency_ms,"
                "created_at,finished_at FROM runs WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- graphs ------------------------------------------------------------
    def save_graph(self, session_id: str, run_id: Optional[str], payload: Dict[str, Any]) -> str:
        gid = _id("graph")
        metrics = payload.get("metrics", {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO graphs(id,session_id,run_id,payload_json,nodes,edges,quality,engine,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (gid, session_id, run_id, json.dumps(payload), metrics.get("total_nodes"),
                 metrics.get("total_edges"), metrics.get("quality_score"),
                 payload.get("engine"), _now()))
            self._conn.commit()
        self.touch_session(session_id)
        return gid

    def get_graph(self, gid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM graphs WHERE id=?", (gid,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["payload"] = json.loads(out.pop("payload_json"))
        return out

    def list_graphs(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,session_id,run_id,nodes,edges,quality,engine,created_at "
                "FROM graphs WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- artifacts ---------------------------------------------------------
    def create_artifact(self, session_id: str, graph_id: Optional[str], kind: str,
                        name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        aid = _id("art")
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts(id,session_id,graph_id,kind,name,meta_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (aid, session_id, graph_id, kind, name[:200], json.dumps(meta or {}), _now()))
            self._conn.commit()
        return self.get_artifact(aid)

    def get_artifact(self, aid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["meta"] = json.loads(out.pop("meta_json") or "{}")
        return out

    def list_artifacts(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,session_id,graph_id,kind,name,created_at FROM artifacts "
                "WHERE session_id=? ORDER BY created_at DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(r) for r in rows]
