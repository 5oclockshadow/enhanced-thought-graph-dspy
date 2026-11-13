"""Graphos FastAPI application — persistent, observable, hardened.

Surface:
  GET  /api/health                      dependency-aware health
  GET  /api/presets                     parameter presets
  POST /api/sessions                    create a workspace session
  GET  /api/sessions                    list sessions
  GET  /api/sessions/{sid}              session + graphs + runs + artifacts
  POST /api/sessions/{sid}/graph        enqueue async generation -> run_id
  GET  /api/runs/{rid}                   poll run status (audit record)
  GET  /api/graphs/{gid}                 fetch a stored graph
  GET  /api/sessions/{sid}/artifacts     list artifacts
  POST /api/graph                        synchronous generate (+persist); back-compat

Every response error is a structured envelope with a request id; every request
is logged with latency. Generation is validated, capped, and persisted.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .jobs import JobManager
from .service import PRESETS, GraphRequest, GraphServiceError, generate_graph
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("graphos.app")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    app.state.store = Store(settings.db_path)
    app.state.jobs = JobManager(app.state.store, workers=settings.job_workers)
    logger.info("Graphos %s started · db=%s", __version__, settings.db_path)
    try:
        yield
    finally:
        if getattr(app.state, "jobs", None):
            app.state.jobs.shutdown()
        if getattr(app.state, "store", None):
            app.state.store.close()
        app.state.jobs = None
        app.state.store = None


app = FastAPI(title="Graphos — Graph Intelligence Workspace", version=__version__,
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])


# -- middleware: request id + access log ----------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = rid
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] unhandled error on %s %s", rid, request.method, request.url.path)
        raise
    ms = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = rid
    if settings.request_log and request.url.path.startswith("/api"):
        logger.info("[%s] %s %s -> %s %dms", rid, request.method, request.url.path,
                    response.status_code, ms)
    return response


def _envelope(status: int, code: str, message: str, request: Request) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message, "request_id": rid}})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return _envelope(exc.status_code, "http_error", str(exc.detail), request)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return _envelope(422, "validation_error", "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]), request)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    return _envelope(500, "internal_error", "an unexpected error occurred", request)


# -- helpers --------------------------------------------------------------
def _store(request: Request) -> Store:
    app = request.app
    store = getattr(app.state, "store", None)
    if store is None or not store.ping():
        settings.ensure_dirs()
        app.state.store = Store(settings.db_path)
        app.state.jobs = None  # rebind the job manager to the fresh store
    return app.state.store


def _jobs(request: Request) -> JobManager:
    app = request.app
    _store(request)  # ensure a live store exists first
    if not getattr(app.state, "jobs", None):
        app.state.jobs = JobManager(app.state.store, settings.job_workers)
    return app.state.jobs


def _cap_nodes(req: GraphRequest) -> GraphRequest:
    if req.max_nodes > settings.max_nodes_cap:
        req.max_nodes = settings.max_nodes_cap
    return req


class SessionCreate(BaseModel):
    title: str = Field(default="Untitled session", max_length=200)


# -- health & presets -----------------------------------------------------
@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    store = _store(request)
    try:
        from . import live_engine
        backend = live_engine.backend_name()
    except Exception:
        backend = "unknown"
    return {"status": "healthy" if store.ping() else "degraded",
            "service": "graphos", "version": __version__,
            "store": "ok" if store.ping() else "down", "model_backend": backend}


@app.get("/api/presets")
async def presets() -> Dict[str, Any]:
    return {"presets": PRESETS}


# -- sessions -------------------------------------------------------------
@app.post("/api/sessions")
async def create_session(request: Request, body: SessionCreate) -> Dict[str, Any]:
    return _store(request).create_session(body.title)


@app.get("/api/sessions")
async def list_sessions(request: Request) -> Dict[str, Any]:
    return {"sessions": _store(request).list_sessions()}


@app.get("/api/sessions/{sid}")
async def get_session(request: Request, sid: str) -> Dict[str, Any]:
    store = _store(request)
    session = store.get_session(sid)
    if not session:
        raise HTTPException(404, f"session {sid} not found")
    return {"session": session, "graphs": store.list_graphs(sid),
            "runs": store.list_runs(sid), "artifacts": store.list_artifacts(sid)}


@app.post("/api/sessions/{sid}/graph")
async def enqueue_graph(request: Request, sid: str, body: GraphRequest) -> Dict[str, Any]:
    store = _store(request)
    if not store.get_session(sid):
        raise HTTPException(404, f"session {sid} not found")
    run_id = _jobs(request).submit(sid, _cap_nodes(body))
    return {"run_id": run_id, "status": "pending", "session_id": sid}


# -- runs, graphs, artifacts ---------------------------------------------
@app.get("/api/runs/{rid}")
async def get_run(request: Request, rid: str) -> Dict[str, Any]:
    run = _store(request).get_run(rid)
    if not run:
        raise HTTPException(404, f"run {rid} not found")
    return run


@app.get("/api/graphs/{gid}")
async def get_graph(request: Request, gid: str) -> Dict[str, Any]:
    graph = _store(request).get_graph(gid)
    if not graph:
        raise HTTPException(404, f"graph {gid} not found")
    return graph


@app.get("/api/sessions/{sid}/artifacts")
async def list_artifacts(request: Request, sid: str) -> Dict[str, Any]:
    return {"artifacts": _store(request).list_artifacts(sid)}


# -- synchronous generate (+persist); back-compat for simple clients ------
@app.post("/api/graph")
async def graph_sync(request: Request, body: GraphRequest) -> JSONResponse:
    store = _store(request)
    body = _cap_nodes(body)
    session = store.create_session(f"Quick · {body.input_text[:48]}")
    run_id = store.create_run(session["id"], body.input_text, body.model_dump(mode="json"))
    store.start_run(run_id)
    start = time.time()
    try:
        payload = generate_graph(body)
    except GraphServiceError as exc:
        store.finish_run(run_id, status="failed", error=str(exc),
                         latency_ms=int((time.time() - start) * 1000))
        raise HTTPException(502, f"graph generation failed: {exc}")
    latency = int((time.time() - start) * 1000)
    gid = store.save_graph(session["id"], run_id, payload)
    store.create_artifact(session["id"], gid, "graph", f"Graph · {body.input_text[:48]}",
                          {"nodes": payload["metrics"]["total_nodes"]})
    store.finish_run(run_id, status="succeeded", engine=payload.get("engine"),
                     graph_id=gid, latency_ms=latency)
    payload["session_id"] = session["id"]
    payload["run_id"] = run_id
    payload["graph_id"] = gid
    return JSONResponse(payload)


# -- SPA ------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    index_file = _FRONTEND_DIR / "index.html"
    if not index_file.exists():  # pragma: no cover
        raise HTTPException(404, "frontend not built")
    return FileResponse(index_file)


if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
