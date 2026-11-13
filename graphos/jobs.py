"""Background job manager for graph generation.

Live model calls take tens of seconds, so generation runs off the request
thread in a bounded worker pool. Each job is a persisted ``run`` row — status,
engine, latency, error — which doubles as the observability record.

Hardened per the council audit: bounded queue (reject, don't buffer unbounded),
per-job wall-clock timeout, and a shared ``run_generation`` so the sync and
async paths cannot drift. In-process/single-node by design; the Store/run
contract is shaped so this can move to a durable broker later.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict

from .config import settings
from .service import GraphRequest, generate_graph, generate_graph_async
from .store import Store

logger = logging.getLogger("graphos.jobs")


class QueueFull(RuntimeError):
    """Raised when the in-flight job bound is reached."""


def _persist(store: Store, session_id: str, run_id: str, request: GraphRequest,
             payload: Dict[str, Any], start: float) -> Dict[str, Any]:
    latency_ms = int((time.time() - start) * 1000)
    gid = store.save_graph(session_id, run_id, payload)
    tokens_est = (len(request.input_text)
                  + sum(len(t.get("content", "")) for t in payload.get("thoughts", []))) // 4
    store.create_artifact(
        session_id, gid, "graph", f"Graph · {request.input_text[:48]}",
        {"nodes": payload["metrics"]["total_nodes"],
         "quality": payload["metrics"]["quality_score"], "engine": payload.get("engine")})
    store.finish_run(run_id, status="succeeded", engine=payload.get("engine"),
                     graph_id=gid, latency_ms=latency_ms, tokens_est=tokens_est)
    logger.info("run %s succeeded in %dms (engine=%s)", run_id, latency_ms, payload.get("engine"))
    payload["graph_id"], payload["run_id"], payload["session_id"] = gid, run_id, session_id
    return payload


def run_generation(store: Store, session_id: str, run_id: str, request: GraphRequest) -> Dict[str, Any]:
    """The single run state machine (sync): start → generate → persist → finish.
    One owner for the lifecycle, shared by the worker and the sync endpoint."""
    store.start_run(run_id)
    start = time.time()
    try:
        return _persist(store, session_id, run_id, request, generate_graph(request), start)
    except Exception as exc:
        store.finish_run(run_id, status="failed", error=str(exc)[:500],
                         latency_ms=int((time.time() - start) * 1000))
        logger.warning("run %s failed: %s", run_id, exc)
        raise


async def run_generation_async(store: Store, session_id: str, run_id: str,
                               request: GraphRequest) -> Dict[str, Any]:
    """Async lifecycle: the model call is awaited (non-blocking); the fast store
    writes happen on the loop. Used by the async /api/graph handler."""
    store.start_run(run_id)
    start = time.time()
    try:
        payload = await generate_graph_async(request)
        return _persist(store, session_id, run_id, request, payload, start)
    except Exception as exc:
        store.finish_run(run_id, status="failed", error=str(exc)[:500],
                         latency_ms=int((time.time() - start) * 1000))
        logger.warning("run %s failed: %s", run_id, exc)
        raise


class JobManager:
    def __init__(self, store: Store, workers: int = 2) -> None:
        self.store = store
        workers = max(1, workers)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graphos-job")
        self._watch = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graphos-watch")
        self._lock = threading.Lock()
        self._inflight = 0
        self._queue_max = settings.job_queue_max
        self._timeout = settings.job_timeout_s

    def submit(self, session_id: str, request: GraphRequest) -> str:
        with self._lock:
            if self._inflight >= self._queue_max:
                raise QueueFull(f"job queue full ({self._queue_max} in flight)")
            self._inflight += 1
        run_id = self.store.create_run(session_id, request.input_text, request.model_dump(mode="json"))
        self._pool.submit(self._execute, run_id, session_id, request)
        return run_id

    def _execute(self, run_id: str, session_id: str, request: GraphRequest) -> None:
        start = time.time()
        try:
            # Run generation under a hard wall-clock cap; a hung model call must
            # not hold a worker forever (Skeptic finding 3).
            fut = self._watch.submit(run_generation, self.store, session_id, run_id, request)
            try:
                fut.result(timeout=self._timeout)
            except FuturesTimeout:
                self.store.finish_run(run_id, status="failed", error="generation timed out",
                                      latency_ms=int((time.time() - start) * 1000))
                logger.warning("run %s timed out after %ss", run_id, self._timeout)
            except Exception:
                pass  # run already marked failed inside run_generation
        finally:
            with self._lock:
                self._inflight -= 1

    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._watch.shutdown(wait=False, cancel_futures=True)
