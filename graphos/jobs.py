"""Background job manager for graph generation.

Live model calls take tens of seconds, so generation runs off the request
thread in a bounded worker pool. Each job is a persisted ``run`` row — status,
engine, latency and error — which doubles as the observability record. On
success the graph is stored and a default artifact is created.

In-process and single-node by design (sufficient here); the Store/run contract
is already shaped so this can be swapped for a real queue (Celery/RQ) later.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .service import GraphRequest, GraphServiceError, generate_graph
from .store import Store

logger = logging.getLogger("graphos.jobs")


class JobManager:
    def __init__(self, store: Store, workers: int = 2) -> None:
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="graphos-job")

    def submit(self, session_id: str, request: GraphRequest) -> str:
        run_id = self.store.create_run(session_id, request.input_text, request.model_dump(mode="json"))
        self._pool.submit(self._execute, run_id, session_id, request)
        return run_id

    def _execute(self, run_id: str, session_id: str, request: GraphRequest) -> None:
        self.store.start_run(run_id)
        start = time.time()
        try:
            payload = generate_graph(request)
            latency_ms = int((time.time() - start) * 1000)
            gid = self.store.save_graph(session_id, run_id, payload)
            tokens_est = (len(request.input_text) +
                          sum(len(t.get("content", "")) for t in payload.get("thoughts", []))) // 4
            self.store.create_artifact(
                session_id, gid, "graph", f"Graph · {request.input_text[:48]}",
                {"nodes": payload["metrics"]["total_nodes"],
                 "quality": payload["metrics"]["quality_score"],
                 "engine": payload.get("engine")})
            self.store.finish_run(run_id, status="succeeded", engine=payload.get("engine"),
                                  graph_id=gid, latency_ms=latency_ms, tokens_est=tokens_est)
            logger.info("run %s succeeded in %dms (engine=%s)", run_id, latency_ms, payload.get("engine"))
        except GraphServiceError as exc:
            self._fail(run_id, start, f"generation error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("run %s crashed", run_id)
            self._fail(run_id, start, f"unexpected error: {exc}")

    def _fail(self, run_id: str, start: float, error: str) -> None:
        self.store.finish_run(run_id, status="failed", error=error[:500],
                              latency_ms=int((time.time() - start) * 1000))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
