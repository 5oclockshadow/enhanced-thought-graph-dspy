"""Graphos FastAPI application.

Serves the neon-graph web UI and the graph-intelligence API. Every handler
uses explicit fault isolation and returns typed responses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .service import PRESETS, GraphRequest, GraphServiceError, generate_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphos.app")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Graphos — Graph Intelligence Workspace",
    description="Dark, neon-graph workspace for 4D spatio-temporal thought graphs.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "service": "graphos", "version": __version__}


@app.get("/api/presets")
async def presets() -> Dict[str, Any]:
    return {"presets": PRESETS}


@app.post("/api/graph")
async def graph(request: GraphRequest) -> JSONResponse:
    """Generate a reinforced, traced thought graph."""
    try:
        payload = generate_graph(request)
        return JSONResponse(payload)
    except GraphServiceError as exc:
        logger.error("Service error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Graph generation failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


@app.get("/")
async def index() -> FileResponse:
    index_file = _FRONTEND_DIR / "index.html"
    if not index_file.exists():  # pragma: no cover - build-time guard
        raise HTTPException(status_code=404, detail="frontend not built")
    return FileResponse(index_file)


# Serve static assets (css/js) alongside the SPA.
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
