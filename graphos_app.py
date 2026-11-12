#!/usr/bin/env python3
"""Entrypoint for the Graphos web application.

Run with:  python graphos_app.py   (or)   uvicorn graphos.app:app
"""

import os

import uvicorn

from graphos.app import app  # noqa: F401  (re-exported for `uvicorn graphos_app:app`)

if __name__ == "__main__":
    port = int(os.environ.get("GRAPHOS_PORT", "8080"))
    print("🕸  Graphos — Graph Intelligence Workspace")
    print("=" * 52)
    print(f"🚀  http://0.0.0.0:{port}   ·   API docs at /docs")
    print("=" * 52)
    uvicorn.run("graphos.app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
