#!/usr/bin/env python3
"""Build a standalone, offline HTML preview of the Graphos workspace.

Generates a real reinforced graph from the engine, inlines the CSS + JS, bakes
the dataset into the page, and writes an immutably-named (ISO-timestamped)
file under ``dist/``. The file opens with no server and no network.

Usage:
    python tools/build_preview.py "churn drivers across our Q4 accounts"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphos.service import GraphRequest, generate_graph  # noqa: E402

FRONTEND = ROOT / "frontend"
DIST = ROOT / "dist"


def build(query: str, preset: str = "churn_intelligence", max_nodes: int = 18) -> Path:
    presets = {
        "churn_intelligence": dict(generation_mode="analytical", clustering_method="causal",
                                   domain_focus="business", enable_risk_analysis=True),
        "creative_brainstorm": dict(generation_mode="creative", clustering_method="thematic",
                                    domain_focus="creative", creativity_level=0.9),
        "technical_planning": dict(generation_mode="practical", clustering_method="functional",
                                   domain_focus="technology", min_confidence=0.6),
    }
    req = GraphRequest(input_text=query, max_nodes=max_nodes, **presets.get(preset, {}))
    data = generate_graph(req)

    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    css = (FRONTEND / "graphos.css").read_text(encoding="utf-8")
    js = (FRONTEND / "graphos.js").read_text(encoding="utf-8")

    baked = json.dumps({"query": query, "data": data}, default=str)
    data_script = (
        '<script>window.__GRAPHOS_DATA__ = ' + baked + ";</script>\n"
        "<script>" + js + "</script>"
    )

    html = html.replace(
        '<link rel="stylesheet" href="/static/graphos.css">',
        "<style>\n" + css + "\n</style>",
    )
    html = html.replace('<script src="/static/graphos.js"></script>', data_script)

    DIST.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = DIST / f"graphos_preview_{stamp}.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "churn drivers across our Q4 accounts"
    path = build(q)
    print(f"✅ preview written: {path}")
    print(f"   size: {path.stat().st_size // 1024} KB")
