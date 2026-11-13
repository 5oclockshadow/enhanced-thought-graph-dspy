"""Graph Foundry server API — the three backend roadmap items.

* /api/foundry/ingest   — parse the connected repo's real source (via the
  Maximillian graph-intelligence AST pass) and surface its functions as an
  operator library; also report which named libraries are importable. (Item 1)
* /api/foundry/propose  — ask the environment's model to write a NEW task
  decoder, validate it by actually running an evolutionary search on it
  (foundry/validate_task.js), and return it only if it measurably works.
  The self-extending skill set, with a safety gate. (Item 2)
* /api/foundry/portfolio (GET/POST) — persist shipped engine specs so a
  synthesized harness becomes a launchable, durable engine. (Item 3)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("graphos.foundry")
router = APIRouter(prefix="/api/foundry", tags=["foundry"])

_REPO = Path(__file__).resolve().parent.parent
_FOUNDRY_DIR = _REPO / "foundry"
_PORTFOLIO = _REPO / "graphos_data" / "foundry_portfolio.json"
_LOCK = threading.Lock()
_NAMED_LIBS = ["gym", "gymnasium", "deap", "dspy", "metagpt", "mcp", "networkx"]


# ---------------- item 1: ingest real source -------------------------------
@router.get("/ingest")
def ingest() -> Dict[str, Any]:
    """Ingest the connected repo's real Python source into an operator library."""
    try:
        from maximillian.graph_intel import GraphIntel
        from maximillian.auditor import discover_python_files
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"graph-intel unavailable: {exc}")

    files = discover_python_files(_REPO)
    gi = GraphIntel().build(_REPO, files)
    summary = gi.summary()
    operators: List[Dict[str, Any]] = []
    for c in summary.top_central[:14]:
        node = c.get("node", "")
        kind = gi.g.nodes.get(node, {}).get("kind", "symbol") if node in gi.g else "symbol"
        if kind in ("function", "class"):
            operators.append({"symbol": node, "kind": kind, "pagerank": c.get("pagerank"),
                              "blast_radius": len(gi.blast_radius(node.split(".")[-1]))})
    libs = {name: (importlib.util.find_spec(name) is not None) for name in _NAMED_LIBS}
    return {
        "repo": _REPO.name, "files": len(files),
        "graph": {"nodes": summary.nodes, "edges": summary.edges, "kinds": summary.node_kinds},
        "operators": operators,
        "named_libraries": libs,
        "note": "Named GA/RL/agent libraries are modelled in-browser; this catalog "
                "surfaces the repo's own functions as operators to widen the library.",
    }


# ---------------- item 3: portfolio ----------------------------------------
class EngineSpecIn(BaseModel):
    engine: str
    task: str
    fitness: float
    genome: List[float]
    source: str = ""
    signature: Optional[Dict[str, Any]] = None
    program_graph: Optional[Dict[str, Any]] = None


def _load_portfolio() -> List[Dict[str, Any]]:
    if not _PORTFOLIO.exists():
        return []
    try:
        return json.loads(_PORTFOLIO.read_text(encoding="utf-8"))
    except Exception:
        return []


@router.get("/portfolio")
def list_portfolio() -> Dict[str, Any]:
    items = sorted(_load_portfolio(), key=lambda e: e.get("fitness", 0), reverse=True)
    return {"engines": items[:100], "count": len(items)}


@router.post("/portfolio")
def ship_engine(spec: EngineSpecIn) -> Dict[str, Any]:
    with _LOCK:
        _PORTFOLIO.parent.mkdir(parents=True, exist_ok=True)
        items = _load_portfolio()
        rec = spec.model_dump()
        rec["id"] = f"eng_{len(items) + 1}_{spec.task}"
        items.append(rec)
        _PORTFOLIO.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return {"saved": True, "id": rec["id"], "count": len(items)}


# ---------------- item 2: self-extending propose ---------------------------
class ProposeIn(BaseModel):
    goal: str = Field(..., min_length=3, max_length=400)


_PROPOSE_SYS = (
    "You extend an in-browser evolutionary code-search engine by writing a NEW task. "
    "Output ONLY one JSON object, no prose, no code fence. Shape:\n"
    '{"name":"snake_case","label":"Human label (metric)","keywords":["..."],'
    '"genomeSize":<int 2..16>,"box":[<lo>,<hi>],'
    '"decode":"function(g){ const body=`...${g[0]}...`; return { src: `function fn(input){ return ...; }`, '
    'fn: new Function(\'input\',\'return ...;\') }; }",'
    '"cases":"function(rng){ /* use rng() in [0,1) */ return <casesObjectOrArray>; }",'
    '"score":"function(fn, cases){ /* run fn on cases */ return <number in [0,1]>; }"}\n'
    "Rules: pure math/array JS only; no imports, no I/O, no DOM. decode MUST return "
    "{src:string, fn:function} where src and fn implement the SAME logic from the genome. "
    "score MUST return a value in [0,1] that a better genome increases. Keep it runnable and small."
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@router.post("/propose")
def propose(body: ProposeIn) -> Dict[str, Any]:
    """Model writes a new task decoder; we validate it by running real search."""
    try:
        from maximillian.synthesizer import detect_backend
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"synthesizer unavailable: {exc}")
    backend = detect_backend()
    if backend is None:
        raise HTTPException(503, "no model available in the environment to propose a task")

    try:
        raw = backend.complete(f"Goal: {body.goal}\nWrite the task JSON.", _PROPOSE_SYS, timeout=150)
    except Exception as exc:
        raise HTTPException(502, f"model call failed: {exc}")
    spec = _extract_json(raw)
    if not spec:
        raise HTTPException(422, "model did not return a parseable task spec")

    # Validate by actually running an evolutionary search on the proposed task.
    try:
        proc = subprocess.run(
            ["node", str(_FOUNDRY_DIR / "validate_task.js")],
            input=json.dumps(spec), capture_output=True, text=True, timeout=30)
        result = json.loads(proc.stdout or "{}")
    except Exception as exc:
        raise HTTPException(500, f"validation harness error: {exc}")

    if not result.get("ok"):
        return {"accepted": False, "reason": result.get("error", "validation failed"),
                "backend": backend.name, "spec": spec}
    return {"accepted": True, "backend": backend.name, "validation": result, "task": spec}


# ---------------- self-improvement: beautify own UI ------------------------
_LAST_IMPROVE: Dict[str, Any] = {"status": "idle"}


@router.post("/self-improve")
def self_improve_trigger() -> Dict[str, Any]:
    """Kick off the self-beautify loop in the background (re-ingest → RAPTOR →
    Hebbian → model rewrites own CSS, gated). Poll GET for the result."""
    global _LAST_IMPROVE
    if _LAST_IMPROVE.get("status") == "running":
        return {"status": "running"}
    _LAST_IMPROVE = {"status": "running"}

    def _work():
        global _LAST_IMPROVE
        try:
            from . import self_improve
            rep = self_improve.run()
            _LAST_IMPROVE = {"status": "done", "backend": rep.backend, "target": rep.target,
                             "steps": rep.steps, "beautify": rep.beautify,
                             "hebbian_top": [h["module"] for h in rep.hebbian[:5]],
                             "elapsed_s": rep.elapsed_s}
        except Exception as exc:  # pragma: no cover
            _LAST_IMPROVE = {"status": "error", "error": str(exc)}

    threading.Thread(target=_work, daemon=True).start()
    return {"status": "running", "note": "self-improve started; poll GET /api/foundry/self-improve"}


@router.get("/self-improve")
def self_improve_status() -> Dict[str, Any]:
    return _LAST_IMPROVE
