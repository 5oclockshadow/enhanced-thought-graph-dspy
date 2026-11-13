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
import os
import signal
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import backend as _be
from .config import settings
from .llm_parse import extract_json
from .security import require_api_key

logger = logging.getLogger("graphos.foundry")
router = APIRouter(prefix="/api/foundry", tags=["foundry"])


def _require_foundry() -> None:
    """Gate the code-execution / self-modifying endpoints behind a flag that is
    OFF by default (Guardian/Skeptic finding 1)."""
    if not settings.enable_foundry:
        raise HTTPException(403, "foundry code-execution endpoints are disabled "
                                 "(set GRAPHOS_ENABLE_FOUNDRY=1 to enable in a trusted environment)")

_REPO = Path(__file__).resolve().parent.parent
_FOUNDRY_DIR = _REPO / "foundry"
_PORTFOLIO = _REPO / "graphos_data" / "foundry_portfolio.json"
_LOCK = threading.RLock()
_NAMED_LIBS = ["gym", "gymnasium", "deap", "dspy", "metagpt", "mcp", "networkx"]


# ---------------- item 1: ingest real source -------------------------------
@router.get("/ingest")
def ingest() -> Dict[str, Any]:
    """Ingest the connected repo's real Python source into an operator library."""
    try:
        from .repo_ingest import ingest_repo
        files, gi, summary = ingest_repo(_REPO)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"graph-intel unavailable: {exc}")

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
    with _LOCK:
        try:
            data = json.loads(_PORTFOLIO.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            # Craftsman #2: a corrupt/transient read must NOT be silently
            # overwritten. Quarantine the bad file so the next write starts
            # fresh without destroying recoverable data.
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                _PORTFOLIO.rename(_PORTFOLIO.with_name(f"foundry_portfolio.corrupt-{stamp}.json"))
                logger.error("portfolio unreadable (%s); quarantined", exc)
            except OSError:
                pass
            return []


@router.get("/portfolio")
def list_portfolio() -> Dict[str, Any]:
    items = sorted(_load_portfolio(), key=lambda e: e.get("fitness", 0), reverse=True)
    return {"engines": items[:100], "count": len(items)}


@router.post("/portfolio", dependencies=[Depends(require_api_key)])
def ship_engine(spec: EngineSpecIn) -> Dict[str, Any]:
    with _LOCK:
        _PORTFOLIO.parent.mkdir(parents=True, exist_ok=True)
        items = _load_portfolio()
        rec = spec.model_dump()
        rec["id"] = f"eng_{uuid.uuid4().hex[:12]}_{spec.task}"
        items.append(rec)
        tmp = _PORTFOLIO.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        os.replace(tmp, _PORTFOLIO)  # atomic swap
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


def _validate_in_sandbox(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run the model-authored task through the Node validator in a hardened
    subprocess: scrubbed env (no inherited secrets), own process group so
    orphaned children die with it, bounded wall-clock (Skeptic finding 1)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "NODE_OPTIONS": "--max-old-space-size=128"}
    proc = subprocess.Popen(
        ["node", str(_FOUNDRY_DIR / "validate_task.js")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=str(_FOUNDRY_DIR), start_new_session=True)
    try:
        out, _ = proc.communicate(input=json.dumps(spec), timeout=25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole group
        except Exception:
            proc.kill()
        proc.communicate()
        raise HTTPException(504, "task validation timed out")
    try:
        return json.loads(out or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "validator produced no result"}


@router.post("/propose", dependencies=[Depends(require_api_key), Depends(_require_foundry)])
def propose(body: ProposeIn) -> Dict[str, Any]:
    """Model writes a new task decoder; we validate it by running real search."""
    backend = _be.get_backend()
    if backend is None:
        raise HTTPException(503, "no model available in the environment to propose a task")
    try:
        raw = backend.complete(f"Goal: {body.goal}\nWrite the task JSON.", _PROPOSE_SYS, timeout=150)
    except Exception as exc:
        raise HTTPException(502, f"model call failed: {exc}")
    spec = extract_json(raw)
    if not spec:
        raise HTTPException(422, "model did not return a parseable task spec")

    result = _validate_in_sandbox(spec)
    if not result.get("ok"):
        return {"accepted": False, "reason": result.get("error", "validation failed"),
                "backend": backend.name, "spec": spec}
    return {"accepted": True, "backend": backend.name, "validation": result, "task": spec}


# ---------------- self-improvement: beautify own UI ------------------------
_LAST_IMPROVE: Dict[str, Any] = {"status": "idle"}


@router.post("/self-improve", dependencies=[Depends(require_api_key), Depends(_require_foundry)])
def self_improve_trigger() -> Dict[str, Any]:
    """Kick off the self-beautify loop in the background (re-ingest → RAPTOR →
    Hebbian → model rewrites own CSS, gated). Poll GET for the result."""
    global _LAST_IMPROVE
    with _LOCK:  # atomic check-and-set (Skeptic finding 5: TOCTOU)
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
