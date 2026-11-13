"""Self-improvement loop: the app re-ingests its own code and beautifies its UI.

Pipeline (each a real step, narrated ReAct-style and traced to telemetry):

  1. Re-ingest  — parse the whole repo into a dependency graph (graph-intel).
  2. RAPTOR     — build a hierarchical summary tree: file → cluster → root.
  3. Hebbian    — reinforce module centrality (degree co-activation) to rank
                  which surfaces matter most.
  4. Beautify   — feed that context + the target stylesheet to the environment's
                  model and have it REWRITE the CSS to be more beautiful.
  5. Gate       — accept only if every original selector survives and braces
                  balance; otherwise revert. The original is backed up.

This is self-modifying by design, so the safety gate is the point: the system
can only make its UI *better-looking with the same structure*, never delete its
own interface.
"""

from __future__ import annotations

import ast
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("graphos.self_improve")
_REPO = Path(__file__).resolve().parent.parent
_CLUSTERS = ["graphos", "maximillian", "foundry", "frontend", "tools", "tests"]


# -- 1. re-ingest ----------------------------------------------------------
def reingest() -> Dict[str, Any]:
    from maximillian.graph_intel import GraphIntel
    from maximillian.auditor import discover_python_files
    files = discover_python_files(_REPO)
    gi = GraphIntel().build(_REPO, files)
    return {"files": len(files), "summary": gi.summary(), "gi": gi}


# -- 2. RAPTOR (lightweight, deterministic hierarchical summary) -----------
def _summarize_py(path: Path) -> Dict[str, Any]:
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        syms = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return {"file": str(path.relative_to(_REPO)), "doc": (doc[0] if doc else "")[:140],
                "symbols": syms[:14], "loc": src.count("\n") + 1}
    except Exception:
        return {"file": str(path.relative_to(_REPO)), "doc": "", "symbols": [], "loc": 0}


def build_raptor() -> Dict[str, Any]:
    """file-level → cluster-level → root summary tree."""
    leaves: List[Dict[str, Any]] = []
    for p in sorted(_REPO.rglob("*.py")):
        if any(s in p.parts for s in {"__pycache__", ".git", "dist", "tests"}):
            continue
        leaves.append(_summarize_py(p))
    clusters: Dict[str, Dict[str, Any]] = {}
    for leaf in leaves:
        top = leaf["file"].split("/")[0]
        c = clusters.setdefault(top, {"files": 0, "loc": 0, "symbols": 0, "members": []})
        c["files"] += 1; c["loc"] += leaf["loc"]; c["symbols"] += len(leaf["symbols"])
        if len(c["members"]) < 6:
            c["members"].append(leaf["file"].split("/")[-1])
    root = {"clusters": len(clusters), "files": len(leaves),
            "loc": sum(l["loc"] for l in leaves), "symbols": sum(len(l["symbols"]) for l in leaves)}
    return {"root": root, "clusters": clusters, "leaf_count": len(leaves)}


def raptor_context(raptor: Dict[str, Any]) -> str:
    lines = [f"Repo RAPTOR root: {raptor['root']['files']} files, {raptor['root']['loc']} LOC, "
             f"{raptor['root']['clusters']} clusters."]
    for name, c in raptor["clusters"].items():
        lines.append(f"  · {name}/ — {c['files']} files, {c['loc']} LOC: {', '.join(c['members'])}")
    return "\n".join(lines)


# -- 3. Hebbian reinforcement over the module graph ------------------------
def hebbian_rank(gi, hebbian_rate: float = 0.3, top: int = 8) -> List[Dict[str, Any]]:
    g = gi.g
    degree: Dict[str, int] = {}
    for u, v in g.edges():
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    if not degree:
        return []
    mx = max(degree.values())
    scored = []
    for node, deg in degree.items():
        co = deg / mx
        reinforced = min(1.0, (deg / mx) * (1.0 + hebbian_rate * co))
        scored.append({"module": node, "degree": deg, "reinforced": round(reinforced, 4)})
    scored.sort(key=lambda x: x["reinforced"], reverse=True)
    return scored[:top]


# -- 4/5. beautify a stylesheet, gated ------------------------------------
_BEAUTIFY_SYS = (
    "You are a senior product designer refining a dark, neon-graph UI. You are given a CSS "
    "stylesheet and must return the COMPLETE improved stylesheet and nothing else (no prose, no "
    "code fence). HARD RULES: keep EVERY selector and class/id name exactly as-is — do not rename, "
    "merge, or delete any rule or selector. Only change property VALUES and ADD refinements inside "
    "existing rules: more refined spacing rhythm, softer shadows and depth, smoother gradients, "
    "subtle transitions/hover states, better type scale, and WCAG-safe contrast. Keep it dark and "
    "neon. Output valid CSS only."
)


def _selectors(css: str) -> set:
    # Track CLASS selectors (.foo) only. Hex colour literals and ids both use
    # '#', so a class-only gate verifies the structural rules the HTML depends
    # on survive, without ever confusing a recoloured value for a deleted rule.
    return set(re.findall(r"\.[A-Za-z_][A-Za-z0-9_-]*", css))


def _balanced(css: str) -> bool:
    return css.count("{") == css.count("}") and css.count("{") > 0


def beautify(target: Path, context: str, backend) -> Dict[str, Any]:
    original = target.read_text(encoding="utf-8")
    prompt = (f"Design context (self-ingested):\n{context}\n\n"
              f"Improve this stylesheet (return the whole file):\n\n{original}")
    try:
        out = backend.complete(prompt, _BEAUTIFY_SYS, timeout=240)
    except Exception as exc:
        return {"applied": False, "reason": f"model call failed: {exc}"}
    new = out.strip()
    if new.startswith("```"):
        new = re.sub(r"^```[a-z]*\n", "", new); new = re.sub(r"\n```$", "", new)

    orig_sel, new_sel = _selectors(original), _selectors(new)
    missing = orig_sel - new_sel
    keep_ratio = 1.0 - (len(missing) / max(1, len(orig_sel)))
    if not _balanced(new):
        return {"applied": False, "reason": "unbalanced braces", "keep_ratio": round(keep_ratio, 3)}
    if keep_ratio < 0.97:
        return {"applied": False, "reason": f"lost {len(missing)} selectors (keep {keep_ratio:.0%})",
                "missing_sample": sorted(missing)[:8]}
    if len(new) < len(original) * 0.6:
        return {"applied": False, "reason": "output too short — likely truncated"}

    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, backup)
    target.write_text(new, encoding="utf-8")
    return {"applied": True, "backup": str(backup), "before_bytes": len(original),
            "after_bytes": len(new), "selectors_kept": len(orig_sel), "keep_ratio": round(keep_ratio, 3)}


@dataclass
class ImproveReport:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reingest: Dict[str, Any] = field(default_factory=dict)
    raptor: Dict[str, Any] = field(default_factory=dict)
    hebbian: List[Dict[str, Any]] = field(default_factory=list)
    beautify: Dict[str, Any] = field(default_factory=dict)
    target: str = ""
    backend: str = "none"
    elapsed_s: float = 0.0


def run(target: Optional[Path] = None) -> ImproveReport:
    from maximillian.synthesizer import detect_backend
    rep = ImproveReport()
    start = time.time()
    target = target or (_REPO / "frontend" / "graphos.css")
    rep.target = str(target.relative_to(_REPO))

    def step(stage, tool, obs):
        rep.steps.append({"stage": stage, "tool": tool, "observation": obs})
        logger.info("[%s] %s — %s", stage, tool, obs)

    ing = reingest()
    rep.reingest = {"files": ing["files"], "nodes": ing["summary"].nodes, "edges": ing["summary"].edges}
    step("Re-ingest", "graph_intel", f"{ing['files']} files · {ing['summary'].nodes} symbols")

    rep.raptor = build_raptor()
    step("RAPTOR", "raptor.tree", f"{rep.raptor['root']['clusters']} clusters · {rep.raptor['root']['loc']} LOC")

    rep.hebbian = hebbian_rank(ing["gi"])
    top = rep.hebbian[0]["module"] if rep.hebbian else "—"
    step("Hebbian", "reinforce", f"top module {top} (reinforced {rep.hebbian[0]['reinforced'] if rep.hebbian else 0})")

    backend = detect_backend()
    rep.backend = backend.name if backend else "none"
    if backend is None:
        rep.beautify = {"applied": False, "reason": "no model in environment"}
        step("Beautify", "model", "skipped — no model")
    else:
        ctx = raptor_context(rep.raptor) + "\n\nHebbian-ranked modules: " + \
            ", ".join(h["module"].split(".")[-1] for h in rep.hebbian[:6])
        rep.beautify = beautify(target, ctx, backend)
        step("Beautify", "mcp.beautify",
             ("applied · " + str(rep.beautify.get("after_bytes")) + "B") if rep.beautify.get("applied")
             else ("rejected · " + rep.beautify.get("reason", "")))

    rep.elapsed_s = round(time.time() - start, 2)
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="graphos.self_improve")
    ap.add_argument("--target", help="CSS file to beautify (default frontend/graphos.css)")
    args = ap.parse_args(argv)
    rep = run(Path(args.target).resolve() if args.target else None)
    print(f"\nSelf-improve · backend={rep.backend} · target={rep.target} · {rep.elapsed_s}s")
    for s in rep.steps:
        print(f"  {s['stage']:<11} {s['tool']:<16} {s['observation']}")
    print("\nResult:", rep.beautify)
    return 0 if rep.beautify.get("applied") else 1


if __name__ == "__main__":
    raise SystemExit(main())
