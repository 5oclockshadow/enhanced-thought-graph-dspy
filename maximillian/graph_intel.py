"""Graph intelligence: parse a Python codebase into a NetworkX DiGraph.

Nodes are modules / classes / functions; edges are IMPORTS / INHERITS / CALLS /
DEFINES. Centrality (PageRank, betweenness) ranks audit priority, and a
blast-radius query returns the upstream dependents of any symbol — the set of
nodes a change could break.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from .schemas import GraphSummary


class GraphIntel:
    def __init__(self) -> None:
        self.g = nx.DiGraph()
        self._module_by_path: Dict[str, str] = {}

    # -- construction ------------------------------------------------------
    def build(self, root: Path, files: List[Path]) -> "GraphIntel":
        """Build the dependency graph from the given Python files."""
        for path in files:
            try:
                self._add_module(root, path)
            except (SyntaxError, UnicodeDecodeError):
                # A syntactically broken file is still a node, just unparsed.
                mod = self._module_name(root, path)
                self.g.add_node(mod, kind="module", path=str(path), parsed=False)
        return self

    def _module_name(self, root: Path, path: Path) -> str:
        rel = path.relative_to(root).with_suffix("")
        return ".".join(rel.parts)

    def _add_module(self, root: Path, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        mod = self._module_name(root, path)
        self._module_by_path[str(path)] = mod
        self.g.add_node(mod, kind="module", path=str(path), parsed=True,
                        loc=source.count("\n") + 1)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in self._imported_modules(node):
                    self.g.add_edge(mod, name, kind="IMPORTS")
            elif isinstance(node, ast.ClassDef):
                qual = f"{mod}.{node.name}"
                self.g.add_node(qual, kind="class", path=str(path), line=node.lineno)
                self.g.add_edge(mod, qual, kind="DEFINES")
                for base in node.bases:
                    base_name = self._name_of(base)
                    if base_name:
                        self.g.add_edge(qual, base_name, kind="INHERITS")
            elif isinstance(node, ast.FunctionDef):
                qual = f"{mod}.{node.name}"
                self.g.add_node(qual, kind="function", path=str(path), line=node.lineno)
                self.g.add_edge(mod, qual, kind="DEFINES")
                for call in self._calls_in(node):
                    self.g.add_edge(qual, call, kind="CALLS")

    @staticmethod
    def _imported_modules(node: ast.AST) -> List[str]:
        out: List[str] = []
        if isinstance(node, ast.Import):
            out = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out = [node.module]
        return out

    @staticmethod
    def _name_of(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @staticmethod
    def _calls_in(func: ast.FunctionDef) -> List[str]:
        names: List[str] = []
        for n in ast.walk(func):
            if isinstance(n, ast.Call):
                name = GraphIntel._name_of(n.func)
                if name:
                    names.append(name)
        return names

    # -- analysis ----------------------------------------------------------
    def cycles(self) -> List[List[str]]:
        try:
            return [c for c in nx.simple_cycles(self.g) if len(c) > 1]
        except Exception:
            return []

    def centrality(self, top: int = 8) -> List[Dict[str, float]]:
        if self.g.number_of_nodes() == 0:
            return []
        try:
            pr = nx.pagerank(self.g, max_iter=200)
        except Exception:
            pr = {n: 0.0 for n in self.g.nodes}
        ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return [{"node": n, "pagerank": round(v, 5)} for n, v in ranked]

    def blast_radius(self, symbol: str) -> List[str]:
        """All nodes that (transitively) depend on ``symbol`` — change impact."""
        if symbol not in self.g:
            # Try a suffix match (bare symbol name).
            matches = [n for n in self.g.nodes if n == symbol or n.endswith("." + symbol)]
            if not matches:
                return []
            symbol = matches[0]
        try:
            ancestors: Set[str] = nx.ancestors(self.g, symbol)
        except Exception:
            ancestors = set()
        return sorted(ancestors)

    def summary(self) -> GraphSummary:
        node_kinds: Dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            k = data.get("kind", "unknown")
            node_kinds[k] = node_kinds.get(k, 0) + 1
        edge_kinds: Dict[str, int] = {}
        for _, _, data in self.g.edges(data=True):
            k = data.get("kind", "unknown")
            edge_kinds[k] = edge_kinds.get(k, 0) + 1
        return GraphSummary(
            nodes=self.g.number_of_nodes(),
            edges=self.g.number_of_edges(),
            node_kinds=node_kinds,
            edge_kinds=edge_kinds,
            top_central=self.centrality(),
            cycles=self.cycles(),
        )
