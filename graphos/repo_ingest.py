"""Shared repo-ingestion prelude.

Craftsman finding #3: `discover_python_files → GraphIntel().build → summary`
was duplicated in foundry_api.ingest and self_improve.reingest. One owner now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def ingest_repo(repo: Path) -> Tuple[List[Path], Any, Any]:
    """Return (python_files, GraphIntel, GraphSummary) for the repo."""
    from maximillian.auditor import discover_python_files
    from maximillian.graph_intel import GraphIntel
    files = discover_python_files(repo)
    gi = GraphIntel().build(repo, files)
    return files, gi, gi.summary()
