"""Preflight verification, Git automation and state snapshots (Stage 13).

Git operations are opt-in (``enable_git=False`` by default) so a demonstration
or CI run never mutates history unexpectedly; the orchestrator turns them on
only when asked. Preflight checks are always safe/read-only.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("maximillian.snapshotter")


def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def preflight(repo: Path, import_names: Optional[List[str]] = None) -> Dict[str, object]:
    """Read-only environment sanity checks."""
    checks: Dict[str, object] = {}

    # Clean working tree?
    if shutil.which("git"):
        status = _git(["status", "--porcelain"], repo)
        checks["git_clean"] = status.returncode == 0 and not status.stdout.strip()
        checks["git_dirty_files"] = len([l for l in status.stdout.splitlines() if l.strip()])
    else:
        checks["git_clean"] = None

    # Imports resolve?
    bad: List[str] = []
    for name in import_names or []:
        try:
            if importlib.util.find_spec(name) is None:
                bad.append(name)
        except Exception:
            bad.append(name)
    checks["imports_ok"] = not bad
    checks["imports_missing"] = bad

    # Disk headroom (best-effort).
    try:
        usage = shutil.disk_usage(repo)
        checks["disk_free_mb"] = usage.free // (1024 * 1024)
    except Exception:
        checks["disk_free_mb"] = None

    checks["passed"] = bool(checks.get("imports_ok"))
    return checks


def git_sha(repo: Path) -> Optional[str]:
    if not shutil.which("git"):
        return None
    proc = _git(["rev-parse", "HEAD"], repo)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_commit(repo: Path, message: str, paths: Optional[List[str]] = None) -> Optional[str]:
    """Stage and commit (opt-in). Returns the new SHA or None."""
    if not shutil.which("git"):
        return None
    _git(["add", *(paths or ["-A"])], repo)
    commit = _git(["commit", "-m", message], repo)
    if commit.returncode != 0:
        logger.info("Nothing committed: %s", commit.stdout.strip() or commit.stderr.strip())
        return git_sha(repo)
    return git_sha(repo)


def snapshot_tag(repo: Path, round_id: int) -> Optional[str]:
    """Create an immutable annotated tag for this round (opt-in)."""
    if not shutil.which("git"):
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = f"maximillian-r{round_id}-{ts}"
    proc = _git(["tag", "-a", tag, "-m", f"Maximillian round {round_id}"], repo)
    return tag if proc.returncode == 0 else None
