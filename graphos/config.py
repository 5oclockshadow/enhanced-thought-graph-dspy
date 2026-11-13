"""Runtime configuration for Graphos, sourced from the environment.

Centralising config (instead of scattering ``os.environ`` reads) is part of
making the service operable: every knob is here, typed, with a safe default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass
class Settings:
    version: str = "1.0.0"
    db_path: Path = field(default_factory=lambda: Path(
        os.environ.get("GRAPHOS_DB_PATH", str(_REPO_ROOT / "graphos_data" / "graphos.db"))))
    live_disabled: bool = field(default_factory=lambda: _env_bool("GRAPHOS_DISABLE_LIVE"))
    live_timeout_s: int = field(default_factory=lambda: _env_int("GRAPHOS_LIVE_TIMEOUT", 180))
    max_nodes_cap: int = field(default_factory=lambda: _env_int("GRAPHOS_MAX_NODES", 40))
    cors_origins: List[str] = field(default_factory=lambda: _env_list("GRAPHOS_CORS_ORIGINS", ["*"]))
    job_workers: int = field(default_factory=lambda: _env_int("GRAPHOS_JOB_WORKERS", 2))
    request_log: bool = field(default_factory=lambda: _env_bool("GRAPHOS_REQUEST_LOG", True))

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
