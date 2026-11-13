"""Lightweight auth boundary.

Guardian finding 2: every route was open. This adds an opt-in API-key gate: when
``GRAPHOS_API_KEY`` is set, write/model/code-exec endpoints require a matching
``X-API-Key`` header; when it is unset the service stays open for local
development. It is deliberately minimal — a real deployment swaps this dependency
for OIDC/JWT and resolves a tenant principal — but it gives one enforceable
default-deny seam that all mutating routes already depend on.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException

from .config import settings


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    """FastAPI dependency. No-op when no key is configured; else enforce it."""
    if not settings.api_key:
        return None
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return x_api_key


def principal(x_api_key: Optional[str] = Header(default=None),
              x_tenant_id: Optional[str] = Header(default=None)) -> str:
    """Resolve the owner/tenant for data scoping.

    Minimal model: the tenant is the X-Tenant-Id header when present, else the
    API key identifies the caller, else a shared 'public' owner for local use.
    """
    if x_tenant_id:
        return x_tenant_id[:64]
    if settings.api_key and x_api_key:
        return "key:" + x_api_key[:8]
    return "public"
