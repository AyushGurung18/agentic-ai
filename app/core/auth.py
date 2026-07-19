"""
app/core/auth.py
────────────────
FastAPI dependency that verifies Supabase JWTs and returns the caller's
user UUID.

How it works
────────────
  • Supabase's newer projects sign session JWTs asymmetrically (ES256, via
    rotating "JWT Signing Keys") rather than with a single shared HS256
    secret. We verify against Supabase's public JWKS endpoint, matching
    the token's `kid` header to the right key — this also means key
    rotation on Supabase's side is handled automatically, nothing to
    keep in sync manually.
  • Falls back to the legacy shared-secret HS256 path if a token's header
    says `alg: HS256` (older Supabase projects still on that system).
  • Both real users AND anonymous users get a valid JWT whose `sub` claim
    is a stable UUID — so they're treated identically in the DB.
  • If no Bearer token is supplied (local dev), we fall back to DEV_USER_ID.

Required .env keys
──────────────────
  SUPABASE_URL          → used to derive the JWKS endpoint
  SUPABASE_JWT_SECRET   → only needed for legacy HS256 projects
"""

import logging
import time
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError

from app.core.config import SUPABASE_JWT_SECRET, SUPABASE_URL, DEV_USER_ID

logger = logging.getLogger("auth")

_bearer = HTTPBearer(auto_error=False)

_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict = {"keys": [], "fetched_at": 0.0}


def _fetch_jwks(force: bool = False) -> list:
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured — cannot reach the JWKS endpoint.")

    now = time.time()
    if force or not _jwks_cache["keys"] or (now - _jwks_cache["fetched_at"]) > _JWKS_TTL_SECONDS:
        resp = httpx.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=10)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json().get("keys", [])
        _jwks_cache["fetched_at"] = now
    return _jwks_cache["keys"]


def _verify_with_jwks(token: str, kid: Optional[str], alg: str) -> dict:
    keys = _fetch_jwks()
    matching = next((k for k in keys if k.get("kid") == kid), None)
    if matching is None:
        # Key may have rotated since our cache was populated — refresh once.
        keys = _fetch_jwks(force=True)
        matching = next((k for k in keys if k.get("kid") == kid), None)
    if matching is None:
        raise JWTError(f"No JWKS key found for kid={kid!r}")

    return jwt.decode(
        token,
        matching,
        algorithms=[alg],
        audience="authenticated",
    )


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """
    Returns the authenticated Supabase user UUID.

    ─ Token valid   → sub claim (stable UUID, anon or real user)
    ─ No token      → DEV_USER_ID  (local dev fallback only)
    ─ Token invalid → HTTP 401
    """
    if credentials is None:
        # No token — allow in dev; in production the frontend always sends one.
        return DEV_USER_ID

    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "")

        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                return DEV_USER_ID
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        else:
            payload = _verify_with_jwks(token, header.get("kid"), alg or "ES256")

        return str(payload["sub"])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired — please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Network/config failure reaching Supabase's JWKS endpoint (bad
        # SUPABASE_URL, DNS hiccup, Supabase outage, etc.) — this is not
        # the caller's fault, so don't let it fall through as a raw 500.
        logger.error("[auth] JWKS verification failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify authentication right now. Please try again shortly.",
        )
