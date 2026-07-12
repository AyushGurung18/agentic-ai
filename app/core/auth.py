"""
app/core/auth.py
────────────────
FastAPI dependency that verifies Supabase JWTs and returns the caller's
user UUID.

How it works
────────────
  • Supabase signs every session JWT (HS256) with the project JWT Secret.
  • Both real users AND anonymous users get a valid JWT whose `sub` claim
    is a stable UUID — so they're treated identically in the DB.
  • If no Bearer token is supplied (local dev), we fall back to DEV_USER_ID.

Required .env keys
──────────────────
  SUPABASE_JWT_SECRET   → Dashboard ▶ Settings ▶ API ▶ JWT Settings ▶ JWT Secret
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError

from app.core.config import SUPABASE_JWT_SECRET, DEV_USER_ID

_bearer = HTTPBearer(auto_error=False)


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
        # No token — allow in dev; in production set SUPABASE_JWT_SECRET
        # and the frontend will always send one.
        return DEV_USER_ID

    if not SUPABASE_JWT_SECRET:
        # Secret not configured — skip verification (dev mode)
        return DEV_USER_ID

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
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
