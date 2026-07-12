"""
app/services/hf_cache.py
─────────────────────────────────────────────────────────────────────────────
Flat-file read cache backed by HuggingFace Space persistent storage (/data).

Problem solved
──────────────
Every chat-switch previously triggered GET /sessions/{id}/messages → Supabase
round-trip (expensive, slow, Sydney pool).  This module caches each session's
full message list as a single JSON file so that reads are local-disk fast.

Write strategy
──────────────
Messages are still persisted to Supabase by the existing history.py path
(RunnableWithMessageHistory calls _DBMessageHistory.add_message on every turn).
The HF cache is a *read* cache only — it is kept warm by appending to the local
JSON file after every completed AI response (called from rag.py).

Folder layout  ($HF_CACHE_DIR defaults to /data/chat_history)
──────────────────────────────────────────────────────────────
  /data/chat_history/
  ├── auth_<8hexChars>/        ← authenticated user (first 8 chars of users.id UUID)
  │   ├── index.json           ← [{ id, title, created_at }] — titles only, no messages
  │   ├── session_<8hex>.json  ← full message array for one session
  │   └── ...
  └── anon_<8hexChars>/        ← anonymous / guest user
      ├── index.json
      └── session_<8hex>.json

Session file schema
───────────────────
{
  "session_id": "<full UUID>",
  "user_id":    "<full UUID>",
  "title":      "My Chat",
  "messages": [
    { "id": 1, "role": "user",      "content": "...", "created_at": "..." },
    { "id": 2, "role": "assistant", "content": "...", "created_at": "..." }
  ],
  "last_updated_at": "<ISO datetime>"
}

Atomicity
─────────
All file writes use NamedTemporaryFile + os.replace — atomic on POSIX,
safe against crashes / partial writes.

Public API
──────────
  get_user_dir(user_id, is_anon)                          -> Path
  get_session_path(user_id, session_id, is_anon)          -> Path
  read_session_cache(user_id, session_id)                 -> dict | None
  write_session_cache(user_id, session_id, title,
                      messages, user_id_full)              -> None
  append_messages_to_cache(user_id, session_id,
                           question, answer, title)        -> None
  update_index(user_id, session_id, title, is_anon,
               created_at)                                -> None
  read_index(user_id, is_anon)                            -> list
  delete_from_cache(user_id, session_id, is_anon)         -> None
  is_anonymous_user(user_id, email)                       -> bool
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("hf_cache")

# ── Config ────────────────────────────────────────────────────────────────────

def _cache_root() -> Path:
    """Return the root cache directory, creating it if necessary."""
    root = Path(os.getenv("HF_CACHE_DIR", "/data/chat_history"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_anonymous_user(user_id: str, email: str = "") -> bool:
    """
    Determine whether this user is anonymous.
    Anonymous users are created with the pattern: anon-<uuid>@thotqen.internal
    (see ensure_user calls in routes.py / rag.py).
    """
    return "@thotqen.internal" in email or email == ""


def _user_prefix(user_id: str, is_anon: bool) -> str:
    """
    Short human-readable folder prefix.
    Uses first 8 hex chars of the UUID (after stripping dashes).
    """
    short = user_id.replace("-", "")[:8]
    return f"{'anon' if is_anon else 'auth'}_{short}"


def _session_filename(session_id: str) -> str:
    short = session_id.replace("-", "")[:8]
    return f"session_{short}.json"


def _atomic_write(path: Path, data: Any) -> None:
    """
    Write *data* as JSON to *path* atomically using a temp file + os.replace.
    This guarantees the file is never left in a partially-written state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file if something goes wrong
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_read(path: Path) -> Any | None:
    """Read and parse JSON from *path*; returns None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def get_user_dir(user_id: str, is_anon: bool = False) -> Path:
    """Return (and create) the per-user cache directory."""
    d = _cache_root() / _user_prefix(user_id, is_anon)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_session_path(user_id: str, session_id: str, is_anon: bool = False) -> Path:
    """Return the path to a session's JSON cache file."""
    return get_user_dir(user_id, is_anon) / _session_filename(session_id)


# ── Session cache ─────────────────────────────────────────────────────────────

def read_session_cache(user_id: str, session_id: str) -> dict | None:
    """
    Try both auth_ and anon_ prefixes.
    Returns the parsed session dict or None on a cold miss.
    """
    for is_anon in (False, True):
        path = get_session_path(user_id, session_id, is_anon)
        data = _safe_read(path)
        if data is not None:
            logger.debug("[HFCache] HIT  session=%s path=%s", session_id[:8], path)
            return data

    logger.debug("[HFCache] MISS session=%s user=%s", session_id[:8], user_id[:8])
    return None


def write_session_cache(
    user_id: str,
    session_id: str,
    title: str | None,
    messages: list[dict],
    is_anon: bool = False,
) -> None:
    """
    Write (or overwrite) the full session JSON file.
    Called on cold-miss population from Supabase, or after session creation.
    """
    path = get_session_path(user_id, session_id, is_anon)
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "title": title or "New Chat",
        "messages": messages,
        "last_updated_at": _now_iso(),
    }
    try:
        _atomic_write(path, payload)
        logger.info("[HFCache] WRITE session=%s (%d msgs)", session_id[:8], len(messages))
    except Exception as exc:
        logger.warning("[HFCache] Write failed session=%s: %s", session_id[:8], exc)


def append_messages_to_cache(
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
    title: str | None = None,
) -> None:
    """
    Append a user+assistant message pair to the cached session file.
    Called from rag.py after every completed AI response (non-blocking thread).

    If the file doesn't exist yet, it is created with just these two messages.
    IDs are assigned as len(existing)+1 / len(existing)+2 (local counter —
    the real BIGSERIAL id from Supabase is set on cold-miss repopulation).
    """
    # Try to find the existing file (auth or anon dir)
    existing: dict | None = None
    found_path: Path | None = None

    for is_anon in (False, True):
        p = get_session_path(user_id, session_id, is_anon)
        data = _safe_read(p)
        if data is not None:
            existing = data
            found_path = p
            break

    if existing is None:
        # Cold path — create minimal file; is_anon guessed as False (corrected later)
        existing = {
            "session_id": session_id,
            "user_id": user_id,
            "title": title or "New Chat",
            "messages": [],
            "last_updated_at": _now_iso(),
        }
        found_path = get_session_path(user_id, session_id, is_anon=False)

    msgs = existing.get("messages", [])
    base_id = len(msgs)
    now = _now_iso()

    msgs.append({
        "id": base_id + 1,
        "role": "user",
        "content": question,
        "created_at": now,
    })
    msgs.append({
        "id": base_id + 2,
        "role": "assistant",
        "content": answer,
        "created_at": now,
    })

    existing["messages"] = msgs
    existing["last_updated_at"] = now
    if title:
        existing["title"] = title

    try:
        _atomic_write(found_path, existing)
        logger.info(
            "[HFCache] APPEND session=%s total_msgs=%d",
            session_id[:8], len(msgs),
        )
    except Exception as exc:
        logger.warning("[HFCache] Append failed session=%s: %s", session_id[:8], exc)


# ── Index (session list) ──────────────────────────────────────────────────────

def read_index(user_id: str, is_anon: bool = False) -> list[dict]:
    """
    Return the list of { id, title, created_at } dicts for this user.
    Falls back to [] on miss (triggers DB call in caller).
    """
    path = get_user_dir(user_id, is_anon) / "index.json"
    data = _safe_read(path)
    if isinstance(data, list):
        return data
    return []


def update_index(
    user_id: str,
    session_id: str,
    title: str | None,
    is_anon: bool = False,
    created_at: str | None = None,
) -> None:
    """
    Upsert a session entry in index.json.
    If the session already exists in the index it is updated in-place,
    otherwise it is prepended (newest first).
    """
    path = get_user_dir(user_id, is_anon) / "index.json"
    index: list[dict] = read_index(user_id, is_anon)

    entry = {
        "id": session_id,
        "title": title or "New Chat",
        "created_at": created_at or _now_iso(),
    }

    # Upsert
    for i, item in enumerate(index):
        if item.get("id") == session_id:
            index[i] = entry
            break
    else:
        index.insert(0, entry)  # Prepend so newest is first

    try:
        _atomic_write(path, index)
        logger.debug("[HFCache] INDEX updated user=%s sessions=%d", user_id[:8], len(index))
    except Exception as exc:
        logger.warning("[HFCache] Index update failed user=%s: %s", user_id[:8], exc)


def delete_from_cache(
    user_id: str,
    session_id: str,
    is_anon: bool = False,
) -> None:
    """
    Remove a session's JSON file and clean it from index.json.
    Tries both auth_ and anon_ dirs so it works regardless of prefix.
    """
    # Remove session file (try both dirs)
    for _is_anon in (False, True):
        p = get_session_path(user_id, session_id, _is_anon)
        try:
            p.unlink(missing_ok=True)
            logger.info("[HFCache] DELETE session=%s", session_id[:8])
        except OSError as exc:
            logger.warning("[HFCache] Delete file failed: %s", exc)

    # Remove from index (try both dirs)
    for _is_anon in (False, True):
        idx_path = get_user_dir(user_id, _is_anon) / "index.json"
        index = _safe_read(idx_path)
        if not isinstance(index, list):
            continue
        new_index = [item for item in index if item.get("id") != session_id]
        if len(new_index) != len(index):
            try:
                _atomic_write(idx_path, new_index)
                logger.debug("[HFCache] INDEX removed session=%s", session_id[:8])
            except Exception as exc:
                logger.warning("[HFCache] Index remove failed: %s", exc)
