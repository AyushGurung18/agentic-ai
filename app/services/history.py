"""
app/services/history.py
────────────────────────
Chat message persistence using the normalized `messages` table.

Replaces the old langchain_postgres.PostgresChatMessageHistory (JSONB blob)
with direct psycopg reads/writes so we get:
  • Fast indexed queries (no JSON parsing)
  • Proper session → message FK relationship
  • User-scoped history via session ownership

Public API
──────────
    get_messages(session_id, limit=10) -> list[BaseMessage]
    save_message(session_id, role, content) -> None
    get_session_history(session_id) -> callable usable by RunnableWithMessageHistory
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from app.db.database import get_conn


# ── Low-level helpers ─────────────────────────────────────────────────────────

def get_messages(session_id: str, limit: int = 10) -> list[BaseMessage]:
    """
    Fetch the last `limit` messages for a session, ordered oldest→newest.
    Returns LangChain BaseMessage objects ready for prompt injection.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM   messages
                WHERE  session_id = %s::uuid
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()

    # Rows come back newest-first; reverse so oldest is first in the prompt
    rows = list(reversed(rows))

    result: list[BaseMessage] = []
    for role, content in rows:
        if role == "user":
            result.append(HumanMessage(content=content))
        else:
            result.append(AIMessage(content=content))
    return result


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Persist a single message to the `messages` table.
    `role` must be 'user' or 'assistant'.
    """
    assert role in ("user", "assistant"), f"Invalid role: {role!r}"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (%s::uuid, %s, %s)
                """,
                (session_id, role, content),
            )
        conn.commit()


# ── LangChain-compatible history factory ──────────────────────────────────────

class _DBMessageHistory(BaseChatMessageHistory):
    """
    Minimal in-memory + DB-backed message history object.
    Compatible with RunnableWithMessageHistory's get_session_history protocol.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        return get_messages(self.session_id)

    def add_user_message(self, message: str) -> None:
        save_message(self.session_id, "user", message)

    def add_ai_message(self, message: str) -> None:
        save_message(self.session_id, "assistant", message)

    def add_message(self, message: BaseMessage) -> None:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        save_message(self.session_id, role, message.content)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages to the history."""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE session_id = %s::uuid",
                    (self.session_id,),
                )
            conn.commit()


def get_chat_history(session_id: str) -> _DBMessageHistory:
    """
    Factory consumed by RunnableWithMessageHistory.
    Example:
        RunnableWithMessageHistory(
            chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    """
    return _DBMessageHistory(session_id)
