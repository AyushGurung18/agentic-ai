from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid


# ── Existing ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    context: list[str]


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    filename: str
    document_id: str


# ── Sessions ──────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str]
    title: Optional[str]
    created_at: datetime

class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]

class DeleteResponse(BaseModel):
    deleted: bool
    id: str


# ── Messages ──────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str          # 'user' | 'assistant'
    content: str
    created_at: datetime

class MessageListResponse(BaseModel):
    messages: list[MessageResponse]