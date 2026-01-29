"""Pydantic models for web API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# User models
# ============================================================================

class User(BaseModel):
    """User model."""

    id: str
    username: str
    display_name: str | None = None
    created_at: datetime


class UserListResponse(BaseModel):
    """Response for listing users."""

    users: list[User]


# ============================================================================
# Auth models
# ============================================================================

class LoginRequest(BaseModel):
    """Login request body."""

    username: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Login response."""

    user: User
    session_id: str


class WhoamiResponse(BaseModel):
    """Response for whoami endpoint."""

    user: User


# ============================================================================
# Session models (internal)
# ============================================================================

class Session(BaseModel):
    """Browser session model."""

    session_id: str
    user_id: str
    created_at: datetime
    last_seen_at: datetime
    revoked_at: datetime | None = None
    user_agent: str | None = None
    ip: str | None = None


# ============================================================================
# Conversation models
# ============================================================================

class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""

    id: str
    title: str | None = None
    created_at: datetime
    last_message_at: datetime | None = None


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""

    conversations: list[ConversationSummary]


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    title: str | None = None


class CreateConversationResponse(BaseModel):
    """Response after creating a conversation."""

    id: str
    title: str | None = None
    created_at: datetime


# ============================================================================
# Message models
# ============================================================================

class RecipeCardMeta(BaseModel):
    """Recipe card metadata in message meta_json."""

    recipe_id: str
    title: str
    rating_avg: float | None = None
    time_total: int | None = None
    key_ingredients: list[str] = Field(default_factory=list)
    one_sentence_summary: str | None = None
    why_match: str | None = None


class MessageMeta(BaseModel):
    """Message metadata structure."""

    recipe_cards: list[RecipeCardMeta] = Field(default_factory=list)
    intent: str | None = None
    error: dict[str, str] | None = None


class Message(BaseModel):
    """Chat message model."""

    id: str
    conversation_id: str
    user_id: str
    role: str  # 'user' or 'assistant'
    content: str
    meta: MessageMeta = Field(default_factory=MessageMeta)
    created_at: datetime


class MessageListResponse(BaseModel):
    """Response for listing messages in a conversation."""

    conversation_id: str
    messages: list[Message]


# ============================================================================
# Chat models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request body."""

    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    conversation_id: str
    message_id: str
    content: str
    meta: MessageMeta


class StreamTokenEvent(BaseModel):
    """SSE token event data."""

    content: str


class StreamDoneEvent(BaseModel):
    """SSE done event data."""

    conversation_id: str
    message_id: str
    meta: MessageMeta


class StreamErrorEvent(BaseModel):
    """SSE error event data."""

    error: dict[str, str]


# ============================================================================
# Health models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
