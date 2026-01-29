"""Conversation management endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.app.logging_config import get_logger
from src.web.dependencies import ConversationServiceDep, CurrentUser
from src.web.models import (
    ConversationListResponse,
    ConversationSummary,
    CreateConversationRequest,
    CreateConversationResponse,
    MessageListResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
    limit: int = 50,
) -> ConversationListResponse:
    """List user's conversations, sorted by last message time (newest first).

    Only returns non-archived conversations.
    """
    conversations = conversation_service.list_for_user(user.id, limit=limit)
    return ConversationListResponse(conversations=conversations)


@router.post("", response_model=CreateConversationResponse)
async def create_conversation(
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
    body: CreateConversationRequest | None = None,
) -> CreateConversationResponse:
    """Create a new conversation."""
    title = body.title if body else None
    conv_id = conversation_service.create(user.id, title=title)

    # Get the created conversation
    conv = conversation_service.get(conv_id, user.id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )

    logger.info(
        "Created conversation",
        conversation_id=conv_id[:8],
        user_id=user.id[:8]
    )

    return CreateConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at
    )


@router.get("/{conversation_id}", response_model=ConversationSummary)
async def get_conversation(
    conversation_id: str,
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
) -> ConversationSummary:
    """Get a specific conversation."""
    conv = conversation_service.get(conversation_id, user.id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conv


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def get_messages(
    conversation_id: str,
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
    limit: int = 100,
    before_id: str | None = None,
) -> MessageListResponse:
    """Get messages for a conversation.

    Returns messages oldest-first for display order.
    """
    # Verify user owns conversation
    conv = conversation_service.get(conversation_id, user.id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    messages = conversation_service.get_messages(
        conversation_id=conversation_id,
        user_id=user.id,
        limit=limit,
        before_id=before_id
    )

    return MessageListResponse(
        conversation_id=conversation_id,
        messages=messages
    )


@router.post("/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
) -> dict:
    """Archive (soft-delete) a conversation."""
    success = conversation_service.archive(conversation_id, user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    logger.info(
        "Archived conversation",
        conversation_id=conversation_id[:8],
        user_id=user.id[:8]
    )

    return {"status": "archived"}


@router.patch("/{conversation_id}/title")
async def update_title(
    conversation_id: str,
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
    title: str,
) -> dict:
    """Update conversation title."""
    success = conversation_service.update_title(conversation_id, user.id, title)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return {"status": "updated", "title": title}
