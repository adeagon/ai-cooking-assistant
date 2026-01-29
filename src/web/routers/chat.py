"""Chat endpoints with SSE streaming support."""

import json

from fastapi import APIRouter, HTTPException, status
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.app.logging_config import get_logger
from src.web.dependencies import (
    ChatServiceDep,
    ConversationServiceDep,
    CurrentUser,
)
from src.web.models import (
    ChatRequest,
    ChatResponse,
    MessageMeta,
    RecipeCardMeta,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    user: CurrentUser,
    chat_service: ChatServiceDep,
    conversation_service: ConversationServiceDep,
) -> EventSourceResponse:
    """Stream chat response as Server-Sent Events.

    This is the source of truth for chat processing.
    Non-streaming endpoint calls this internally.

    SSE Event Types:
    - token: {"content": "partial text"} - Streaming response tokens
    - cards: {"cards": [...]} - Recipe cards when available
    - done: {"conversation_id": "...", "message_id": "...", "meta": {...}}
    - error: {"error": {"message": "...", "code": "..."}}
    """

    async def event_generator():
        # Get or create conversation
        conv_id = body.conversation_id
        if not conv_id:
            conv_id = conversation_service.create(user.id)
            logger.info(
                "Created new conversation for chat",
                conversation_id=conv_id[:8],
                user_id=user.id[:8]
            )
        else:
            # Verify user owns conversation
            conv = conversation_service.get(conv_id, user.id)
            if not conv:
                yield ServerSentEvent(
                    event="error",
                    data=json.dumps({
                        "error": {
                            "message": "Conversation not found",
                            "code": "not_found"
                        }
                    })
                )
                return

        # Save user message first
        conversation_service.add_message(
            conversation_id=conv_id,
            user_id=user.id,
            role="user",
            content=body.message,
            meta={}
        )

        # Accumulate response for saving
        full_response = ""
        recipe_cards = []
        final_meta = {}
        had_error = False

        try:
            # Stream from chat service
            async for event_type, data in chat_service.stream_message(
                message=body.message,
                user=user
            ):
                if event_type == "token":
                    content = data.get("content", "")
                    full_response += content
                    yield ServerSentEvent(
                        event="token",
                        data=json.dumps({"content": content})
                    )

                elif event_type == "cards":
                    # Convert RecipeCard objects to dicts
                    cards = data.get("cards", [])
                    recipe_cards = [
                        {
                            "recipe_id": c.recipe_id,
                            "title": c.title,
                            "rating_avg": c.rating_avg,
                            "time_total": c.time_total,
                            "key_ingredients": c.key_ingredients,
                            "one_sentence_summary": c.one_sentence_summary,
                            "why_match": c.why_match,
                        }
                        for c in cards
                    ]
                    yield ServerSentEvent(
                        event="cards",
                        data=json.dumps({"cards": recipe_cards})
                    )

                elif event_type == "done":
                    final_meta = data.get("meta", {})

                elif event_type == "error":
                    had_error = True
                    error_data = data.get("error", {})
                    # Save error as assistant message
                    conversation_service.add_message(
                        conversation_id=conv_id,
                        user_id=user.id,
                        role="assistant",
                        content="[Error occurred]",
                        meta={"error": error_data}
                    )
                    yield ServerSentEvent(
                        event="error",
                        data=json.dumps({"error": error_data})
                    )
                    return

            # Save assistant message if we got a response
            if full_response and not had_error:
                msg_id = conversation_service.add_message(
                    conversation_id=conv_id,
                    user_id=user.id,
                    role="assistant",
                    content=full_response,
                    meta={"recipe_cards": recipe_cards}
                )

                yield ServerSentEvent(
                    event="done",
                    data=json.dumps({
                        "conversation_id": conv_id,
                        "message_id": msg_id,
                        "meta": {"recipe_cards": recipe_cards}
                    })
                )

                logger.info(
                    "Chat stream completed",
                    conversation_id=conv_id[:8],
                    message_id=msg_id[:8],
                    response_length=len(full_response),
                    card_count=len(recipe_cards)
                )

        except Exception as e:
            logger.exception("Error in chat stream", user_id=user.id[:8])
            # Save error message
            conversation_service.add_message(
                conversation_id=conv_id,
                user_id=user.id,
                role="assistant",
                content="[Error occurred]",
                meta={"error": {"message": str(e), "code": "stream_error"}}
            )
            yield ServerSentEvent(
                event="error",
                data=json.dumps({
                    "error": {"message": str(e), "code": "stream_error"}
                })
            )

    return EventSourceResponse(event_generator())


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUser,
    chat_service: ChatServiceDep,
    conversation_service: ConversationServiceDep,
) -> ChatResponse:
    """Non-streaming chat endpoint.

    Calls streaming internally and buffers output.
    For real-time experience, use /api/chat/stream instead.
    """
    # Get or create conversation
    conv_id = body.conversation_id
    if not conv_id:
        conv_id = conversation_service.create(user.id)
    else:
        # Verify user owns conversation
        conv = conversation_service.get(conv_id, user.id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

    # Save user message
    conversation_service.add_message(
        conversation_id=conv_id,
        user_id=user.id,
        role="user",
        content=body.message,
        meta={}
    )

    # Process message (non-streaming)
    full_response = ""
    recipe_cards = []

    try:
        async for event_type, data in chat_service.stream_message(
            message=body.message,
            user=user
        ):
            if event_type == "token":
                full_response += data.get("content", "")
            elif event_type == "cards":
                cards = data.get("cards", [])
                recipe_cards = [
                    RecipeCardMeta(
                        recipe_id=c.recipe_id,
                        title=c.title,
                        rating_avg=c.rating_avg,
                        time_total=c.time_total,
                        key_ingredients=c.key_ingredients,
                        one_sentence_summary=c.one_sentence_summary,
                        why_match=c.why_match,
                    )
                    for c in cards
                ]
            elif event_type == "error":
                error = data.get("error", {})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error.get("message", "Chat processing failed")
                )

        # Save assistant message
        msg_id = conversation_service.add_message(
            conversation_id=conv_id,
            user_id=user.id,
            role="assistant",
            content=full_response,
            meta={"recipe_cards": [c.model_dump() for c in recipe_cards]}
        )

        return ChatResponse(
            conversation_id=conv_id,
            message_id=msg_id,
            content=full_response,
            meta=MessageMeta(recipe_cards=recipe_cards)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error", user_id=user.id[:8])
        # Save error message
        conversation_service.add_message(
            conversation_id=conv_id,
            user_id=user.id,
            role="assistant",
            content="[Error occurred]",
            meta={"error": {"message": str(e), "code": "chat_error"}}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
