"""Chat blueprint for Flask web application."""

import json
from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from src.app.logging_config import get_logger
from src.services.chat_service import ChatService, ChatResult, UserContext
from src.web.auth import get_session_sid

logger = get_logger(__name__)

chat_bp = Blueprint("chat", __name__)


def _get_db_path() -> Path:
    """Get SQLite database path from app config."""
    return Path(current_app.config.get("SQLITE_DB_PATH", "data/sqlite/recipes.db"))


@chat_bp.route("/chat", methods=["GET"])
@login_required
def chat_page():
    """Display chat interface."""
    sid = get_session_sid()

    if not sid:
        # No web session - redirect to login
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    # Load web session
    web_session_store = current_app.web_session_store
    web_session = web_session_store.get(sid)

    if not web_session:
        # Session expired or invalid
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    # Touch TTL for read request
    web_session_store.touch(sid)

    # Load messages for display
    messages = web_session_store.get_messages(sid)

    return render_template(
        "chat.html",
        user=current_user,
        messages=messages,
        rolling_summary=web_session.rolling_summary or "",
    )


@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat_submit():
    """Handle chat message submission with ChatService integration."""
    sid = get_session_sid()

    if not sid:
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    web_session_store = current_app.web_session_store
    web_session = web_session_store.get(sid)

    if not web_session:
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    message = request.form.get("message", "").strip()

    if not message:
        # Empty message - just reload chat
        return redirect(url_for("chat.chat_page"))

    # Parse last cards from server-side session state (NEVER from client)
    last_cards = []
    if web_session.last_cards_json:
        try:
            last_cards = json.loads(web_session.last_cards_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse last_cards_json", session_id=sid)
            last_cards = []

    # Create user context with per-request stores
    user_ctx = UserContext(
        user_id=current_user.id,
        db_path=_get_db_path(),
    )

    # Process message through ChatService
    chat_service = ChatService(user_ctx, last_cards)
    result: ChatResult = chat_service.process_message(
        message=message,
        rolling_summary=web_session.rolling_summary,
    )

    # Prepare updated state
    # If result has new cards, serialize them; otherwise keep existing
    new_cards_json = web_session.last_cards_json
    if result.cards:
        new_cards_json = json.dumps([
            card.model_dump() if hasattr(card, "model_dump") else card
            for card in result.cards
        ])

    # If result provides new rolling summary, use it; otherwise keep existing
    new_rolling_summary = (
        result.rolling_summary
        if result.rolling_summary is not None
        else web_session.rolling_summary
    )

    # Store exchange atomically (session state + messages + prune + TTL refresh)
    web_session_store.append_exchange(
        session_id=sid,
        user_text=message,
        assistant_text=result.response,
        rolling_summary=new_rolling_summary,
        last_cards_json=new_cards_json,
    )

    logger.info(
        "Chat message processed",
        user_id=current_user.id,
        message_length=len(message),
        response_length=len(result.response),
        command_executed=result.command_executed,
    )

    return redirect(url_for("chat.chat_page"))


@chat_bp.route("/chat/clear", methods=["POST"])
@login_required
def clear_chat():
    """Clear chat history (start new session)."""
    sid = get_session_sid()

    if not sid:
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("auth.login"))

    web_session_store = current_app.web_session_store
    web_session = web_session_store.get(sid)

    if web_session:
        # Clear messages and rolling summary
        web_session_store.clear_messages(sid)
        web_session_store.update(
            sid,
            rolling_summary=None,
            last_cards_json=None,
        )

    flash("Chat history cleared.", "info")
    logger.info("Chat cleared", user_id=current_user.id, session_id=sid)

    return redirect(url_for("chat.chat_page"))
