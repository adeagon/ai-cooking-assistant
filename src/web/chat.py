"""Chat blueprint for Flask web application."""

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from src.app.logging_config import get_logger
from src.web.auth import get_session_sid

logger = get_logger(__name__)

chat_bp = Blueprint("chat", __name__)


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
    """Handle chat message submission.

    Phase 3: Basic message handling (echo).
    Phase 4: Full chat integration with LLM.
    """
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

    # Phase 3: Basic echo response for testing
    # Phase 4 will replace this with actual chat logic
    response = _process_message_basic(message)

    # Store exchange atomically
    web_session_store.append_exchange(
        session_id=sid,
        user_text=message,
        assistant_text=response,
        rolling_summary=web_session.rolling_summary,
        last_cards_json=web_session.last_cards_json,
    )

    logger.info(
        "Chat message processed",
        user_id=current_user.id,
        message_length=len(message),
        response_length=len(response),
    )

    return redirect(url_for("chat.chat_page"))


def _process_message_basic(message: str) -> str:
    """Basic message processing for Phase 3.

    This is a placeholder that will be replaced with full chat logic in Phase 4.

    Args:
        message: User's message

    Returns:
        Assistant response
    """
    # Handle basic commands
    message_lower = message.lower().strip()

    if message_lower in ("/commands", "/help"):
        return (
            "**Available Commands**\n\n"
            "**Session:**\n"
            "- `/new` - Start a new session\n"
            "- `/prefs` - Show your preferences\n"
            "- `/commands` - Show this help\n\n"
            "**Recipe Feedback:**\n"
            "- `/like <ref>` - Like a recipe\n"
            "- `/dislike <ref>` - Dislike a recipe\n"
            "- `/rate <1-5> <ref>` - Rate a recipe\n"
            "- `/cooked <ref>` - Mark as cooked\n\n"
            "**Recipe Box:**\n"
            "- `/save <ref>` - Save recipe\n"
            "- `/unsave <ref>` - Remove from box\n"
            "- `/box` - View saved recipes\n"
            "- `/show <ref>` - Show full recipe\n\n"
            "**Meal Planning:**\n"
            "- `/mealplan` - Plan meals\n"
            "- `/plan` - View current plan\n"
            "- `/grocery` - Generate grocery list\n\n"
            "*Full chat integration coming in Phase 4.*"
        )

    if message_lower == "/new":
        return "Starting a new session. (Full implementation in Phase 4)"

    if message_lower == "/prefs":
        return "Your preferences will be displayed here. (Full implementation in Phase 4)"

    if message_lower == "/box":
        return "Your Recipe Box will be displayed here. (Full implementation in Phase 4)"

    if message_lower == "/history":
        return "Your cooking history will be displayed here. (Full implementation in Phase 4)"

    # Default response
    return (
        f"I received your message: \"{message}\"\n\n"
        "*Full chat integration with recipe recommendations coming in Phase 4.*\n\n"
        "Try `/commands` to see available commands."
    )


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
