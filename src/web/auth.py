"""Authentication blueprint for Flask web application."""

import os

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import UserMixin, current_user, login_required, login_user, logout_user

from src.app.constants import DEFAULT_USER_USERNAME
from src.app.logging_config import get_logger
from src.web.app import limiter

logger = get_logger(__name__)

auth_bp = Blueprint("auth", __name__)


class FlaskUser(UserMixin):
    """Flask-Login compatible user wrapper."""

    def __init__(self, user):
        """Wrap a User from UserStore.

        Args:
            user: User object from UserStore
        """
        self._user = user

    def get_id(self) -> str:
        """Return user ID for Flask-Login."""
        return self._user.id

    @property
    def id(self) -> str:
        """User UUID."""
        return self._user.id

    @property
    def username(self) -> str:
        """Username."""
        return self._user.username

    @property
    def is_active(self) -> bool:
        """Whether user is active."""
        return self._user.is_active


def set_session_sid(sid: str) -> None:
    """Set session ID in Flask session.

    INVARIANT: Flask cookie session must contain ONLY 'sid'.
    This is the single point for setting session state.

    Args:
        sid: Web session ID (UUID)
    """
    session.clear()
    session["sid"] = sid
    session.permanent = True


def get_session_sid() -> str | None:
    """Get session ID from Flask session.

    Returns:
        Session ID or None if not set
    """
    return session.get("sid")


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5/minute")  # Rate limit login attempts
def login():
    """Login page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for("chat.chat_page"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Block default_user from web login
        if username.lower() == DEFAULT_USER_USERNAME.lower():
            flash("Invalid username or password", "error")
            return redirect(url_for("auth.login"))

        user_store = current_app.user_store
        user = user_store.get_user_by_username(username)

        # Same error message for all failures (prevent enumeration)
        if not user or not user.is_active:
            flash("Invalid username or password", "error")
            return redirect(url_for("auth.login"))

        # Check if user needs to set password first
        if user.password_hash is None:
            logger.info("User needs to set password", username=username)
            return redirect(url_for("auth.set_password", user_id=user.id))

        # Verify password
        verified_user = user_store.verify_password(username, password)
        if not verified_user:
            flash("Invalid username or password", "error")
            return redirect(url_for("auth.login"))

        # Login successful - create new web session (session fixation defense)
        web_session_store = current_app.web_session_store
        new_sid = web_session_store.create(verified_user.id)
        set_session_sid(new_sid)

        # Login with Flask-Login
        flask_user = FlaskUser(verified_user)
        login_user(flask_user, remember=True)

        logger.info("User logged in", username=username, user_id=verified_user.id)

        # Redirect to next page or chat
        next_page = request.args.get("next")
        if next_page and next_page.startswith("/"):
            return redirect(next_page)
        return redirect(url_for("chat.chat_page"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    """Logout handler."""
    # Delete web session
    sid = get_session_sid()
    if sid:
        current_app.web_session_store.delete(sid)

    # Clear Flask session and logout
    session.clear()
    logout_user()

    flash("You have been logged out.", "info")
    logger.info("User logged out")

    return redirect(url_for("auth.login"))


@auth_bp.route("/set-password/<user_id>", methods=["GET", "POST"])
@limiter.limit("5/minute")
def set_password(user_id: str):
    """First-login password setup.

    Token-gated when ALLOW_LAN=1 or ENV=production.

    SECURITY:
    - Accept setup token via POST form field only (NOT query string)
    - Do NOT echo token back in templates after failure
    - Do NOT log token values
    - Return generic "Invalid request" for invalid user_id OR already-set password
    """
    user_store = current_app.user_store
    user = user_store.get_user_by_id(user_id)

    # SECURITY: Generic message prevents user enumeration
    if not user or user.password_hash is not None:
        flash("Invalid request", "error")
        return redirect(url_for("auth.login"))

    require_token = current_app.config.get("REQUIRE_SETUP_TOKEN", False)
    expected_token = current_app.config.get("INITIAL_SETUP_TOKEN")

    if request.method == "POST":
        token = request.form.get("setup_token", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        # Check token if required
        if require_token or expected_token:
            # Token required when exposed or if explicitly set
            if not expected_token:
                flash("Setup token required but not configured", "error")
                return render_template("set_password.html", user=user, require_token=True)

            if token != expected_token:
                flash("Invalid setup token", "error")
                logger.warning("Invalid setup token attempt", user_id=user_id)
                return render_template("set_password.html", user=user, require_token=True)
        elif request.remote_addr not in ("127.0.0.1", "::1"):
            # Not localhost and no token required - still require token for safety
            flash("Setup token required for remote access", "error")
            return render_template("set_password.html", user=user, require_token=True)

        # Validate password
        if not password or len(password) < 8:
            flash("Password must be at least 8 characters", "error")
            return render_template("set_password.html", user=user, require_token=bool(expected_token))

        if password != confirm:
            flash("Passwords do not match", "error")
            return render_template("set_password.html", user=user, require_token=bool(expected_token))

        # Set password
        user_store.set_password(user_id, password)
        flash("Password set successfully. Please login.", "success")
        logger.info("Password set for user", user_id=user_id, username=user.username)

        return redirect(url_for("auth.login"))

    return render_template("set_password.html", user=user, require_token=bool(expected_token))


@auth_bp.route("/users")
@login_required
def list_users():
    """List users (for development/testing only)."""
    if not current_app.debug:
        flash("Not available in production", "error")
        return redirect(url_for("chat.chat_page"))

    users = current_app.user_store.list_users()
    return render_template("users.html", users=users)
