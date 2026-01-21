"""Flask application factory."""

import time
from pathlib import Path

from flask import Flask, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from src.app.logging_config import get_logger
from src.memory.user_store import UserStore
from src.memory.web_session_store import WebSessionStore
from src.web.config import get_config

logger = get_logger(__name__)

# Global extensions
limiter = Limiter(key_func=get_remote_address)
csrf = CSRFProtect()
login_manager = LoginManager()

# Cleanup tracking (per-worker)
_last_cleanup = 0
CLEANUP_INTERVAL = 3600  # 1 hour


def create_app(config_class=None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_class: Optional configuration class (defaults to env-based)

    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder=Path(__file__).parent / "templates",
        static_folder=Path(__file__).parent / "static",
    )

    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize extensions
    limiter.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)

    # Configure login manager
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    # Initialize stores
    db_path = Path(app.config.get("SQLITE_DB_PATH", "data/sqlite/recipes.db"))
    app.user_store = UserStore(db_path)
    app.web_session_store = WebSessionStore(db_path)

    # Ensure WAL mode
    _ensure_wal_mode(db_path)

    # Register blueprints
    from src.web.auth import auth_bp
    from src.web.chat import chat_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(chat_bp)

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id: str):
        """Load user by ID for Flask-Login."""
        from src.web.auth import FlaskUser

        user = app.user_store.get_user_by_id(user_id)
        if user and user.is_active:
            return FlaskUser(user)
        return None

    # Root redirect
    @app.route("/")
    def index():
        """Redirect root to chat."""
        return redirect(url_for("chat.chat_page"))

    # Opportunistic session cleanup
    @app.before_request
    def maybe_cleanup_sessions():
        """Clean up expired sessions periodically (time-guarded)."""
        global _last_cleanup
        now = time.time()
        if now - _last_cleanup > CLEANUP_INTERVAL:
            _last_cleanup = now
            try:
                count = app.web_session_store.cleanup_expired()
                if count > 0:
                    logger.info("Opportunistic session cleanup", count=count)
            except Exception as e:
                logger.warning("Session cleanup failed", error=str(e))

    # Run cleanup on startup
    with app.app_context():
        try:
            app.web_session_store.cleanup_expired()
        except Exception as e:
            logger.warning("Startup session cleanup failed", error=str(e))

    logger.info(
        "Flask app created",
        debug=app.debug,
        db_path=str(db_path),
    )

    return app


def _ensure_wal_mode(db_path: Path) -> None:
    """Ensure SQLite database uses WAL mode.

    WAL mode is critical for concurrent access with multiple workers.
    """
    import sqlite3

    if not db_path.exists():
        logger.warning("Database does not exist, skipping WAL mode check", path=str(db_path))
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute("PRAGMA journal_mode").fetchone()
        if row and row[0].lower() == "wal":
            logger.info("WAL mode confirmed", path=str(db_path))
        else:
            logger.warning("Could not enable WAL mode", path=str(db_path), mode=row[0] if row else None)
    finally:
        conn.close()


def run_dev_server():
    """Run development server."""
    from src.web.config import get_bind_host, get_port

    app = create_app()
    host = get_bind_host()
    port = get_port()

    print(f"\n{'='*60}")
    print("AI Cooking Assistant - Web Interface")
    print(f"{'='*60}")
    print(f"Running at: http://{host}:{port}")
    if host == "127.0.0.1":
        print("Accessible only from this machine (ALLOW_LAN=0)")
    else:
        print("Accessible from LAN (ALLOW_LAN=1)")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    run_dev_server()
