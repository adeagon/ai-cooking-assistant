"""Session state storage for dinner planning sessions."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.domain.models import SessionState

logger = get_logger(__name__)


class SessionStore:
    """Manages session state persistence in SQLite.

    Each user has their own isolated sessions identified by user_id (UUID).
    """

    def __init__(self, db_path: Path, user_id: str):
        """Initialize SessionStore with database path and user ID.

        Args:
            db_path: Path to SQLite database file
            user_id: UUID string identifying the user (required)

        Raises:
            ValueError: If user_id is not provided
        """
        if not user_id:
            raise ValueError("user_id is required")
        self.db_path = db_path
        self.user_id = user_id
        self._current_session_id: str | None = None
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create sessions table if it doesn't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                ingredients_on_hand TEXT,
                avoid_tonight TEXT,
                goals TEXT,
                time_limit INTEGER,
                servings INTEGER,
                rolling_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

        conn.commit()
        conn.close()

        logger.info("Sessions table ensured", db_path=str(self.db_path))

    def create(self) -> str:
        """Create a new session with empty state.

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO sessions (
                id, user_id, ingredients_on_hand, avoid_tonight, goals,
                time_limit, servings, rolling_summary, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, self.user_id, "[]", "[]", "[]", None, None, "", now, now),
        )

        conn.commit()
        conn.close()

        self._current_session_id = session_id

        logger.info("Created new session", session_id=session_id, user_id=self.user_id)

        return session_id

    def get(self, session_id: str) -> SessionState | None:
        """Get session state by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionState if found and belongs to current user, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Only return session if it belongs to current user
        cursor.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, self.user_id),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            logger.warning("Session not found", session_id=session_id, user_id=self.user_id)
            return None

        # Parse JSON arrays
        ingredients_on_hand = json.loads(row["ingredients_on_hand"])
        avoid_tonight = json.loads(row["avoid_tonight"])
        goals = json.loads(row["goals"])

        session = SessionState(
            ingredients_on_hand=ingredients_on_hand,
            avoid_tonight=avoid_tonight,
            goals=goals,
            time_limit_minutes=row["time_limit"],
            servings=row["servings"],
        )

        logger.info(
            "Loaded session",
            session_id=session_id,
            user_id=self.user_id,
            ingredients_count=len(ingredients_on_hand),
            goals=goals,
        )

        return session

    def update(self, session_id: str, **updates) -> SessionState | None:
        """Update session state fields.

        Args:
            session_id: Session ID to update
            **updates: Fields to update (ingredients_on_hand, goals, etc.)

        Returns:
            Updated SessionState if found and belongs to current user, None otherwise
        """
        session = self.get(session_id)
        if session is None:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                logger.warning("Unknown session field", field=key)

        # Save to database
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        # Only update if session belongs to current user
        cursor.execute(
            """
            UPDATE sessions
            SET ingredients_on_hand = ?,
                avoid_tonight = ?,
                goals = ?,
                time_limit = ?,
                servings = ?,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                json.dumps(session.ingredients_on_hand),
                json.dumps(session.avoid_tonight),
                json.dumps(session.goals),
                session.time_limit_minutes,
                session.servings,
                now,
                session_id,
                self.user_id,
            ),
        )

        conn.commit()
        conn.close()

        logger.info("Updated session", session_id=session_id, user_id=self.user_id, updates=list(updates.keys()))

        return session

    def update_summary(self, session_id: str, summary: str) -> None:
        """Update rolling summary for session.

        Args:
            session_id: Session ID
            summary: New rolling summary text
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        # Only update if session belongs to current user
        cursor.execute(
            "UPDATE sessions SET rolling_summary = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (summary, now, session_id, self.user_id),
        )

        conn.commit()
        conn.close()

        logger.info("Updated session summary", session_id=session_id, user_id=self.user_id, summary_length=len(summary))

    def get_summary(self, session_id: str) -> str:
        """Get rolling summary for session.

        Args:
            session_id: Session ID

        Returns:
            Rolling summary text, or empty string if not found or not owned by user
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Only return summary if session belongs to current user
        cursor.execute(
            "SELECT rolling_summary FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, self.user_id),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return ""

        return row["rolling_summary"] or ""

    def get_or_create_current(self) -> tuple[str, SessionState]:
        """Get current session or create new one for current user.

        Returns:
            Tuple of (session_id, SessionState)
        """
        if self._current_session_id is not None:
            session = self.get(self._current_session_id)
            if session is not None:
                return self._current_session_id, session

        # Create new session for this user
        session_id = self.create()
        session = self.get(session_id)

        return session_id, session or SessionState()
