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

    Each user has their own sessions, tracked separately.
    """

    def __init__(self, db_path: Path):
        """Initialize SessionStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Track current session per user (dict instead of single string)
        self._current_session_ids: dict[str, str] = {}
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create sessions table if it doesn't exist.

        Handles migration from old single-user schema to multi-user schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
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

        # Check if username column exists, add if missing (migration)
        cursor.execute("PRAGMA table_info(sessions)")
        columns = {col[1]: col for col in cursor.fetchall()}

        if "username" not in columns:
            logger.info("Adding username column to sessions table")
            cursor.execute("ALTER TABLE sessions ADD COLUMN username TEXT")
            # Assign existing rows to 'guest'
            cursor.execute("UPDATE sessions SET username = 'guest' WHERE username IS NULL")

        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_username
            ON sessions(username)
        """)

        conn.commit()
        conn.close()

        logger.info("Sessions table ensured", db_path=str(self.db_path))

    def create(self, user_id: str | None = None) -> str:
        """Create a new session with empty state.

        Args:
            user_id: Username to create session for. Defaults to 'guest' if None.

        Returns:
            Session ID (UUID string)
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO sessions (
                id, ingredients_on_hand, avoid_tonight, goals,
                time_limit, servings, rolling_summary, created_at, updated_at, username
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, "[]", "[]", "[]", None, None, "", now, now, username),
        )

        conn.commit()
        conn.close()

        # Track current session per user
        self._current_session_ids[username] = session_id

        logger.info("Created new session", session_id=session_id, username=username)

        return session_id

    def get(self, session_id: str) -> SessionState | None:
        """Get session state by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionState if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            logger.warning("Session not found", session_id=session_id)
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
            Updated SessionState if found, None otherwise
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            UPDATE sessions
            SET ingredients_on_hand = ?,
                avoid_tonight = ?,
                goals = ?,
                time_limit = ?,
                servings = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(session.ingredients_on_hand),
                json.dumps(session.avoid_tonight),
                json.dumps(session.goals),
                session.time_limit_minutes,
                session.servings,
                now,
                session_id,
            ),
        )

        conn.commit()
        conn.close()

        logger.info("Updated session", session_id=session_id, updates=list(updates.keys()))

        return session

    def update_summary(self, session_id: str, summary: str) -> None:
        """Update rolling summary for session.

        Args:
            session_id: Session ID
            summary: New rolling summary text
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            "UPDATE sessions SET rolling_summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, session_id),
        )

        conn.commit()
        conn.close()

        logger.info("Updated session summary", session_id=session_id, summary_length=len(summary))

    def get_summary(self, session_id: str) -> str:
        """Get rolling summary for session.

        Args:
            session_id: Session ID

        Returns:
            Rolling summary text, or empty string if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT rolling_summary FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return ""

        return row["rolling_summary"] or ""

    def get_or_create_current(self, user_id: str | None = None) -> tuple[str, SessionState]:
        """Get current session or create new one for a user.

        Args:
            user_id: Username to get/create session for. Defaults to 'guest' if None.

        Returns:
            Tuple of (session_id, SessionState)
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        # Check if we have a current session for this user in memory
        if username in self._current_session_ids:
            session_id = self._current_session_ids[username]
            session = self.get(session_id)
            if session is not None:
                return session_id, session

        # Try to find existing session for this user in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id FROM sessions
            WHERE username = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (username,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            session_id = row[0]
            session = self.get(session_id)
            if session is not None:
                self._current_session_ids[username] = session_id
                return session_id, session

        # Create new session for this user
        session_id = self.create(user_id=username)
        session = self.get(session_id)

        return session_id, session or SessionState()
