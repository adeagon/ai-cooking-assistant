"""Web session storage using SQLite for server-side session state."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.constants import MAX_MESSAGES_PER_SESSION, WEB_SESSION_TTL_HOURS
from src.app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WebSession:
    """Server-side web session data."""

    id: str  # UUID
    user_id: str  # UUID
    chat_session_id: str | None  # Pointer to CLI-style session
    rolling_summary: str | None
    last_cards_json: str | None
    expires_at: datetime
    created_at: datetime | None
    updated_at: datetime | None


@dataclass
class WebMessage:
    """Chat message in web session."""

    id: int
    web_session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    created_at: datetime | None


class WebSessionStore:
    """Manages server-side web session state in SQLite.

    IMPORTANT: Flask cookie session must only contain 'sid' (session ID).
    All state (rolling_summary, last_cards_json) must be stored server-side.
    """

    SESSION_TTL_HOURS = WEB_SESSION_TTL_HOURS
    MAX_MESSAGES = MAX_MESSAGES_PER_SESSION

    def __init__(self, db_path: Path):
        """Initialize WebSessionStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        """Create web session tables if they don't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                chat_session_id TEXT,
                rolling_summary TEXT,
                last_cards_json TEXT,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_web_sessions_expires ON web_sessions(expires_at)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                web_session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (web_session_id) REFERENCES web_sessions(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_web_messages_session_id ON web_messages(web_session_id, id)")

        conn.commit()
        conn.close()

        logger.info("Web session tables ensured", db_path=str(self.db_path))

    def create(self, user_id: str) -> str:
        """Create new web session with expiration.

        Plan B: Does NOT expire other sessions for this user.
        Users can have multiple active sessions (different browsers/devices).

        Args:
            user_id: UUID of the user

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=self.SESSION_TTL_HOURS)
        now = datetime.now()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO web_sessions (
                id, user_id, chat_session_id, rolling_summary, last_cards_json,
                expires_at, created_at, updated_at
            )
            VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (session_id, user_id, expires_at, now, now),
        )

        conn.commit()
        conn.close()

        logger.info("Created web session", session_id=session_id, user_id=user_id)

        return session_id

    def get(self, session_id: str) -> WebSession | None:
        """Load web session if not expired.

        Args:
            session_id: Session ID to retrieve

        Returns:
            WebSession if found and not expired, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, user_id, chat_session_id, rolling_summary, last_cards_json,
                   expires_at, created_at, updated_at
            FROM web_sessions
            WHERE id = ? AND expires_at > CURRENT_TIMESTAMP
            """,
            (session_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return WebSession(
            id=row["id"],
            user_id=row["user_id"],
            chat_session_id=row["chat_session_id"],
            rolling_summary=row["rolling_summary"],
            last_cards_json=row["last_cards_json"],
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def touch(self, session_id: str) -> None:
        """Refresh TTL on authenticated request (e.g., GET /).

        For write requests (POST /chat), use append_exchange() instead,
        which already touches TTL.

        Args:
            session_id: Session ID to refresh
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        expires_at = datetime.now() + timedelta(hours=self.SESSION_TTL_HOURS)

        cursor.execute(
            """
            UPDATE web_sessions
            SET expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (expires_at, session_id),
        )

        conn.commit()
        conn.close()

    def update(
        self,
        session_id: str,
        rolling_summary: str | None = None,
        last_cards_json: str | None = None,
        chat_session_id: str | None = None,
    ) -> None:
        """Update session state and refresh expiration.

        Note: For POST /chat, prefer append_exchange() for full atomicity.

        Args:
            session_id: Session ID to update
            rolling_summary: New rolling summary (None to keep current)
            last_cards_json: New cards JSON (None to keep current)
            chat_session_id: New chat session ID (None to keep current)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        expires_at = datetime.now() + timedelta(hours=self.SESSION_TTL_HOURS)

        # Build dynamic UPDATE
        updates = ["expires_at = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [expires_at]

        if rolling_summary is not None:
            updates.append("rolling_summary = ?")
            params.append(rolling_summary)

        if last_cards_json is not None:
            updates.append("last_cards_json = ?")
            params.append(last_cards_json)

        if chat_session_id is not None:
            updates.append("chat_session_id = ?")
            params.append(chat_session_id)

        params.append(session_id)

        cursor.execute(
            f"""
            UPDATE web_sessions
            SET {', '.join(updates)}
            WHERE id = ?
            """,
            params,
        )

        conn.commit()
        conn.close()

    def delete(self, session_id: str) -> None:
        """Delete web session (logout).

        Messages cascade delete via FK constraint.

        Args:
            session_id: Session ID to delete
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM web_sessions WHERE id = ?", (session_id,))

        conn.commit()
        conn.close()

        logger.info("Deleted web session", session_id=session_id)

    def cleanup_expired(self) -> int:
        """Delete all expired sessions.

        Returns:
            Count of sessions deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM web_sessions WHERE expires_at < CURRENT_TIMESTAMP")

        count = cursor.rowcount
        conn.commit()
        conn.close()

        if count > 0:
            logger.info("Cleaned up expired web sessions", count=count)

        return count

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message and prune to max in ONE TRANSACTION (atomic).

        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            content: Message content
        """
        conn = self._get_connection()
        try:
            with conn:  # Transaction context
                # Insert new message
                conn.execute(
                    """
                    INSERT INTO web_messages (web_session_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, role, content),
                )

                # Prune to last MAX_MESSAGES in same transaction
                conn.execute(
                    """
                    DELETE FROM web_messages
                    WHERE id IN (
                        SELECT id FROM web_messages
                        WHERE web_session_id = ?
                        ORDER BY id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (session_id, self.MAX_MESSAGES),
                )
        finally:
            conn.close()

    def get_messages(self, session_id: str, limit: int = 50) -> list[WebMessage]:
        """Get recent messages for display (oldest first for UI).

        Args:
            session_id: Session ID
            limit: Maximum messages to return

        Returns:
            List of WebMessage objects, oldest first
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Use id ordering for consistency with pruning
        cursor.execute(
            """
            SELECT id, web_session_id, role, content, created_at
            FROM web_messages
            WHERE web_session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id, limit),
        )

        messages = [
            WebMessage(
                id=row["id"],
                web_session_id=row["web_session_id"],
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            )
            for row in cursor.fetchall()
        ]

        conn.close()
        return messages

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        rolling_summary: str | None = None,
        last_cards_json: str | None = None,
    ) -> None:
        """Single transaction: update session state + add messages + prune + touch TTL.

        This is the recommended method for POST /chat to ensure atomicity.
        Avoids "user message stored, assistant message missing" on crash.

        Args:
            session_id: Session ID
            user_text: User's message
            assistant_text: Assistant's response
            rolling_summary: New rolling summary (None to keep current)
            last_cards_json: New cards JSON (None to keep current)
        """
        conn = self._get_connection()
        expires_at = datetime.now() + timedelta(hours=self.SESSION_TTL_HOURS)

        try:
            with conn:  # Single transaction
                # Build update query
                if rolling_summary is not None and last_cards_json is not None:
                    conn.execute(
                        """
                        UPDATE web_sessions
                        SET rolling_summary = ?, last_cards_json = ?,
                            updated_at = CURRENT_TIMESTAMP, expires_at = ?
                        WHERE id = ?
                        """,
                        (rolling_summary, last_cards_json, expires_at, session_id),
                    )
                elif rolling_summary is not None:
                    conn.execute(
                        """
                        UPDATE web_sessions
                        SET rolling_summary = ?,
                            updated_at = CURRENT_TIMESTAMP, expires_at = ?
                        WHERE id = ?
                        """,
                        (rolling_summary, expires_at, session_id),
                    )
                elif last_cards_json is not None:
                    conn.execute(
                        """
                        UPDATE web_sessions
                        SET last_cards_json = ?,
                            updated_at = CURRENT_TIMESTAMP, expires_at = ?
                        WHERE id = ?
                        """,
                        (last_cards_json, expires_at, session_id),
                    )
                else:
                    # Just touch TTL
                    conn.execute(
                        """
                        UPDATE web_sessions
                        SET updated_at = CURRENT_TIMESTAMP, expires_at = ?
                        WHERE id = ?
                        """,
                        (expires_at, session_id),
                    )

                # Insert user message
                conn.execute(
                    """
                    INSERT INTO web_messages (web_session_id, role, content)
                    VALUES (?, 'user', ?)
                    """,
                    (session_id, user_text),
                )

                # Insert assistant message
                conn.execute(
                    """
                    INSERT INTO web_messages (web_session_id, role, content)
                    VALUES (?, 'assistant', ?)
                    """,
                    (session_id, assistant_text),
                )

                # Prune to last MAX_MESSAGES in same transaction
                conn.execute(
                    """
                    DELETE FROM web_messages
                    WHERE id IN (
                        SELECT id FROM web_messages
                        WHERE web_session_id = ?
                        ORDER BY id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (session_id, self.MAX_MESSAGES),
                )
        finally:
            conn.close()

    def get_last_cards(self, session_id: str) -> list[dict]:
        """Get last recipe cards from session.

        Args:
            session_id: Session ID

        Returns:
            List of card dictionaries, empty if none
        """
        session = self.get(session_id)
        if not session or not session.last_cards_json:
            return []

        try:
            return json.loads(session.last_cards_json)
        except json.JSONDecodeError:
            logger.warning("Invalid last_cards_json", session_id=session_id)
            return []

    def clear_messages(self, session_id: str) -> None:
        """Clear all messages for a session (e.g., /new command).

        Args:
            session_id: Session ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM web_messages WHERE web_session_id = ?", (session_id,))

        conn.commit()
        conn.close()

        logger.info("Cleared messages for session", session_id=session_id)

    def get_sessions_for_user(self, user_id: str) -> list[WebSession]:
        """Get all active sessions for a user (Plan B: multiple sessions allowed).

        Args:
            user_id: User's UUID

        Returns:
            List of active WebSession objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, user_id, chat_session_id, rolling_summary, last_cards_json,
                   expires_at, created_at, updated_at
            FROM web_sessions
            WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )

        sessions = [
            WebSession(
                id=row["id"],
                user_id=row["user_id"],
                chat_session_id=row["chat_session_id"],
                rolling_summary=row["rolling_summary"],
                last_cards_json=row["last_cards_json"],
                expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            )
            for row in cursor.fetchall()
        ]

        conn.close()
        return sessions
