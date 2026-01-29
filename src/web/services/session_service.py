"""Session service for managing browser sessions with SQLite persistence."""

import uuid
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.web.db import get_db_connection
from src.web.models import Session

logger = get_logger(__name__)


class SessionService:
    """Service for managing browser sessions.

    Sessions are stored in SQLite with revocation semantics:
    - Active sessions have revoked_at = NULL
    - Revoked sessions have revoked_at set to revocation timestamp
    - Sessions expire after max_age_days based on last_seen_at
    """

    def __init__(self, db_path: Path, max_age_days: int = 30):
        """Initialize session service.

        Args:
            db_path: Path to SQLite database
            max_age_days: Maximum session age in days
        """
        self.db_path = db_path
        self.max_age_days = max_age_days

    def create(
        self,
        user_id: str,
        user_agent: str | None = None,
        ip: str | None = None
    ) -> Session:
        """Create a new session for a user.

        Args:
            user_id: User UUID
            user_agent: Browser user agent string
            ip: Client IP address

        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        with get_db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO web_sessions
                    (session_id, user_id, created_at, last_seen_at, user_agent, ip)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, now.isoformat(), now.isoformat(), user_agent, ip)
            )
            conn.commit()
            logger.info(
                "Created session",
                session_id=session_id[:8],
                user_id=user_id[:8]
            )

        return Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_seen_at=now,
            revoked_at=None,
            user_agent=user_agent,
            ip=ip
        )

    def get_valid(self, session_id: str) -> Session | None:
        """Get a session if it's valid (not revoked and not expired).

        A session is valid if:
        - It exists
        - revoked_at IS NULL
        - last_seen_at is within max_age_days

        Args:
            session_id: Session ID to validate

        Returns:
            Session if valid, None otherwise
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, created_at, last_seen_at,
                       revoked_at, user_agent, ip
                FROM web_sessions
                WHERE session_id = ?
                  AND revoked_at IS NULL
                  AND last_seen_at > datetime('now', ? || ' days')
                """,
                (session_id, f"-{self.max_age_days}")
            ).fetchone()

            if row:
                return self._row_to_session(row)
            return None

    def touch(self, session_id: str) -> bool:
        """Update session's last_seen_at timestamp.

        Args:
            session_id: Session ID to touch

        Returns:
            True if updated, False if session not found
        """
        now = datetime.now()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE web_sessions
                SET last_seen_at = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """,
                (now.isoformat(), session_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def revoke(self, session_id: str) -> bool:
        """Revoke a session by setting revoked_at.

        Args:
            session_id: Session ID to revoke

        Returns:
            True if revoked, False if session not found or already revoked
        """
        now = datetime.now()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE web_sessions
                SET revoked_at = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """,
                (now.isoformat(), session_id)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("Revoked session", session_id=session_id[:8])
                return True
            return False

    def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all sessions for a user.

        Args:
            user_id: User UUID

        Returns:
            Number of sessions revoked
        """
        now = datetime.now()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE web_sessions
                SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (now.isoformat(), user_id)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    "Revoked all sessions for user",
                    user_id=user_id[:8],
                    count=cursor.rowcount
                )
            return cursor.rowcount

    def get_active_sessions_for_user(self, user_id: str) -> list[Session]:
        """Get all active (non-revoked, non-expired) sessions for a user.

        Args:
            user_id: User UUID

        Returns:
            List of active sessions
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, created_at, last_seen_at,
                       revoked_at, user_agent, ip
                FROM web_sessions
                WHERE user_id = ?
                  AND revoked_at IS NULL
                  AND last_seen_at > datetime('now', ? || ' days')
                ORDER BY last_seen_at DESC
                """,
                (user_id, f"-{self.max_age_days}")
            ).fetchall()
            return [self._row_to_session(row) for row in rows]

    def cleanup_expired(self) -> int:
        """Remove sessions older than max_age_days.

        Returns:
            Number of sessions cleaned up
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM web_sessions
                WHERE last_seen_at < datetime('now', ? || ' days')
                """,
                (f"-{self.max_age_days}",)
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info("Cleaned up expired sessions", count=count)
            return count

    def _row_to_session(self, row) -> Session:
        """Convert database row to Session model.

        Args:
            row: SQLite Row object

        Returns:
            Session model
        """
        def parse_datetime(val):
            if val is None:
                return None
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return val

        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=parse_datetime(row["created_at"]),
            last_seen_at=parse_datetime(row["last_seen_at"]),
            revoked_at=parse_datetime(row["revoked_at"]),
            user_agent=row["user_agent"],
            ip=row["ip"]
        )
