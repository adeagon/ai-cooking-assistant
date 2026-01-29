"""User service for managing users and UUID-username mapping."""

import uuid
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.web.db import get_db_connection
from src.web.models import User

logger = get_logger(__name__)


class UserService:
    """Service for user management and UUID-username mapping.

    Handles user CRUD operations and provides mapping between
    UUID (used by web app) and username (used by StoreFactory).
    """

    def __init__(self, db_path: Path):
        """Initialize user service.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def get_all(self) -> list[User]:
        """Get all users.

        Returns:
            List of all users
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, username, display_name, created_at FROM users ORDER BY username"
            ).fetchall()
            return [self._row_to_user(row) for row in rows]

    def get_by_id(self, user_id: str) -> User | None:
        """Get user by UUID.

        Args:
            user_id: User UUID

        Returns:
            User if found, None otherwise
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, username, display_name, created_at FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            return self._row_to_user(row) if row else None

    def get_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: Username to look up

        Returns:
            User if found, None otherwise
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, username, display_name, created_at FROM users WHERE username = ?",
                (username,)
            ).fetchone()
            return self._row_to_user(row) if row else None

    def get_username(self, user_id: str) -> str | None:
        """Map UUID to username for StoreFactory.

        This is the key mapping function that bridges the web app's
        UUID-based user system with the existing username-based stores.

        Args:
            user_id: User UUID

        Returns:
            Username if found, None otherwise
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT username FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            return row["username"] if row else None

    def create(
        self,
        username: str,
        display_name: str | None = None
    ) -> User:
        """Create a new user.

        Args:
            username: Unique username
            display_name: Optional display name

        Returns:
            Created user

        Raises:
            ValueError: If username already exists
        """
        user_id = str(uuid.uuid4())
        now = datetime.now()

        with get_db_connection(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (id, username, display_name, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, username, display_name, now.isoformat())
                )
                conn.commit()
                logger.info("Created user", username=username, user_id=user_id)
            except Exception as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Username '{username}' already exists")
                raise

        return User(
            id=user_id,
            username=username,
            display_name=display_name,
            created_at=now
        )

    def update_display_name(self, user_id: str, display_name: str) -> bool:
        """Update user's display name.

        Args:
            user_id: User UUID
            display_name: New display name

        Returns:
            True if updated, False if user not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE users SET display_name = ? WHERE id = ?",
                (display_name, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_user(self, row) -> User:
        """Convert database row to User model.

        Args:
            row: SQLite Row object

        Returns:
            User model
        """
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return User(
            id=row["id"],
            username=row["username"],
            display_name=row["display_name"],
            created_at=created_at
        )
