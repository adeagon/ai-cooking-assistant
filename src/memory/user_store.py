"""User account storage using SQLite."""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

from src.app.constants import DEFAULT_USER_USERNAME
from src.app.logging_config import get_logger
from src.memory._table_init import is_table_initialized, mark_table_initialized

logger = get_logger(__name__)


@dataclass
class User:
    """User account data."""

    id: str  # UUID
    username: str
    password_hash: str | None
    created_at: datetime | None
    is_active: bool


class UserStore:
    """Manages persistent storage of user accounts in SQLite."""

    def __init__(self, db_path: Path):
        """Initialize UserStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create users table if it doesn't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        Uses module-level tracking to avoid redundant CREATE TABLE calls.
        """
        if is_table_initialized("users"):
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        conn.commit()
        conn.close()

        mark_table_initialized("users")
        logger.info("Users table ensured", db_path=str(self.db_path))

    def create_user(self, username: str, password: str | None = None) -> User:
        """Create a new user account.

        Args:
            username: Unique username (case-insensitive)
            password: Optional password (None = set on first login)

        Returns:
            Created User object

        Raises:
            sqlite3.IntegrityError: If username already exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password) if password else None
        now = datetime.now()

        try:
            cursor.execute(
                """
                INSERT INTO users (id, username, password_hash, created_at, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (user_id, username.strip(), password_hash, now),
            )
            conn.commit()

            logger.info("Created user", user_id=user_id, username=username)

            return User(
                id=user_id,
                username=username.strip(),
                password_hash=password_hash,
                created_at=now,
                is_active=True,
            )
        except sqlite3.IntegrityError as e:
            logger.warning("Username already exists", username=username)
            raise e
        finally:
            conn.close()

    def create_user_if_not_exists(
        self, username: str, password_hash: str | None = None, user_id: str | None = None
    ) -> User | None:
        """Create a user if they don't already exist.

        Args:
            username: Unique username (case-insensitive)
            password_hash: Optional pre-hashed password
            user_id: Optional specific UUID to use

        Returns:
            Created User object or None if user already exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
        if cursor.fetchone():
            conn.close()
            return None

        # Create user
        uid = user_id or str(uuid.uuid4())
        now = datetime.now()

        cursor.execute(
            """
            INSERT INTO users (id, username, password_hash, created_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (uid, username.strip(), password_hash, now),
        )
        conn.commit()
        conn.close()

        logger.info("Created user (if not exists)", user_id=uid, username=username)

        return User(
            id=uid,
            username=username.strip(),
            password_hash=password_hash,
            created_at=now,
            is_active=True,
        )

    def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by UUID.

        Args:
            user_id: User's UUID

        Returns:
            User object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, password_hash, created_at, is_active
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return User(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            is_active=bool(row["is_active"]),
        )

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username (case-insensitive).

        Args:
            username: Username to look up

        Returns:
            User object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # COLLATE NOCASE handles case-insensitivity
        cursor.execute(
            """
            SELECT id, username, password_hash, created_at, is_active
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return User(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            is_active=bool(row["is_active"]),
        )

    def verify_password(self, username: str, password: str) -> User | None:
        """Verify password for a user.

        Args:
            username: Username to verify
            password: Password to check

        Returns:
            User object if credentials are valid, None otherwise
        """
        user = self.get_user_by_username(username)

        if not user:
            logger.debug("User not found for password verification", username=username)
            return None

        if not user.is_active:
            logger.debug("User is not active", username=username)
            return None

        # Block default_user from password login
        if user.username.lower() == DEFAULT_USER_USERNAME.lower():
            logger.warning("Attempted login as default_user", username=username)
            return None

        if not user.password_hash:
            logger.debug("User has no password set", username=username)
            return None

        if check_password_hash(user.password_hash, password):
            logger.info("Password verified", username=username)
            return user

        logger.debug("Invalid password", username=username)
        return None

    def set_password(self, user_id: str, password: str) -> bool:
        """Set or update user's password.

        Args:
            user_id: User's UUID
            password: New password (will be hashed)

        Returns:
            True if password was set, False if user not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        password_hash = generate_password_hash(password)

        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (password_hash, user_id),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Password set", user_id=user_id)
            return True
        else:
            logger.warning("User not found for password set", user_id=user_id)
            return False

    def set_active(self, user_id: str, is_active: bool) -> bool:
        """Enable or disable a user account.

        Args:
            user_id: User's UUID
            is_active: True to enable, False to disable

        Returns:
            True if updated, False if user not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET is_active = ?
            WHERE id = ?
            """,
            (1 if is_active else 0, user_id),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("User active status updated", user_id=user_id, is_active=is_active)
            return True
        else:
            logger.warning("User not found for active update", user_id=user_id)
            return False

    def list_users(self, include_inactive: bool = False) -> list[User]:
        """List all users.

        Args:
            include_inactive: If True, include inactive users

        Returns:
            List of User objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if include_inactive:
            cursor.execute("""
                SELECT id, username, password_hash, created_at, is_active
                FROM users
                ORDER BY username
            """)
        else:
            cursor.execute("""
                SELECT id, username, password_hash, created_at, is_active
                FROM users
                WHERE is_active = 1
                ORDER BY username
            """)

        users = [
            User(
                id=row["id"],
                username=row["username"],
                password_hash=row["password_hash"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                is_active=bool(row["is_active"]),
            )
            for row in cursor.fetchall()
        ]

        conn.close()
        return users

    def needs_password_setup(self, username: str) -> bool:
        """Check if user exists but has no password set.

        Args:
            username: Username to check

        Returns:
            True if user exists and has no password, False otherwise
        """
        user = self.get_user_by_username(username)
        return user is not None and user.password_hash is None
