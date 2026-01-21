"""Cooking history storage using SQLite."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import CookingHistoryEntry

logger = get_logger(__name__)


class HistoryStore:
    """Manages persistent storage of cooking history in SQLite.

    Each user has their own isolated cooking history identified by user_id (UUID).
    """

    def __init__(self, db_path: Path, user_id: str):
        """Initialize HistoryStore with database path and user ID.

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
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create cooking history table and index if they don't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cooking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                cooked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_user_date
            ON cooking_history(user_id, cooked_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_recipe
            ON cooking_history(recipe_id)
        """)

        conn.commit()
        conn.close()

        logger.info("Cooking history table ensured", db_path=str(self.db_path))

    def add_cooked(self, recipe_id: str, notes: str | None = None) -> int:
        """Record a cooked recipe for current user.

        Args:
            recipe_id: Recipe ID that was cooked
            notes: Optional user notes about the cooking experience

        Returns:
            ID of the inserted history record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cooking_history (user_id, recipe_id, cooked_at, notes)
            VALUES (?, ?, ?, ?)
            """,
            (self.user_id, recipe_id, datetime.now(), notes),
        )

        history_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info("Recorded cooked recipe", history_id=history_id, user_id=self.user_id, recipe_id=recipe_id)

        return history_id

    def get_recently_cooked_ids(self, days: int = 14) -> set[str]:
        """Get recipe IDs cooked by current user in the last N days for exclusion.

        Args:
            days: Number of days to look back

        Returns:
            Set of recipe IDs cooked by current user in the last N days
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM cooking_history
            WHERE user_id = ? AND cooked_at >= ?
            """,
            (self.user_id, cutoff_date),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug(
            "Retrieved recently cooked recipe IDs",
            user_id=self.user_id,
            count=len(recipe_ids),
            days=days,
        )
        return recipe_ids

    def get_cooking_history(self, limit: int = 20) -> list[CookingHistoryEntry]:
        """Get recent cooking history for current user.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of CookingHistoryEntry objects, most recent first
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, cooked_at, notes
            FROM cooking_history
            WHERE user_id = ?
            ORDER BY cooked_at DESC
            LIMIT ?
            """,
            (self.user_id, limit),
        )

        history = [
            CookingHistoryEntry(
                id=row["id"],
                recipe_id=row["recipe_id"],
                cooked_at=datetime.fromisoformat(row["cooked_at"]) if row["cooked_at"] else None,
                notes=row["notes"],
            )
            for row in cursor.fetchall()
        ]

        conn.close()

        return history

    def get_cooking_count(self, recipe_id: str) -> int:
        """Get number of times a recipe has been cooked by current user.

        Args:
            recipe_id: Recipe ID to query

        Returns:
            Number of times this recipe has been cooked by current user
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM cooking_history
            WHERE user_id = ? AND recipe_id = ?
            """,
            (self.user_id, recipe_id),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0
