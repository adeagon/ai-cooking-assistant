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
    """Manages persistent storage of cooking history in SQLite."""

    def __init__(self, db_path: Path):
        """Initialize HistoryStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create cooking history table and index if they don't exist.

        Handles migration from old single-user schema to multi-user schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cooking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL,
                cooked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Check if username column exists, add if missing (migration)
        cursor.execute("PRAGMA table_info(cooking_history)")
        columns = {col[1]: col for col in cursor.fetchall()}

        if "username" not in columns:
            logger.info("Adding username column to cooking_history table")
            cursor.execute("ALTER TABLE cooking_history ADD COLUMN username TEXT")
            # Assign existing rows to 'guest'
            cursor.execute("UPDATE cooking_history SET username = 'guest' WHERE username IS NULL")

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_recipe
            ON cooking_history(recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_cooked_at
            ON cooking_history(cooked_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_username
            ON cooking_history(username)
        """)

        conn.commit()
        conn.close()

        logger.info("Cooking history table ensured", db_path=str(self.db_path))

    def add_cooked(self, recipe_id: str, notes: str | None = None, user_id: str | None = None) -> int:
        """Record a cooked recipe.

        Args:
            recipe_id: Recipe ID that was cooked
            notes: Optional user notes about the cooking experience
            user_id: Username to record for. Defaults to 'guest' if None.

        Returns:
            ID of the inserted history record
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cooking_history (recipe_id, cooked_at, notes, username)
            VALUES (?, ?, ?, ?)
            """,
            (recipe_id, datetime.now(), notes, username),
        )

        history_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info("Recorded cooked recipe", history_id=history_id, recipe_id=recipe_id, username=username)

        return history_id

    def get_recently_cooked_ids(self, days: int = 14, user_id: str | None = None) -> set[str]:
        """Get recipe IDs cooked in the last N days for exclusion.

        Args:
            days: Number of days to look back
            user_id: Username to get history for. If None, returns for all users.

        Returns:
            Set of recipe IDs cooked in the last N days
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        if user_id:
            cursor.execute(
                """
                SELECT DISTINCT recipe_id
                FROM cooking_history
                WHERE cooked_at >= ? AND username = ?
                """,
                (cutoff_date, user_id),
            )
        else:
            cursor.execute(
                """
                SELECT DISTINCT recipe_id
                FROM cooking_history
                WHERE cooked_at >= ?
                """,
                (cutoff_date,),
            )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug(
            "Retrieved recently cooked recipe IDs",
            count=len(recipe_ids),
            days=days,
            user_id=user_id,
        )
        return recipe_ids

    def get_cooking_history(self, limit: int = 20, user_id: str | None = None) -> list[CookingHistoryEntry]:
        """Get recent cooking history.

        Args:
            limit: Maximum number of history entries to return
            user_id: Username to get history for. If None, returns for all users.

        Returns:
            List of CookingHistoryEntry objects, most recent first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT id, recipe_id, cooked_at, notes
                FROM cooking_history
                WHERE username = ?
                ORDER BY cooked_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, recipe_id, cooked_at, notes
                FROM cooking_history
                ORDER BY cooked_at DESC
                LIMIT ?
                """,
                (limit,),
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

    def get_cooking_count(self, recipe_id: str, user_id: str | None = None) -> int:
        """Get number of times a recipe has been cooked.

        Args:
            recipe_id: Recipe ID to query
            user_id: Username to get count for. If None, returns count for all users.

        Returns:
            Number of times this recipe has been cooked
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM cooking_history
                WHERE recipe_id = ? AND username = ?
                """,
                (recipe_id, user_id),
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM cooking_history
                WHERE recipe_id = ?
                """,
                (recipe_id,),
            )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0
