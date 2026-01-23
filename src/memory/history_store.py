"""Cooking history storage using SQLite."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import CookingHistoryEntry
from src.memory.base_store import BaseUserBoundStore

logger = get_logger(__name__)


class HistoryStore(BaseUserBoundStore):
    """Manages persistent storage of cooking history in SQLite.

    Each store instance is bound to a specific user at instantiation.
    The user cannot be changed after initialization.
    """

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

    def add_cooked(self, recipe_id: str, notes: str | None = None) -> int:
        """Record a cooked recipe.

        Args:
            recipe_id: Recipe ID that was cooked
            notes: Optional user notes about the cooking experience

        Returns:
            ID of the inserted history record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cooking_history (recipe_id, cooked_at, notes, username)
            VALUES (?, ?, ?, ?)
            """,
            (recipe_id, datetime.now(), notes, self.user),
        )

        history_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info("Recorded cooked recipe", history_id=history_id, recipe_id=recipe_id, user=self.user)

        return history_id

    def get_recently_cooked_ids(self, days: int = 14) -> set[str]:
        """Get recipe IDs cooked in the last N days for exclusion.

        Args:
            days: Number of days to look back

        Returns:
            Set of recipe IDs cooked in the last N days
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM cooking_history
            WHERE cooked_at >= ? AND username = ?
            """,
            (cutoff_date, self.user),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug(
            "Retrieved recently cooked recipe IDs",
            count=len(recipe_ids),
            days=days,
            user=self.user,
        )
        return recipe_ids

    def get_cooking_history(self, limit: int = 20) -> list[CookingHistoryEntry]:
        """Get recent cooking history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of CookingHistoryEntry objects, most recent first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, cooked_at, notes
            FROM cooking_history
            WHERE username = ?
            ORDER BY cooked_at DESC
            LIMIT ?
            """,
            (self.user, limit),
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
        """Get number of times a recipe has been cooked by this user.

        Args:
            recipe_id: Recipe ID to query

        Returns:
            Number of times this recipe has been cooked
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM cooking_history
            WHERE recipe_id = ? AND username = ?
            """,
            (recipe_id, self.user),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0
