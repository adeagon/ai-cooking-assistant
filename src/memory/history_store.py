"""Cooking history storage using SQLite."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

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
        """Create cooking history table and index if they don't exist."""
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

        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_recipe
            ON cooking_history(recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_cooked_at
            ON cooking_history(cooked_at)
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
            INSERT INTO cooking_history (recipe_id, cooked_at, notes)
            VALUES (?, ?, ?)
            """,
            (recipe_id, datetime.now(), notes),
        )

        history_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info("Recorded cooked recipe", history_id=history_id, recipe_id=recipe_id)

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

    def get_cooking_count(self, recipe_id: str) -> int:
        """Get number of times a recipe has been cooked.

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
            WHERE recipe_id = ?
            """,
            (recipe_id,),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 0
