"""Recipe Box storage using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import SavedRecipe

logger = get_logger(__name__)


class RecipeBoxStore:
    """Manages persistent storage of saved recipes in SQLite.

    Each user has their own isolated recipe box identified by user_id (UUID).
    """

    def __init__(self, db_path: Path, user_id: str):
        """Initialize RecipeBoxStore with database path and user ID.

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
        """Create saved recipes table and index if they don't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                title TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                UNIQUE(user_id, recipe_id),
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_user
            ON saved_recipes(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_recipe
            ON saved_recipes(recipe_id)
        """)

        conn.commit()
        conn.close()

        logger.info("Saved recipes table ensured", db_path=str(self.db_path))

    def save_recipe(self, recipe_id: str, title: str, notes: str | None = None) -> int:
        """Save a recipe to the Recipe Box for current user.

        Args:
            recipe_id: Recipe ID to save
            title: Recipe title (for display without DB join)
            notes: Optional user notes about the recipe

        Returns:
            ID of the inserted saved recipe record

        Raises:
            sqlite3.IntegrityError: If recipe is already saved by this user (UNIQUE constraint)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO saved_recipes (user_id, recipe_id, title, saved_at, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.user_id, recipe_id, title, datetime.now(), notes),
            )

            saved_id = cursor.lastrowid
            conn.commit()

            logger.info("Saved recipe to box", saved_id=saved_id, user_id=self.user_id, recipe_id=recipe_id)
            return saved_id

        except sqlite3.IntegrityError as e:
            logger.warning("Recipe already saved", user_id=self.user_id, recipe_id=recipe_id)
            raise e
        finally:
            conn.close()

    def get_saved_recipes(self, limit: int = 50) -> list[SavedRecipe]:
        """Get saved recipes from current user's Recipe Box.

        Args:
            limit: Maximum number of saved recipes to return

        Returns:
            List of SavedRecipe objects, most recently saved first
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, title, saved_at, notes
            FROM saved_recipes
            WHERE user_id = ?
            ORDER BY saved_at DESC
            LIMIT ?
            """,
            (self.user_id, limit),
        )

        saved_recipes = [
            SavedRecipe(
                id=row["id"],
                recipe_id=row["recipe_id"],
                title=row["title"],
                saved_at=datetime.fromisoformat(row["saved_at"]) if row["saved_at"] else None,
                notes=row["notes"],
            )
            for row in cursor.fetchall()
        ]

        conn.close()

        return saved_recipes

    def remove_recipe(self, recipe_id: str) -> bool:
        """Remove a recipe from current user's Recipe Box.

        Args:
            recipe_id: Recipe ID to remove

        Returns:
            True if recipe was removed, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM saved_recipes
            WHERE user_id = ? AND recipe_id = ?
            """,
            (self.user_id, recipe_id),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Removed recipe from box", user_id=self.user_id, recipe_id=recipe_id)
            return True
        else:
            logger.warning("Recipe not found in box", user_id=self.user_id, recipe_id=recipe_id)
            return False

    def is_saved(self, recipe_id: str) -> bool:
        """Check if a recipe is saved in current user's Recipe Box.

        Args:
            recipe_id: Recipe ID to check

        Returns:
            True if recipe is saved by current user, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM saved_recipes
            WHERE user_id = ? AND recipe_id = ?
            """,
            (self.user_id, recipe_id),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] > 0 if result else False
