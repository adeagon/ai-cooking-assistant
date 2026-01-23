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
    """Manages persistent storage of saved recipes in SQLite."""

    def __init__(self, db_path: Path):
        """Initialize RecipeBoxStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create saved recipes table and index if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_recipe
            ON saved_recipes(recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_at
            ON saved_recipes(saved_at)
        """)

        conn.commit()
        conn.close()

        logger.info("Saved recipes table ensured", db_path=str(self.db_path))

    def save_recipe(self, recipe_id: str, title: str, notes: str | None = None, user_id: str | None = None) -> int:
        """Save a recipe to the Recipe Box.

        Args:
            recipe_id: Recipe ID to save
            title: Recipe title (for display without DB join)
            notes: Optional user notes about the recipe
            user_id: User ID (reserved for Phase 2 multi-user support)

        Returns:
            ID of the inserted saved recipe record

        Raises:
            sqlite3.IntegrityError: If recipe is already saved (UNIQUE constraint)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO saved_recipes (recipe_id, title, saved_at, notes)
                VALUES (?, ?, ?, ?)
                """,
                (recipe_id, title, datetime.now(), notes),
            )

            saved_id = cursor.lastrowid
            conn.commit()

            logger.info("Saved recipe to box", saved_id=saved_id, recipe_id=recipe_id)
            return saved_id

        except sqlite3.IntegrityError as e:
            logger.warning("Recipe already saved", recipe_id=recipe_id)
            raise e
        finally:
            conn.close()

    def get_saved_recipes(self, limit: int = 50) -> list[SavedRecipe]:
        """Get saved recipes from the Recipe Box.

        Args:
            limit: Maximum number of saved recipes to return

        Returns:
            List of SavedRecipe objects, most recently saved first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, title, saved_at, notes
            FROM saved_recipes
            ORDER BY saved_at DESC
            LIMIT ?
            """,
            (limit,),
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
        """Remove a recipe from the Recipe Box.

        Args:
            recipe_id: Recipe ID to remove

        Returns:
            True if recipe was removed, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM saved_recipes
            WHERE recipe_id = ?
            """,
            (recipe_id,),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Removed recipe from box", recipe_id=recipe_id)
            return True
        else:
            logger.warning("Recipe not found in box", recipe_id=recipe_id)
            return False

    def is_saved(self, recipe_id: str) -> bool:
        """Check if a recipe is saved in the Recipe Box.

        Args:
            recipe_id: Recipe ID to check

        Returns:
            True if recipe is saved, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM saved_recipes
            WHERE recipe_id = ?
            """,
            (recipe_id,),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] > 0 if result else False
