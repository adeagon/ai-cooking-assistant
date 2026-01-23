"""Recipe Box storage using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import SavedRecipe
from src.memory.base_store import BaseUserBoundStore

logger = get_logger(__name__)


class RecipeBoxStore(BaseUserBoundStore):
    """Manages persistent storage of saved recipes in SQLite.

    Each store instance is bound to a specific user at instantiation.
    The user cannot be changed after initialization.
    """

    def _ensure_table(self) -> None:
        """Create saved recipes table and index if they don't exist.

        Handles migration from old single-user schema to multi-user schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if table exists and has old schema (no username)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='saved_recipes'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check for old schema (no username column)
            cursor.execute("PRAGMA table_info(saved_recipes)")
            columns = {col[1]: col for col in cursor.fetchall()}

            if "username" not in columns:
                logger.info("Migrating saved_recipes table to multi-user schema")

                # Rename old table
                cursor.execute("ALTER TABLE saved_recipes RENAME TO saved_recipes_old")

                # Create new table with username and composite unique constraint
                cursor.execute("""
                    CREATE TABLE saved_recipes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL DEFAULT 'guest',
                        recipe_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT,
                        UNIQUE(username, recipe_id),
                        FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
                    )
                """)

                # Migrate data (assign all to 'guest')
                cursor.execute("""
                    INSERT INTO saved_recipes (id, username, recipe_id, title, saved_at, notes)
                    SELECT id, 'guest', recipe_id, title, saved_at, notes
                    FROM saved_recipes_old
                """)

                # Drop old table
                cursor.execute("DROP TABLE saved_recipes_old")

                logger.info("Migrated saved_recipes to multi-user schema")
        else:
            # Create new table with username and composite unique constraint
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_recipes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL DEFAULT 'guest',
                    recipe_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    UNIQUE(username, recipe_id),
                    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
                )
            """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_recipe
            ON saved_recipes(recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_at
            ON saved_recipes(saved_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_saved_recipes_username
            ON saved_recipes(username)
        """)

        conn.commit()
        conn.close()

        logger.info("Saved recipes table ensured", db_path=str(self.db_path))

    def save_recipe(self, recipe_id: str, title: str, notes: str | None = None) -> int:
        """Save a recipe to the Recipe Box.

        Args:
            recipe_id: Recipe ID to save
            title: Recipe title (for display without DB join)
            notes: Optional user notes about the recipe

        Returns:
            ID of the inserted saved recipe record

        Raises:
            sqlite3.IntegrityError: If recipe is already saved by this user
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO saved_recipes (username, recipe_id, title, saved_at, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.user, recipe_id, title, datetime.now(), notes),
            )

            saved_id = cursor.lastrowid
            conn.commit()

            logger.info("Saved recipe to box", saved_id=saved_id, recipe_id=recipe_id, user=self.user)
            return saved_id

        except sqlite3.IntegrityError as e:
            logger.warning("Recipe already saved", recipe_id=recipe_id, user=self.user)
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
            WHERE username = ?
            ORDER BY saved_at DESC
            LIMIT ?
            """,
            (self.user, limit),
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
            WHERE recipe_id = ? AND username = ?
            """,
            (recipe_id, self.user),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Removed recipe from box", recipe_id=recipe_id, user=self.user)
            return True
        else:
            logger.warning("Recipe not found in box", recipe_id=recipe_id, user=self.user)
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
            WHERE recipe_id = ? AND username = ?
            """,
            (recipe_id, self.user),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] > 0 if result else False
