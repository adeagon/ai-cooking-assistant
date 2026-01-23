"""User preference profile storage using SQLite."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.domain.models import PreferenceProfile
from src.memory.base_store import BaseUserBoundStore

logger = get_logger(__name__)


class ProfileStore(BaseUserBoundStore):
    """Manages persistent storage of user preferences in SQLite.

    Each store instance is bound to a specific user at instantiation.
    The user cannot be changed after initialization.
    """

    def _ensure_table(self) -> None:
        """Create preferences table if it doesn't exist.

        Handles migration from old id-based or user_id-based schema to username-based schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if table exists and has old schema (id-based or user_id-based)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preferences'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check for old schema (id or user_id column, no username column)
            cursor.execute("PRAGMA table_info(preferences)")
            columns = {col[1]: col for col in cursor.fetchall()}

            needs_migration = "username" not in columns and ("id" in columns or "user_id" in columns)

            if needs_migration:
                logger.info("Migrating preferences table to username-based schema")

                # Determine which column has the user identifier
                user_col = "user_id" if "user_id" in columns else "id"

                # Rename old table
                cursor.execute("ALTER TABLE preferences RENAME TO preferences_old")

                # Create new table with username PK
                cursor.execute("""
                    CREATE TABLE preferences (
                        username TEXT PRIMARY KEY,
                        spice_level TEXT DEFAULT 'medium',
                        diet TEXT DEFAULT 'none',
                        avoid_ingredients TEXT,
                        preferred_cuisines TEXT,
                        time_limit_default INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Migrate existing rows - use user_id/id as username, or 'guest' for id=1
                if user_col == "user_id":
                    cursor.execute("""
                        INSERT OR IGNORE INTO preferences (
                            username, spice_level, diet, avoid_ingredients,
                            preferred_cuisines, time_limit_default, created_at, updated_at
                        )
                        SELECT user_id, spice_level, diet, avoid_ingredients,
                               preferred_cuisines, time_limit_default, created_at, updated_at
                        FROM preferences_old
                    """)
                else:
                    cursor.execute("""
                        INSERT OR IGNORE INTO preferences (
                            username, spice_level, diet, avoid_ingredients,
                            preferred_cuisines, time_limit_default, created_at, updated_at
                        )
                        SELECT 'guest', spice_level, diet, avoid_ingredients,
                               preferred_cuisines, time_limit_default, created_at, updated_at
                        FROM preferences_old
                        WHERE id = 1
                    """)

                # Drop old table
                cursor.execute("DROP TABLE preferences_old")

                logger.info("Migrated preferences to username-based schema")
        else:
            # Create new table with username PK
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    username TEXT PRIMARY KEY,
                    spice_level TEXT DEFAULT 'medium',
                    diet TEXT DEFAULT 'none',
                    avoid_ingredients TEXT,
                    preferred_cuisines TEXT,
                    time_limit_default INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()
        conn.close()

        logger.info("Preferences table ensured", db_path=str(self.db_path))

    def load(self) -> PreferenceProfile:
        """Load user preferences from database.

        Returns:
            PreferenceProfile with user preferences, or default profile if none exists
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM preferences WHERE username = ?", (self.user,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            logger.info("No preferences found, returning defaults", user=self.user)
            return PreferenceProfile()

        # Parse JSON arrays
        avoid_ingredients = json.loads(row["avoid_ingredients"]) if row["avoid_ingredients"] else []
        preferred_cuisines = json.loads(row["preferred_cuisines"]) if row["preferred_cuisines"] else []

        profile = PreferenceProfile(
            spice_level=row["spice_level"],
            diet=row["diet"],
            avoid_ingredients=avoid_ingredients,
            preferred_cuisines=preferred_cuisines,
            time_limit_default_minutes=row["time_limit_default"],
        )

        logger.info(
            "Loaded user preferences",
            user=self.user,
            spice_level=profile.spice_level,
            diet=profile.diet,
            avoid_count=len(profile.avoid_ingredients),
        )

        return profile

    def save(self, profile: PreferenceProfile) -> None:
        """Save user preferences to database.

        Args:
            profile: PreferenceProfile to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize lists to JSON
        avoid_ingredients_json = json.dumps(profile.avoid_ingredients)
        preferred_cuisines_json = json.dumps(profile.preferred_cuisines)
        now = datetime.now().isoformat()

        # Insert or replace (upsert)
        cursor.execute(
            """
            INSERT OR REPLACE INTO preferences (
                username, spice_level, diet, avoid_ingredients, preferred_cuisines,
                time_limit_default, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user,
                profile.spice_level,
                profile.diet,
                avoid_ingredients_json,
                preferred_cuisines_json,
                profile.time_limit_default_minutes,
                now,
            ),
        )

        conn.commit()
        conn.close()

        logger.info("Saved user preferences", user=self.user, spice_level=profile.spice_level, diet=profile.diet)

    def update(self, **updates) -> PreferenceProfile:
        """Update specific preference fields.

        Args:
            **updates: Fields to update (spice_level, diet, avoid_ingredients, etc.)

        Returns:
            Updated PreferenceProfile
        """
        # Load current profile
        profile = self.load()

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                logger.warning("Unknown preference field", field=key, user=self.user)

        # Save updated profile
        self.save(profile)

        logger.info("Updated user preferences", user=self.user, updates=list(updates.keys()))

        return profile
