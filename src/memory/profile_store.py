"""User preference profile storage using SQLite."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.domain.models import PreferenceProfile
from src.memory._table_init import is_table_initialized, mark_table_initialized

logger = get_logger(__name__)


class ProfileStore:
    """Manages persistent storage of user preferences in SQLite.

    Each user has their own isolated preferences identified by user_id (UUID).
    """

    def __init__(self, db_path: Path, user_id: str):
        """Initialize ProfileStore with database path and user ID.

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
        """Create preferences table if it doesn't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        Uses module-level tracking to avoid redundant CREATE TABLE calls.
        """
        if is_table_initialized("preferences"):
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT PRIMARY KEY,
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

        mark_table_initialized("preferences")
        logger.info("Preferences table ensured", db_path=str(self.db_path))

    def load(self) -> PreferenceProfile:
        """Load user preferences from database.

        Returns:
            PreferenceProfile with user preferences, or default profile if none exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM preferences WHERE user_id = ?", (self.user_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            logger.info("No preferences found, returning defaults", user_id=self.user_id)
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
            user_id=self.user_id,
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
        conn = self._get_connection()
        cursor = conn.cursor()

        # Serialize lists to JSON
        avoid_ingredients_json = json.dumps(profile.avoid_ingredients)
        preferred_cuisines_json = json.dumps(profile.preferred_cuisines)
        now = datetime.now().isoformat()

        # Insert or replace (upsert) for this user
        cursor.execute(
            """
            INSERT OR REPLACE INTO preferences (
                user_id, spice_level, diet, avoid_ingredients, preferred_cuisines,
                time_limit_default, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
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

        logger.info(
            "Saved user preferences",
            user_id=self.user_id,
            spice_level=profile.spice_level,
            diet=profile.diet,
        )

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
                logger.warning("Unknown preference field", field=key)

        # Save updated profile
        self.save(profile)

        logger.info("Updated user preferences", updates=list(updates.keys()))

        return profile
