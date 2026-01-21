"""Recipe feedback storage using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import RecipeFeedback

logger = get_logger(__name__)


class FeedbackStore:
    """Manages persistent storage of recipe feedback (likes, dislikes, ratings) in SQLite.

    Each user has their own isolated feedback identified by user_id (UUID).
    """

    def __init__(self, db_path: Path, user_id: str):
        """Initialize FeedbackStore with database path and user ID.

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
        """Create feedback table and indexes if they don't exist.

        Note: The full schema migration is handled by scripts/migrate_multiuser.py.
        This method ensures backward compatibility for fresh databases.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_user_recipe
            ON recipe_feedback(user_id, recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_type
            ON recipe_feedback(feedback_type)
        """)

        conn.commit()
        conn.close()

        logger.info("Recipe feedback table ensured", db_path=str(self.db_path))

    def add_feedback(self, feedback: RecipeFeedback) -> int:
        """Store recipe feedback for current user.

        Args:
            feedback: RecipeFeedback object

        Returns:
            ID of the inserted feedback record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO recipe_feedback
                (user_id, recipe_id, feedback_type, rating, session_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
                feedback.recipe_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.session_id,
                feedback.created_at or datetime.now(),
            ),
        )

        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            "Added feedback",
            feedback_id=feedback_id,
            user_id=self.user_id,
            recipe_id=feedback.recipe_id,
            feedback_type=feedback.feedback_type,
        )

        return feedback_id

    def get_liked_recipe_ids(self, limit: int = 50) -> set[str]:
        """Get recently liked recipe IDs for current user for exclusion.

        Args:
            limit: Maximum number of recent likes to return

        Returns:
            Set of recipe IDs that have been liked by current user
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM recipe_feedback
            WHERE user_id = ? AND feedback_type = 'like'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self.user_id, limit),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug("Retrieved liked recipe IDs", user_id=self.user_id, count=len(recipe_ids))
        return recipe_ids

    def get_disliked_recipe_ids(self) -> set[str]:
        """Get all disliked recipe IDs for current user for exclusion.

        Returns:
            Set of recipe IDs that have been disliked by current user
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM recipe_feedback
            WHERE user_id = ? AND feedback_type = 'dislike'
            """,
            (self.user_id,),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug("Retrieved disliked recipe IDs", user_id=self.user_id, count=len(recipe_ids))
        return recipe_ids

    def get_feedback_for_recipe(self, recipe_id: str) -> list[RecipeFeedback]:
        """Get all feedback for a specific recipe by current user.

        Args:
            recipe_id: Recipe ID to query

        Returns:
            List of RecipeFeedback objects for this recipe by current user
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, feedback_type, rating, session_id, created_at
            FROM recipe_feedback
            WHERE user_id = ? AND recipe_id = ?
            ORDER BY created_at DESC
            """,
            (self.user_id, recipe_id),
        )

        feedbacks = [
            RecipeFeedback(
                id=row["id"],
                recipe_id=row["recipe_id"],
                feedback_type=row["feedback_type"],
                rating=row["rating"],
                session_id=row["session_id"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            )
            for row in cursor.fetchall()
        ]

        conn.close()

        return feedbacks

    def get_average_rating(self, recipe_id: str) -> float | None:
        """Get average user rating for a recipe by current user.

        Args:
            recipe_id: Recipe ID to query

        Returns:
            Average rating (1-5) or None if no ratings exist
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT AVG(rating) as avg_rating
            FROM recipe_feedback
            WHERE user_id = ? AND recipe_id = ? AND feedback_type = 'rate' AND rating IS NOT NULL
            """,
            (self.user_id, recipe_id),
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] is not None else None

    def get_preferred_cuisines_from_likes(self, min_count: int = 3) -> list[str]:
        """Analyze liked recipes to find preferred cuisines for current user.

        Args:
            min_count: Minimum number of likes for a cuisine to be considered preferred

        Returns:
            List of preferred cuisine names
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Join with recipes table to get tags
        # Tags are stored as JSON array, need to extract cuisine tags
        cursor.execute(
            """
            SELECT r.tags
            FROM recipe_feedback f
            JOIN recipes r ON f.recipe_id = r.recipe_id
            WHERE f.user_id = ? AND f.feedback_type = 'like'
            """,
            (self.user_id,),
        )

        # Parse tags and extract cuisines
        # Cuisine tags are typically like "italian", "mexican", "asian", etc.
        import json
        cuisine_counts: dict[str, int] = {}

        known_cuisines = {
            "italian", "mexican", "asian", "chinese", "japanese", "indian",
            "thai", "greek", "french", "american", "southern", "korean",
            "mediterranean", "middle-eastern", "spanish", "vietnamese",
        }

        for row in cursor.fetchall():
            if row["tags"]:
                tags = json.loads(row["tags"])
                for tag in tags:
                    tag_lower = tag.lower().replace("-", " ").replace("_", " ")
                    if tag_lower in known_cuisines:
                        cuisine_counts[tag_lower] = cuisine_counts.get(tag_lower, 0) + 1

        conn.close()

        # Filter by min_count and sort by frequency
        preferred = [
            cuisine
            for cuisine, count in cuisine_counts.items()
            if count >= min_count
        ]
        preferred.sort(key=lambda c: cuisine_counts[c], reverse=True)

        logger.info("Extracted preferred cuisines from likes", user_id=self.user_id, cuisines=preferred)
        return preferred
