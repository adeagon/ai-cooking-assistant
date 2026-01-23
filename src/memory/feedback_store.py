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
    """Manages persistent storage of recipe feedback (likes, dislikes, ratings) in SQLite."""

    def __init__(self, db_path: Path):
        """Initialize FeedbackStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create feedback table and indexes if they don't exist.

        Handles migration from old single-user schema to multi-user schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)

        # Check if username column exists, add if missing (migration)
        cursor.execute("PRAGMA table_info(recipe_feedback)")
        columns = {col[1]: col for col in cursor.fetchall()}

        if "username" not in columns:
            logger.info("Adding username column to recipe_feedback table")
            cursor.execute("ALTER TABLE recipe_feedback ADD COLUMN username TEXT")
            # Assign existing rows to 'guest'
            cursor.execute("UPDATE recipe_feedback SET username = 'guest' WHERE username IS NULL")

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_recipe
            ON recipe_feedback(recipe_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_type
            ON recipe_feedback(feedback_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_username
            ON recipe_feedback(username)
        """)

        conn.commit()
        conn.close()

        logger.info("Recipe feedback table ensured", db_path=str(self.db_path))

    def add_feedback(self, feedback: RecipeFeedback, user_id: str | None = None) -> int:
        """Store recipe feedback.

        Args:
            feedback: RecipeFeedback object
            user_id: Username to store feedback for. Defaults to 'guest' if None.

        Returns:
            ID of the inserted feedback record
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO recipe_feedback
                (recipe_id, feedback_type, rating, session_id, created_at, username)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.recipe_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.session_id,
                feedback.created_at or datetime.now(),
                username,
            ),
        )

        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            "Added feedback",
            feedback_id=feedback_id,
            recipe_id=feedback.recipe_id,
            feedback_type=feedback.feedback_type,
            username=username,
        )

        return feedback_id

    def get_liked_recipe_ids(self, limit: int = 50, user_id: str | None = None) -> set[str]:
        """Get recently liked recipe IDs for exclusion.

        Args:
            limit: Maximum number of recent likes to return
            user_id: Username to get likes for. Defaults to 'guest' if None.

        Returns:
            Set of recipe IDs that have been liked
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM recipe_feedback
            WHERE feedback_type = 'like' AND username = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (username, limit),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug("Retrieved liked recipe IDs", count=len(recipe_ids), username=username)
        return recipe_ids

    def get_disliked_recipe_ids(self, user_id: str | None = None) -> set[str]:
        """Get all disliked recipe IDs for exclusion.

        Args:
            user_id: Username to get dislikes for. Defaults to 'guest' if None.

        Returns:
            Set of recipe IDs that have been disliked
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT recipe_id
            FROM recipe_feedback
            WHERE feedback_type = 'dislike' AND username = ?
            """,
            (username,),
        )

        recipe_ids = {row["recipe_id"] for row in cursor.fetchall()}
        conn.close()

        logger.debug("Retrieved disliked recipe IDs", count=len(recipe_ids), username=username)
        return recipe_ids

    def get_feedback_for_recipe(self, recipe_id: str, user_id: str | None = None) -> list[RecipeFeedback]:
        """Get all feedback for a specific recipe by a user.

        Args:
            recipe_id: Recipe ID to query
            user_id: Username to get feedback for. Defaults to 'guest' if None.

        Returns:
            List of RecipeFeedback objects for this recipe
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, recipe_id, feedback_type, rating, session_id, created_at
            FROM recipe_feedback
            WHERE recipe_id = ? AND username = ?
            ORDER BY created_at DESC
            """,
            (recipe_id, username),
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

    def get_average_rating(self, recipe_id: str, user_id: str | None = None) -> float | None:
        """Get average user rating for a recipe.

        Args:
            recipe_id: Recipe ID to query
            user_id: Username to get rating for. If None, returns global average.

        Returns:
            Average rating (1-5) or None if no ratings exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            # Per-user average
            cursor.execute(
                """
                SELECT AVG(rating) as avg_rating
                FROM recipe_feedback
                WHERE recipe_id = ? AND feedback_type = 'rate' AND rating IS NOT NULL AND username = ?
                """,
                (recipe_id, user_id),
            )
        else:
            # Global average (all users)
            cursor.execute(
                """
                SELECT AVG(rating) as avg_rating
                FROM recipe_feedback
                WHERE recipe_id = ? AND feedback_type = 'rate' AND rating IS NOT NULL
                """,
                (recipe_id,),
            )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] is not None else None

    def get_preferred_cuisines_from_likes(self, min_count: int = 3, user_id: str | None = None) -> list[str]:
        """Analyze liked recipes to find preferred cuisines.

        Args:
            min_count: Minimum number of likes for a cuisine to be considered preferred
            user_id: Username to analyze likes for. Defaults to 'guest' if None.

        Returns:
            List of preferred cuisine names
        """
        # Default to guest if no user_id provided
        username = user_id or "guest"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Join with recipes table to get tags
        # Tags are stored as JSON array, need to extract cuisine tags
        cursor.execute(
            """
            SELECT r.tags
            FROM recipe_feedback f
            JOIN recipes r ON f.recipe_id = r.recipe_id
            WHERE f.feedback_type = 'like' AND f.username = ?
            """,
            (username,),
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

        logger.info("Extracted preferred cuisines from likes", cuisines=preferred, username=username)
        return preferred
