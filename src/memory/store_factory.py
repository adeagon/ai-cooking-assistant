"""Factory for creating user-scoped store instances."""

from dataclasses import dataclass
from pathlib import Path

from src.memory.profile_store import ProfileStore
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from src.memory.recipe_box_store import RecipeBoxStore
from src.memory.session_store import SessionStore
from src.memory.meal_plan_store import MealPlanStore
from src.app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UserStores:
    """Container for all user-scoped stores."""

    profile: ProfileStore
    feedback: FeedbackStore
    history: HistoryStore
    recipe_box: RecipeBoxStore
    session: SessionStore
    meal_plan: MealPlanStore


class StoreFactory:
    """Factory for creating and caching user-scoped store instances.

    Stores are cached per username to avoid recreation overhead
    when users switch back and forth.
    """

    def __init__(self, db_path: Path):
        """Initialize factory with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._cache: dict[str, UserStores] = {}

    def get_stores(self, username: str) -> UserStores:
        """Get or create cached stores for a user.

        Args:
            username: Username to get stores for

        Returns:
            UserStores container with all stores bound to username
        """
        if username not in self._cache:
            logger.info("Creating stores for user", user=username)
            self._cache[username] = UserStores(
                profile=ProfileStore(self.db_path, username=username),
                feedback=FeedbackStore(self.db_path, username=username),
                history=HistoryStore(self.db_path, username=username),
                recipe_box=RecipeBoxStore(self.db_path, username=username),
                session=SessionStore(self.db_path, username=username),
                meal_plan=MealPlanStore(self.db_path, username=username),
            )
        else:
            logger.debug("Using cached stores for user", user=username)

        return self._cache[username]

    def clear_cache(self, username: str | None = None) -> None:
        """Clear cache for a specific user or all users.

        Args:
            username: Username to clear cache for, or None to clear all
        """
        if username:
            self._cache.pop(username, None)
            logger.info("Cleared store cache for user", user=username)
        else:
            self._cache.clear()
            logger.info("Cleared all store caches")
