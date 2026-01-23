"""User memory: preferences, session state, and summaries."""

# Import sqlite compat first to register datetime adapters (fixes Python 3.12+ warning)
from src.memory import _sqlite_compat  # noqa: F401

from src.memory.base_store import BaseUserBoundStore
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from src.memory.meal_plan_store import MealPlanStore
from src.memory.profile_store import ProfileStore
from src.memory.recipe_box_store import RecipeBoxStore
from src.memory.session_store import SessionStore
from src.memory.store_factory import StoreFactory, UserStores
from src.memory.summarizer import RollingSummarizer

__all__ = [
    "BaseUserBoundStore",
    "FeedbackStore",
    "HistoryStore",
    "MealPlanStore",
    "ProfileStore",
    "RecipeBoxStore",
    "SessionStore",
    "StoreFactory",
    "UserStores",
    "RollingSummarizer",
]
