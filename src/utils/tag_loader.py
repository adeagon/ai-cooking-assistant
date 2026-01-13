"""Load valid cuisines and goals from recipe database."""

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from src.app.logging_config import get_logger

logger = get_logger(__name__)

# Known cuisine tags in Food.com data
CUISINE_TAGS = {
    "asian",
    "american",
    "italian",
    "mexican",
    "chinese",
    "japanese",
    "indian",
    "thai",
    "french",
    "greek",
    "korean",
    "vietnamese",
    "spanish",
    "middle-eastern",
    "mediterranean",
    "african",
    "caribbean",
    "european",
    "north-american",
    "south-american",
    "central-american",
    "british-columbian",
    "southwestern-united-states",
    "southern-united-states",
    "moroccan",
    "lebanese",
    "turkish",
    "portuguese",
    "brazilian",
    "south-african",
    "native-american",
    "south-west-pacific",
}

# Known goal/taste tags
GOAL_TAGS = {
    "sweet",
    "savory",
    "spicy",
    "healthy",
    "comfort-food",
    "low-calorie",
    "high-protein",
    "easy",
    "inexpensive",
    "kid-friendly",
    "weeknight",
    "beginner-cook",
}

# Fallback mappings for user-friendly terms → actual tags
GOAL_FALLBACKS = {
    "light": "low-calorie",
    "hearty": "comfort-food",
    "filling": "comfort-food",
    "cheap": "inexpensive",
    "budget": "inexpensive",
    "quick": "easy",
    "simple": "easy",
    "protein": "high-protein",
    "comfort": "comfort-food",
}


@lru_cache(maxsize=1)
def load_cuisines_from_db(db_path: str | None = None) -> set[str]:
    """Load cuisines that actually exist in recipes.

    Args:
        db_path: Path to SQLite database. If None, uses default from Settings.

    Returns:
        Set of cuisine tag strings (e.g., {"asian", "italian", "korean", ...})
    """
    if db_path is None:
        from src.app.settings import Settings

        db_path = str(Settings().sqlite_db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT tags FROM recipes WHERE tags IS NOT NULL")

    found = set()
    for (tags_json,) in cursor:
        try:
            tags = json.loads(tags_json)
            found.update(tag for tag in tags if tag in CUISINE_TAGS)
        except json.JSONDecodeError:
            continue

    conn.close()

    logger.info("Loaded cuisines from DB", count=len(found))
    return found


@lru_cache(maxsize=1)
def load_goals_from_db(db_path: str | None = None) -> set[str]:
    """Load goal tags that actually exist in recipes.

    Args:
        db_path: Path to SQLite database. If None, uses default from Settings.

    Returns:
        Set of goal tag strings (e.g., {"savory", "healthy", "spicy", ...})
    """
    if db_path is None:
        from src.app.settings import Settings

        db_path = str(Settings().sqlite_db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT tags FROM recipes WHERE tags IS NOT NULL")

    found = set()
    for (tags_json,) in cursor:
        try:
            tags = json.loads(tags_json)
            found.update(tag for tag in tags if tag in GOAL_TAGS)
        except json.JSONDecodeError:
            continue

    conn.close()

    logger.info("Loaded goals from DB", count=len(found))
    return found


def resolve_goal(user_term: str) -> str | None:
    """Resolve user term to actual tag, using fallbacks if needed.

    Args:
        user_term: User's input term (e.g., "light", "cheap", "hearty")

    Returns:
        Mapped goal tag or None if not recognized
    """
    term = user_term.lower().strip()

    # Direct match
    if term in GOAL_TAGS:
        return term

    # Fallback mapping
    if term in GOAL_FALLBACKS:
        return GOAL_FALLBACKS[term]

    return None
