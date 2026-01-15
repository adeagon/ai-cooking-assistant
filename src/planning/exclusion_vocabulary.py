"""Dataset-derived exclusion vocabulary for meal planning.

This module loads valid exclusion terms from the recipe database to validate
user input. Rather than maintaining a hand-crafted list of valid terms,
we derive the vocabulary from the actual dataset.
"""

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from src.app.logging_config import get_logger
from src.planning.ingredient_normalizer import IngredientNormalizer

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_exclusion_vocabulary(db_path: Path) -> set[str]:
    """Load valid exclusion terms from recipe database.

    Includes:
    - All unique normalized ingredient tokens from recipes
    - All unique tags from recipes

    Results are cached for performance (LRU cache).

    Args:
        db_path: Path to the SQLite database

    Returns:
        Set of valid exclusion terms
    """
    normalizer = IngredientNormalizer()
    vocabulary: set[str] = set()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Load ingredients
        cursor.execute("SELECT DISTINCT ingredients_normalized FROM recipes")
        for (ingredients_json,) in cursor.fetchall():
            if ingredients_json:
                try:
                    ingredients = json.loads(ingredients_json)
                    for ing in ingredients:
                        normalized = normalizer.normalize(ing)
                        # Add individual tokens
                        vocabulary.update(normalized.split())
                except json.JSONDecodeError:
                    pass

        # Load tags
        cursor.execute("SELECT DISTINCT tags FROM recipes")
        for (tags_json,) in cursor.fetchall():
            if tags_json:
                try:
                    tags = json.loads(tags_json)
                    vocabulary.update(t.lower() for t in tags)
                except json.JSONDecodeError:
                    pass

        conn.close()

    except sqlite3.Error as e:
        logger.warning(
            "Failed to load exclusion vocabulary from database",
            error=str(e),
            db_path=str(db_path),
        )

    # Add known category keywords (always valid even if not in DB)
    category_terms = {
        "dairy",
        "meat",
        "seafood",
        "fish",
        "gluten",
        "nuts",
        "eggs",
        "soy",
        "poultry",
        "pork",
        "beef",
        "chicken",
        "turkey",
        "lamb",
        "shellfish",
        "wheat",
        "lactose",
    }
    vocabulary.update(category_terms)

    # Add common dish types (always valid)
    dish_types = {
        "casserole",
        "casseroles",
        "soup",
        "soups",
        "stew",
        "stews",
        "salad",
        "salads",
        "sandwich",
        "sandwiches",
        "pizza",
        "pizzas",
        "curry",
        "curries",
        "pasta",
        "risotto",
        "stir-fry",
        "stirfry",
        "roast",
        "grill",
        "grilled",
        "fried",
        "baked",
        "steamed",
        "braised",
    }
    vocabulary.update(dish_types)

    logger.info("Loaded exclusion vocabulary", term_count=len(vocabulary))
    return vocabulary


def is_valid_exclusion_term(term: str, db_path: Path) -> bool:
    """Check if a term is a valid exclusion (exists in dataset vocabulary).

    Args:
        term: The term to validate
        db_path: Path to the SQLite database

    Returns:
        True if the term is a valid exclusion term
    """
    vocabulary = load_exclusion_vocabulary(db_path)
    return term.lower() in vocabulary


def clear_vocabulary_cache() -> None:
    """Clear the vocabulary cache.

    Useful for testing or when the database has been updated.
    """
    load_exclusion_vocabulary.cache_clear()


def get_vocabulary_size(db_path: Path) -> int:
    """Get the size of the vocabulary.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Number of terms in the vocabulary
    """
    return len(load_exclusion_vocabulary(db_path))
