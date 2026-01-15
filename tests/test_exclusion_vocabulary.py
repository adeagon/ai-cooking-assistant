"""Tests for exclusion vocabulary loading and validation."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.planning.exclusion_vocabulary import (
    clear_vocabulary_cache,
    get_vocabulary_size,
    is_valid_exclusion_term,
    load_exclusion_vocabulary,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with test recipes."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create recipes table
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            ingredients_normalized TEXT,
            tags TEXT
        )
    """)

    # Insert test recipes
    recipes = [
        (
            "1",
            "Chicken Stir Fry",
            json.dumps(["chicken", "soy sauce", "garlic", "ginger", "broccoli"]),
            json.dumps(["asian", "quick", "healthy"]),
        ),
        (
            "2",
            "Italian Pasta",
            json.dumps(["pasta", "tomato", "basil", "parmesan cheese", "olive oil"]),
            json.dumps(["italian", "pasta", "vegetarian"]),
        ),
        (
            "3",
            "Beef Tacos",
            json.dumps(["ground beef", "tortilla", "cheese", "lettuce", "tomato"]),
            json.dumps(["mexican", "quick"]),
        ),
    ]

    cursor.executemany(
        "INSERT INTO recipes VALUES (?, ?, ?, ?)",
        recipes,
    )

    conn.commit()
    conn.close()

    # Clear cache to ensure fresh load
    clear_vocabulary_cache()

    yield db_path

    # Cleanup
    clear_vocabulary_cache()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def empty_db():
    """Create a temporary empty database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            ingredients_normalized TEXT,
            tags TEXT
        )
    """)

    conn.commit()
    conn.close()

    clear_vocabulary_cache()

    yield db_path

    clear_vocabulary_cache()
    db_path.unlink(missing_ok=True)


class TestLoadExclusionVocabulary:
    """Test vocabulary loading from database."""

    def test_loads_ingredients(self, temp_db):
        """Vocabulary includes ingredients from recipes."""
        vocab = load_exclusion_vocabulary(temp_db)

        # Check ingredients are loaded
        assert "chicken" in vocab
        assert "pasta" in vocab
        assert "broccoli" in vocab
        assert "lettuce" in vocab

    def test_loads_tags(self, temp_db):
        """Vocabulary includes tags from recipes."""
        vocab = load_exclusion_vocabulary(temp_db)

        # Check tags are loaded
        assert "asian" in vocab
        assert "italian" in vocab
        assert "mexican" in vocab
        assert "vegetarian" in vocab

    def test_includes_category_keywords(self, temp_db):
        """Vocabulary always includes category keywords."""
        vocab = load_exclusion_vocabulary(temp_db)

        # Category keywords are always present
        assert "dairy" in vocab
        assert "meat" in vocab
        assert "seafood" in vocab
        assert "gluten" in vocab
        assert "nuts" in vocab
        assert "eggs" in vocab

    def test_includes_dish_types(self, temp_db):
        """Vocabulary always includes dish types."""
        vocab = load_exclusion_vocabulary(temp_db)

        # Dish types are always present
        assert "casserole" in vocab
        assert "soup" in vocab
        assert "stew" in vocab
        assert "salad" in vocab

    def test_empty_db_still_has_category_terms(self, empty_db):
        """Empty database still includes category and dish terms."""
        vocab = load_exclusion_vocabulary(empty_db)

        # Category keywords always present
        assert "dairy" in vocab
        assert "chicken" in vocab

        # Dish types always present
        assert "casserole" in vocab

    def test_caching_works(self, temp_db):
        """Vocabulary is cached (same object returned)."""
        vocab1 = load_exclusion_vocabulary(temp_db)
        vocab2 = load_exclusion_vocabulary(temp_db)

        # Should be the same object
        assert vocab1 is vocab2

    def test_normalizes_ingredients(self, temp_db):
        """Ingredients are normalized before adding to vocabulary."""
        vocab = load_exclusion_vocabulary(temp_db)

        # "ground beef" becomes "ground_beef" which splits to "ground" and "beef"
        # or stays as "ground_beef" depending on normalizer
        assert "beef" in vocab or "ground_beef" in vocab


class TestIsValidExclusionTerm:
    """Test term validation."""

    def test_valid_ingredient(self, temp_db):
        """Valid ingredients return True."""
        assert is_valid_exclusion_term("chicken", temp_db)
        assert is_valid_exclusion_term("tomato", temp_db)

    def test_valid_tag(self, temp_db):
        """Valid tags return True."""
        assert is_valid_exclusion_term("italian", temp_db)
        assert is_valid_exclusion_term("asian", temp_db)

    def test_valid_category(self, temp_db):
        """Valid category terms return True."""
        assert is_valid_exclusion_term("dairy", temp_db)
        assert is_valid_exclusion_term("meat", temp_db)

    def test_valid_dish_type(self, temp_db):
        """Valid dish types return True."""
        assert is_valid_exclusion_term("casserole", temp_db)
        assert is_valid_exclusion_term("soup", temp_db)

    def test_invalid_term(self, temp_db):
        """Invalid terms return False."""
        assert not is_valid_exclusion_term("xyz_random_nonsense", temp_db)
        assert not is_valid_exclusion_term("more", temp_db)  # "no more than 30 minutes"

    def test_case_insensitive(self, temp_db):
        """Validation is case-insensitive."""
        assert is_valid_exclusion_term("CHICKEN", temp_db)
        assert is_valid_exclusion_term("Dairy", temp_db)


class TestCacheClear:
    """Test cache clearing."""

    def test_clear_cache(self, temp_db):
        """Cache can be cleared."""
        vocab1 = load_exclusion_vocabulary(temp_db)
        clear_vocabulary_cache()
        vocab2 = load_exclusion_vocabulary(temp_db)

        # Should be different objects after cache clear
        assert vocab1 is not vocab2


class TestVocabularySize:
    """Test vocabulary size reporting."""

    def test_get_size(self, temp_db):
        """Can get vocabulary size."""
        size = get_vocabulary_size(temp_db)
        assert size > 0

    def test_empty_db_has_base_terms(self, empty_db):
        """Empty DB still has base category/dish terms."""
        size = get_vocabulary_size(empty_db)
        # Should have at least category keywords + dish types
        assert size >= 20


class TestMissingDatabase:
    """Test behavior with missing database."""

    def test_missing_db_returns_base_terms(self):
        """Missing database still returns category terms."""
        clear_vocabulary_cache()
        vocab = load_exclusion_vocabulary(Path("/nonexistent/path/db.db"))

        # Should still have category keywords
        assert "dairy" in vocab
        assert "meat" in vocab
        assert "casserole" in vocab
