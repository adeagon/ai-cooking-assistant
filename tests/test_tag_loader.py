"""Tests for tag_loader utility module."""

import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.tag_loader import (
    CUISINE_TAGS,
    GOAL_TAGS,
    GOAL_FALLBACKS,
    load_cuisines_from_db,
    load_goals_from_db,
    resolve_goal
)


class TestCuisineTags:
    """Tests for cuisine tag constants."""

    def test_cuisine_tags_is_set(self):
        """Test that CUISINE_TAGS is a set."""
        assert isinstance(CUISINE_TAGS, set)

    def test_common_cuisines_present(self):
        """Test that common cuisines are included."""
        common_cuisines = [
            "asian", "american", "italian", "mexican", "chinese",
            "japanese", "indian", "thai", "french", "greek"
        ]
        for cuisine in common_cuisines:
            assert cuisine in CUISINE_TAGS, f"Missing cuisine: {cuisine}"

    def test_regional_cuisines_present(self):
        """Test that regional cuisines are included."""
        regional = [
            "korean", "vietnamese", "mediterranean",
            "southern-united-states", "middle-eastern"
        ]
        for cuisine in regional:
            assert cuisine in CUISINE_TAGS, f"Missing regional cuisine: {cuisine}"

    def test_cuisine_tags_are_lowercase(self):
        """Test that all cuisine tags are lowercase."""
        for tag in CUISINE_TAGS:
            assert tag == tag.lower(), f"Tag not lowercase: {tag}"


class TestGoalTags:
    """Tests for goal tag constants."""

    def test_goal_tags_is_set(self):
        """Test that GOAL_TAGS is a set."""
        assert isinstance(GOAL_TAGS, set)

    def test_taste_tags_present(self):
        """Test that taste tags are included."""
        taste_tags = ["sweet", "savory", "spicy", "mild", "rich", "light"]
        for tag in taste_tags:
            assert tag in GOAL_TAGS, f"Missing taste tag: {tag}"

    def test_goal_tags_present(self):
        """Test that goal tags are included."""
        goal_tags = [
            "healthy", "comfort-food", "low-calorie", "high-protein",
            "easy", "inexpensive", "kid-friendly"
        ]
        for tag in goal_tags:
            assert tag in GOAL_TAGS, f"Missing goal tag: {tag}"


class TestGoalFallbacks:
    """Tests for goal fallback mappings."""

    def test_goal_fallbacks_is_dict(self):
        """Test that GOAL_FALLBACKS is a dict."""
        assert isinstance(GOAL_FALLBACKS, dict)

    def test_hearty_maps_to_rich(self):
        """Test that 'hearty' maps to 'rich'."""
        assert GOAL_FALLBACKS["hearty"] == "rich"

    def test_filling_maps_to_rich(self):
        """Test that 'filling' maps to 'rich'."""
        assert GOAL_FALLBACKS["filling"] == "rich"

    def test_cheap_maps_to_inexpensive(self):
        """Test that 'cheap' maps to 'inexpensive'."""
        assert GOAL_FALLBACKS["cheap"] == "inexpensive"

    def test_budget_maps_to_inexpensive(self):
        """Test that 'budget' maps to 'inexpensive'."""
        assert GOAL_FALLBACKS["budget"] == "inexpensive"

    def test_quick_maps_to_easy(self):
        """Test that 'quick' maps to 'easy'."""
        assert GOAL_FALLBACKS["quick"] == "easy"

    def test_protein_maps_to_high_protein(self):
        """Test that 'protein' maps to 'high-protein'."""
        assert GOAL_FALLBACKS["protein"] == "high-protein"

    def test_fallback_targets_are_valid_goal_tags(self):
        """Test that all fallback targets exist in GOAL_TAGS."""
        for target in GOAL_FALLBACKS.values():
            assert target in GOAL_TAGS, f"Fallback target not in GOAL_TAGS: {target}"


class TestLoadCuisinesFromDb:
    """Tests for loading cuisines from database."""

    @pytest.fixture
    def db_with_cuisines(self, tmp_path):
        """Create a test database with cuisine tags."""
        db_path = tmp_path / "recipes.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE recipes (
                recipe_id TEXT PRIMARY KEY,
                tags TEXT
            )
        """)

        # Insert recipes with various tags
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("1", json.dumps(["italian", "pasta", "dinner"]))
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("2", json.dumps(["mexican", "spicy", "tacos"]))
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("3", json.dumps(["chinese", "asian", "stir-fry"]))
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("4", json.dumps(["dessert", "easy"]))  # No cuisine tags
        )

        conn.commit()
        conn.close()

        return str(db_path)

    def test_load_cuisines_returns_set(self, db_with_cuisines):
        """Test that load_cuisines_from_db returns a set."""
        # Clear cache first
        load_cuisines_from_db.cache_clear()

        result = load_cuisines_from_db(db_with_cuisines)
        assert isinstance(result, set)

    def test_load_cuisines_finds_valid_cuisines(self, db_with_cuisines):
        """Test that valid cuisine tags are found."""
        load_cuisines_from_db.cache_clear()

        result = load_cuisines_from_db(db_with_cuisines)

        assert "italian" in result
        assert "mexican" in result
        assert "chinese" in result
        assert "asian" in result

    def test_load_cuisines_ignores_non_cuisine_tags(self, db_with_cuisines):
        """Test that non-cuisine tags are not included."""
        load_cuisines_from_db.cache_clear()

        result = load_cuisines_from_db(db_with_cuisines)

        assert "pasta" not in result
        assert "dinner" not in result
        assert "spicy" not in result
        assert "dessert" not in result

    def test_load_cuisines_handles_invalid_json(self, tmp_path):
        """Test handling of invalid JSON in tags column."""
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE recipes (
                recipe_id TEXT PRIMARY KEY,
                tags TEXT
            )
        """)

        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("1", "not valid json")
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("2", json.dumps(["italian"]))
        )

        conn.commit()
        conn.close()

        load_cuisines_from_db.cache_clear()
        result = load_cuisines_from_db(str(db_path))

        # Should still find valid cuisines
        assert "italian" in result

    def test_load_cuisines_caches_result(self, db_with_cuisines):
        """Test that results are cached."""
        load_cuisines_from_db.cache_clear()

        # First call
        result1 = load_cuisines_from_db(db_with_cuisines)
        # Second call (should use cache)
        result2 = load_cuisines_from_db(db_with_cuisines)

        assert result1 == result2

        # Check cache info
        cache_info = load_cuisines_from_db.cache_info()
        assert cache_info.hits >= 1


class TestLoadGoalsFromDb:
    """Tests for loading goals from database."""

    @pytest.fixture
    def db_with_goals(self, tmp_path):
        """Create a test database with goal tags."""
        db_path = tmp_path / "recipes.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE recipes (
                recipe_id TEXT PRIMARY KEY,
                tags TEXT
            )
        """)

        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("1", json.dumps(["savory", "dinner", "easy"]))
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("2", json.dumps(["sweet", "dessert", "healthy"]))
        )
        cursor.execute(
            "INSERT INTO recipes VALUES (?, ?)",
            ("3", json.dumps(["spicy", "comfort-food"]))
        )

        conn.commit()
        conn.close()

        return str(db_path)

    def test_load_goals_returns_set(self, db_with_goals):
        """Test that load_goals_from_db returns a set."""
        load_goals_from_db.cache_clear()

        result = load_goals_from_db(db_with_goals)
        assert isinstance(result, set)

    def test_load_goals_finds_valid_goals(self, db_with_goals):
        """Test that valid goal tags are found."""
        load_goals_from_db.cache_clear()

        result = load_goals_from_db(db_with_goals)

        assert "savory" in result
        assert "sweet" in result
        assert "spicy" in result
        assert "easy" in result
        assert "healthy" in result
        assert "comfort-food" in result

    def test_load_goals_ignores_non_goal_tags(self, db_with_goals):
        """Test that non-goal tags are not included."""
        load_goals_from_db.cache_clear()

        result = load_goals_from_db(db_with_goals)

        assert "dinner" not in result
        assert "dessert" not in result

    def test_load_goals_caches_result(self, db_with_goals):
        """Test that results are cached."""
        load_goals_from_db.cache_clear()

        # First call
        result1 = load_goals_from_db(db_with_goals)
        # Second call (should use cache)
        result2 = load_goals_from_db(db_with_goals)

        assert result1 == result2

        cache_info = load_goals_from_db.cache_info()
        assert cache_info.hits >= 1


class TestResolveGoal:
    """Tests for resolving user terms to goal tags."""

    def test_resolve_goal_direct_match(self):
        """Test resolving a direct goal match."""
        assert resolve_goal("healthy") == "healthy"
        assert resolve_goal("spicy") == "spicy"
        assert resolve_goal("easy") == "easy"

    def test_resolve_goal_fallback_mapping(self):
        """Test resolving via fallback mapping."""
        assert resolve_goal("hearty") == "rich"
        assert resolve_goal("cheap") == "inexpensive"
        assert resolve_goal("quick") == "easy"
        assert resolve_goal("protein") == "high-protein"

    def test_resolve_goal_case_insensitive(self):
        """Test that resolution is case-insensitive."""
        assert resolve_goal("HEALTHY") == "healthy"
        assert resolve_goal("Spicy") == "spicy"
        assert resolve_goal("HEARTY") == "rich"

    def test_resolve_goal_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert resolve_goal("  healthy  ") == "healthy"
        assert resolve_goal("\tchеap\n") is None  # Note: this has a Cyrillic 'е'
        assert resolve_goal("  cheap  ") == "inexpensive"

    def test_resolve_goal_unknown_term(self):
        """Test that unknown terms return None."""
        assert resolve_goal("unknown") is None
        assert resolve_goal("xyz123") is None
        assert resolve_goal("") is None

    def test_resolve_goal_all_direct_tags(self):
        """Test that all GOAL_TAGS can be resolved directly."""
        for tag in GOAL_TAGS:
            assert resolve_goal(tag) == tag

    def test_resolve_goal_all_fallbacks(self):
        """Test that all GOAL_FALLBACKS resolve correctly."""
        for user_term, expected in GOAL_FALLBACKS.items():
            assert resolve_goal(user_term) == expected


class TestLoadFunctionsUseDefaultPath:
    """Tests for loading functions using default Settings path."""

    def test_load_cuisines_uses_settings_when_no_path(self):
        """Test that load_cuisines_from_db uses Settings when no path given."""
        load_cuisines_from_db.cache_clear()

        with patch('src.app.settings.Settings') as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.sqlite_db_path = "/fake/path/recipes.db"
            mock_settings_class.return_value = mock_settings

            with patch('sqlite3.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = []
                mock_conn = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                load_cuisines_from_db(None)

                mock_connect.assert_called_with("/fake/path/recipes.db")

    def test_load_goals_uses_settings_when_no_path(self):
        """Test that load_goals_from_db uses Settings when no path given."""
        load_goals_from_db.cache_clear()

        with patch('src.app.settings.Settings') as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.sqlite_db_path = "/fake/path/recipes.db"
            mock_settings_class.return_value = mock_settings

            with patch('sqlite3.connect') as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = []
                mock_conn = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                load_goals_from_db(None)

                mock_connect.assert_called_with("/fake/path/recipes.db")
