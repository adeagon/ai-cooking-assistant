"""Tests for RecipeBoxStore."""

import sqlite3
from datetime import datetime

import pytest

from src.domain.models import SavedRecipe
from src.memory.recipe_box_store import RecipeBoxStore


class TestRecipeBoxStore:
    """Test RecipeBoxStore functionality."""

    def test_initialization_creates_table(self, temp_db, test_user_id):
        """Test that initialization creates the saved recipes table."""
        store = RecipeBoxStore(temp_db, test_user_id)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='saved_recipes'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_save_recipe(self, temp_db, test_user_id):
        """Test saving a recipe to the box."""
        store = RecipeBoxStore(temp_db, test_user_id)

        saved_id = store.save_recipe("123", "Test Recipe")
        assert saved_id > 0

    def test_save_recipe_with_notes(self, temp_db, test_user_id):
        """Test saving a recipe with notes."""
        store = RecipeBoxStore(temp_db, test_user_id)

        saved_id = store.save_recipe("123", "Test Recipe", notes="Want to try this!")
        assert saved_id > 0

        # Verify notes were stored
        saved_recipes = store.get_saved_recipes(limit=1)
        assert len(saved_recipes) == 1
        assert saved_recipes[0].notes == "Want to try this!"

    def test_save_duplicate_recipe_raises_error(self, temp_db, test_user_id):
        """Test that saving a duplicate recipe raises IntegrityError."""
        store = RecipeBoxStore(temp_db, test_user_id)

        # Save first time
        store.save_recipe("123", "Test Recipe")

        # Try to save again - should raise error
        with pytest.raises(sqlite3.IntegrityError):
            store.save_recipe("123", "Test Recipe")

    def test_get_saved_recipes(self, temp_db, test_user_id):
        """Test getting saved recipes."""
        store = RecipeBoxStore(temp_db, test_user_id)

        # Save multiple recipes
        store.save_recipe("123", "Test Recipe")
        store.save_recipe("456", "Another Recipe")

        saved_recipes = store.get_saved_recipes()
        assert len(saved_recipes) == 2
        # Should be ordered by saved_at DESC (most recent first)
        assert saved_recipes[0].recipe_id == "456"
        assert saved_recipes[1].recipe_id == "123"

    def test_get_saved_recipes_with_limit(self, temp_db, test_user_id):
        """Test getting saved recipes with limit."""
        store = RecipeBoxStore(temp_db, test_user_id)

        # Save three recipes
        store.save_recipe("123", "Test Recipe")
        store.save_recipe("456", "Another Recipe")
        store.save_recipe("789", "Third Recipe")

        saved_recipes = store.get_saved_recipes(limit=2)
        assert len(saved_recipes) == 2

    def test_get_saved_recipes_empty_box(self, temp_db, test_user_id):
        """Test getting saved recipes from empty box."""
        store = RecipeBoxStore(temp_db, test_user_id)

        saved_recipes = store.get_saved_recipes()
        assert len(saved_recipes) == 0

    def test_remove_recipe(self, temp_db, test_user_id):
        """Test removing a recipe from the box."""
        store = RecipeBoxStore(temp_db, test_user_id)

        # Save and then remove
        store.save_recipe("123", "Test Recipe")
        result = store.remove_recipe("123")
        assert result is True

        # Verify it's gone
        saved_recipes = store.get_saved_recipes()
        assert len(saved_recipes) == 0

    def test_remove_nonexistent_recipe(self, temp_db, test_user_id):
        """Test removing a recipe that doesn't exist."""
        store = RecipeBoxStore(temp_db, test_user_id)

        result = store.remove_recipe("999")
        assert result is False

    def test_is_saved_true(self, temp_db, test_user_id):
        """Test checking if a recipe is saved."""
        store = RecipeBoxStore(temp_db, test_user_id)

        store.save_recipe("123", "Test Recipe")
        assert store.is_saved("123") is True

    def test_is_saved_false(self, temp_db, test_user_id):
        """Test checking if a recipe is not saved."""
        store = RecipeBoxStore(temp_db, test_user_id)

        assert store.is_saved("999") is False

    def test_saved_recipe_model(self, temp_db, test_user_id):
        """Test SavedRecipe model fields."""
        store = RecipeBoxStore(temp_db, test_user_id)

        store.save_recipe("123", "Test Recipe", notes="Looks good!")
        saved_recipes = store.get_saved_recipes()

        assert len(saved_recipes) == 1
        saved = saved_recipes[0]

        assert isinstance(saved, SavedRecipe)
        assert saved.recipe_id == "123"
        assert saved.title == "Test Recipe"
        assert saved.notes == "Looks good!"
        assert isinstance(saved.saved_at, datetime)
        assert saved.id is not None
