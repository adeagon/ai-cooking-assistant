"""Tests for multi-user data isolation.

These tests verify that user data is properly segregated and that
users cannot access each other's data.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from src.domain.models import RecipeFeedback, PreferenceProfile
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from src.memory.profile_store import ProfileStore
from src.memory.recipe_box_store import RecipeBoxStore
from src.memory.session_store import SessionStore
from src.memory.meal_plan_store import MealPlanStore


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing with required tables."""
    db_path = tmp_path / "test.db"

    # Create recipes table (required for foreign key constraints)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT,
            tags TEXT
        )
    """)
    # Add test recipes
    cursor.execute("INSERT INTO recipes VALUES ('recipe_1', 'Chicken Tacos', '[\"mexican\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('recipe_2', 'Pasta Carbonara', '[\"italian\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('recipe_3', 'Vegetable Stir Fry', '[\"asian\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('recipe_4', 'Greek Salad', '[\"greek\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('recipe_5', 'Beef Stew', '[\"american\"]')")
    conn.commit()
    conn.close()

    return db_path


class TestProfileIsolation:
    """Test that preferences are isolated per user."""

    def test_different_users_different_prefs(self, temp_db):
        """Alex's preferences should not affect Caitlyn's."""
        store = ProfileStore(temp_db)

        # Set Alex's preferences
        alex_profile = PreferenceProfile(
            spice_level="hot",
            diet="none",
            preferred_cuisines=["mexican", "indian"],
        )
        store.save(alex_profile, user_id="alex")

        # Set Caitlyn's preferences
        caitlyn_profile = PreferenceProfile(
            spice_level="mild",
            diet="vegetarian",
            preferred_cuisines=["italian", "greek"],
        )
        store.save(caitlyn_profile, user_id="caitlyn")

        # Verify isolation
        loaded_alex = store.load(user_id="alex")
        loaded_caitlyn = store.load(user_id="caitlyn")

        assert loaded_alex.spice_level == "hot"
        assert loaded_caitlyn.spice_level == "mild"
        assert loaded_alex.diet == "none"
        assert loaded_caitlyn.diet == "vegetarian"
        assert "mexican" in loaded_alex.preferred_cuisines
        assert "italian" in loaded_caitlyn.preferred_cuisines

    def test_guest_gets_defaults(self, temp_db):
        """Guest user gets default preferences."""
        store = ProfileStore(temp_db)

        # Load without any saved profile
        profile = store.load(user_id="guest")

        assert profile.spice_level == "medium"
        assert profile.diet == "none"
        assert profile.avoid_ingredients == []
        assert profile.preferred_cuisines == []

    def test_none_user_id_uses_guest(self, temp_db):
        """Passing None for user_id should use 'guest'."""
        store = ProfileStore(temp_db)

        # Save with explicit guest
        guest_profile = PreferenceProfile(spice_level="hot")
        store.save(guest_profile, user_id="guest")

        # Load with None (should use guest)
        loaded = store.load(user_id=None)
        assert loaded.spice_level == "hot"

    def test_update_isolated_by_user(self, temp_db):
        """Updates should only affect the specified user."""
        store = ProfileStore(temp_db)

        # Set initial preferences for both
        store.save(PreferenceProfile(spice_level="medium"), user_id="alex")
        store.save(PreferenceProfile(spice_level="medium"), user_id="caitlyn")

        # Update only Alex's
        store.update(user_id="alex", spice_level="hot")

        # Verify only Alex changed
        assert store.load(user_id="alex").spice_level == "hot"
        assert store.load(user_id="caitlyn").spice_level == "medium"


class TestFeedbackIsolation:
    """Test that feedback is isolated per user."""

    def test_likes_isolated_by_user(self, temp_db):
        """Alex's likes don't show in Caitlyn's liked list."""
        store = FeedbackStore(temp_db)

        # Alex likes recipe_1
        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ), user_id="alex")

        # Caitlyn likes recipe_2
        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_2",
            feedback_type="like",
            session_id="session2"
        ), user_id="caitlyn")

        # Verify isolation
        alex_likes = store.get_liked_recipe_ids(user_id="alex")
        caitlyn_likes = store.get_liked_recipe_ids(user_id="caitlyn")

        assert "recipe_1" in alex_likes
        assert "recipe_1" not in caitlyn_likes
        assert "recipe_2" in caitlyn_likes
        assert "recipe_2" not in alex_likes

    def test_dislikes_isolated_by_user(self, temp_db):
        """Dislikes are user-specific."""
        store = FeedbackStore(temp_db)

        # Alex dislikes recipe_3
        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_3",
            feedback_type="dislike",
            session_id="session1"
        ), user_id="alex")

        # Verify Caitlyn doesn't see it
        alex_dislikes = store.get_disliked_recipe_ids(user_id="alex")
        caitlyn_dislikes = store.get_disliked_recipe_ids(user_id="caitlyn")

        assert "recipe_3" in alex_dislikes
        assert "recipe_3" not in caitlyn_dislikes

    def test_none_user_id_uses_guest(self, temp_db):
        """Passing None for user_id should use 'guest'."""
        store = FeedbackStore(temp_db)

        # Add feedback with None (should use guest)
        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ), user_id=None)

        # Verify it's under guest
        guest_likes = store.get_liked_recipe_ids(user_id="guest")
        assert "recipe_1" in guest_likes


class TestRecipeBoxIsolation:
    """Test that recipe box is isolated per user."""

    def test_saved_recipes_isolated(self, temp_db):
        """Alex's saved recipes don't appear in Caitlyn's box."""
        store = RecipeBoxStore(temp_db)

        # Alex saves recipe_1
        store.save_recipe("recipe_1", "Chicken Tacos", user_id="alex")

        # Caitlyn saves recipe_2
        store.save_recipe("recipe_2", "Pasta Carbonara", user_id="caitlyn")

        # Verify isolation
        alex_box = store.get_saved_recipes(user_id="alex")
        caitlyn_box = store.get_saved_recipes(user_id="caitlyn")

        alex_ids = {r.recipe_id for r in alex_box}
        caitlyn_ids = {r.recipe_id for r in caitlyn_box}

        assert "recipe_1" in alex_ids
        assert "recipe_1" not in caitlyn_ids
        assert "recipe_2" in caitlyn_ids
        assert "recipe_2" not in alex_ids

    def test_same_recipe_different_users(self, temp_db):
        """Same recipe can be saved by multiple users without conflict."""
        store = RecipeBoxStore(temp_db)

        # Both users save the same recipe
        store.save_recipe("recipe_1", "Chicken Tacos", user_id="alex")
        store.save_recipe("recipe_1", "Chicken Tacos", user_id="caitlyn")

        # Both should have it
        alex_box = store.get_saved_recipes(user_id="alex")
        caitlyn_box = store.get_saved_recipes(user_id="caitlyn")

        assert any(r.recipe_id == "recipe_1" for r in alex_box)
        assert any(r.recipe_id == "recipe_1" for r in caitlyn_box)

    def test_is_saved_isolated_by_user(self, temp_db):
        """is_saved checks are user-specific."""
        store = RecipeBoxStore(temp_db)

        # Alex saves recipe_1
        store.save_recipe("recipe_1", "Chicken Tacos", user_id="alex")

        # Check isolation
        assert store.is_saved("recipe_1", user_id="alex") is True
        assert store.is_saved("recipe_1", user_id="caitlyn") is False


class TestCrossUserProtection:
    """Test that users can't affect each other's data."""

    def test_cannot_unsave_another_users_recipe(self, temp_db):
        """Alex can't unsave a recipe from Caitlyn's box."""
        store = RecipeBoxStore(temp_db)

        # Caitlyn saves recipe_1
        store.save_recipe("recipe_1", "Chicken Tacos", user_id="caitlyn")

        # Alex tries to unsave it (should fail/return False)
        result = store.remove_recipe("recipe_1", user_id="alex")
        assert result is False

        # Caitlyn's saved recipe should still exist
        caitlyn_box = store.get_saved_recipes(user_id="caitlyn")
        assert any(r.recipe_id == "recipe_1" for r in caitlyn_box)

    def test_likes_dont_cross_users(self, temp_db):
        """Alex's like doesn't appear in Caitlyn's liked set."""
        store = FeedbackStore(temp_db)

        # Alex likes many recipes
        for i in range(1, 4):
            store.add_feedback(RecipeFeedback(
                recipe_id=f"recipe_{i}",
                feedback_type="like",
                session_id="session1"
            ), user_id="alex")

        # Caitlyn's likes should be empty
        caitlyn_likes = store.get_liked_recipe_ids(user_id="caitlyn")
        assert len(caitlyn_likes) == 0


class TestHistoryIsolation:
    """Test that cooking history is isolated per user."""

    def test_history_isolated_by_user(self, temp_db):
        """Alex's cooking history doesn't appear in Caitlyn's."""
        store = HistoryStore(temp_db)

        # Alex cooked recipe_1
        store.add_cooked("recipe_1", user_id="alex")

        # Caitlyn cooked recipe_2
        store.add_cooked("recipe_2", user_id="caitlyn")

        # Verify isolation
        alex_history = store.get_cooking_history(user_id="alex")
        caitlyn_history = store.get_cooking_history(user_id="caitlyn")

        alex_ids = {h.recipe_id for h in alex_history}
        caitlyn_ids = {h.recipe_id for h in caitlyn_history}

        assert "recipe_1" in alex_ids
        assert "recipe_1" not in caitlyn_ids
        assert "recipe_2" in caitlyn_ids
        assert "recipe_2" not in alex_ids

    def test_recently_cooked_isolated(self, temp_db):
        """Recently cooked exclusion is user-specific."""
        store = HistoryStore(temp_db)

        # Alex cooked recipe_1
        store.add_cooked("recipe_1", user_id="alex")

        # Verify isolation in recently cooked
        alex_recent = store.get_recently_cooked_ids(days=7, user_id="alex")
        caitlyn_recent = store.get_recently_cooked_ids(days=7, user_id="caitlyn")

        assert "recipe_1" in alex_recent
        assert "recipe_1" not in caitlyn_recent

    def test_cooking_count_isolated(self, temp_db):
        """Cooking count is user-specific."""
        store = HistoryStore(temp_db)

        # Alex cooked recipe_1 twice
        store.add_cooked("recipe_1", user_id="alex")
        store.add_cooked("recipe_1", user_id="alex")

        # Caitlyn cooked recipe_1 once
        store.add_cooked("recipe_1", user_id="caitlyn")

        # Verify counts are isolated
        assert store.get_cooking_count("recipe_1", user_id="alex") == 2
        assert store.get_cooking_count("recipe_1", user_id="caitlyn") == 1


class TestSessionIsolation:
    """Test that sessions are isolated per user."""

    def test_sessions_per_user(self, temp_db):
        """Each user gets their own session."""
        store = SessionStore(temp_db)

        # Create sessions for different users
        alex_session_id = store.create(user_id="alex")
        caitlyn_session_id = store.create(user_id="caitlyn")

        # Sessions should be different
        assert alex_session_id != caitlyn_session_id

    def test_get_or_create_current_per_user(self, temp_db):
        """get_or_create_current returns user-specific session."""
        store = SessionStore(temp_db)

        # Get or create for Alex
        alex_id1, _ = store.get_or_create_current(user_id="alex")
        alex_id2, _ = store.get_or_create_current(user_id="alex")

        # Should return same session for Alex
        assert alex_id1 == alex_id2

        # Get or create for Caitlyn
        caitlyn_id, _ = store.get_or_create_current(user_id="caitlyn")

        # Should be different from Alex's
        assert caitlyn_id != alex_id1


class TestDefaultGuestFallback:
    """Test that None user_id defaults to guest across all stores."""

    def test_profile_none_uses_guest(self, temp_db):
        """ProfileStore: None user_id uses guest."""
        store = ProfileStore(temp_db)

        # Save with None
        store.save(PreferenceProfile(spice_level="hot"), user_id=None)

        # Load with "guest" explicitly
        profile = store.load(user_id="guest")
        assert profile.spice_level == "hot"

    def test_feedback_none_uses_guest(self, temp_db):
        """FeedbackStore: None user_id uses guest."""
        store = FeedbackStore(temp_db)

        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ), user_id=None)

        # Should appear under guest
        likes = store.get_liked_recipe_ids(user_id="guest")
        assert "recipe_1" in likes

    def test_recipe_box_none_uses_guest(self, temp_db):
        """RecipeBoxStore: None user_id uses guest."""
        store = RecipeBoxStore(temp_db)

        store.save_recipe("recipe_1", "Chicken Tacos", user_id=None)

        # Should appear under guest
        box = store.get_saved_recipes(user_id="guest")
        assert any(r.recipe_id == "recipe_1" for r in box)

    def test_history_none_uses_guest(self, temp_db):
        """HistoryStore: None user_id uses guest."""
        store = HistoryStore(temp_db)

        store.add_cooked("recipe_1", user_id=None)

        # Should appear under guest
        history = store.get_cooking_history(user_id="guest")
        assert any(h.recipe_id == "recipe_1" for h in history)

    def test_session_none_uses_guest(self, temp_db):
        """SessionStore: None user_id uses guest."""
        store = SessionStore(temp_db)

        session_id = store.create(user_id=None)

        # Should be retrievable for guest
        session_id2, _ = store.get_or_create_current(user_id="guest")
        assert session_id == session_id2


class TestSchemaMigration:
    """Test that schema migration works correctly."""

    def test_profile_migration_preserves_guest_data(self, tmp_path):
        """Old profile data should migrate to 'guest' user."""
        db_path = tmp_path / "test.db"

        # Create old schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE preferences (
                id INTEGER PRIMARY KEY DEFAULT 1,
                spice_level TEXT DEFAULT 'medium',
                diet TEXT DEFAULT 'none',
                avoid_ingredients TEXT,
                preferred_cuisines TEXT,
                time_limit_default INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO preferences (id, spice_level, diet)
            VALUES (1, 'hot', 'vegetarian')
        """)
        conn.commit()
        conn.close()

        # Initialize store (triggers migration)
        store = ProfileStore(db_path)

        # Old data should be under 'guest'
        profile = store.load(user_id="guest")
        assert profile.spice_level == "hot"
        assert profile.diet == "vegetarian"

    def test_recipe_box_migration_preserves_guest_data(self, tmp_path):
        """Old recipe box data should migrate to 'guest' user."""
        db_path = tmp_path / "test.db"

        # Create old schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE saved_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO saved_recipes (recipe_id, title)
            VALUES ('old_recipe', 'Old Saved Recipe')
        """)
        conn.commit()
        conn.close()

        # Initialize store (triggers migration)
        store = RecipeBoxStore(db_path)

        # Old data should be under 'guest'
        box = store.get_saved_recipes(user_id="guest")
        assert any(r.recipe_id == "old_recipe" for r in box)
