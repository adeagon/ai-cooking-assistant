"""Tests for multi-user data isolation.

These tests verify that user data is properly segregated and that
users cannot access each other's data.

Phase 3: Uses user-bound store instances (username at __init__).
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
from src.memory.store_factory import StoreFactory, UserStores


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
        # Create separate store instances per user
        store_alex = ProfileStore(temp_db, username="alex")
        store_caitlyn = ProfileStore(temp_db, username="caitlyn")

        # Set Alex's preferences
        alex_profile = PreferenceProfile(
            spice_level="hot",
            diet="none",
            preferred_cuisines=["mexican", "indian"],
        )
        store_alex.save(alex_profile)

        # Set Caitlyn's preferences
        caitlyn_profile = PreferenceProfile(
            spice_level="mild",
            diet="vegetarian",
            preferred_cuisines=["italian", "greek"],
        )
        store_caitlyn.save(caitlyn_profile)

        # Verify isolation
        loaded_alex = store_alex.load()
        loaded_caitlyn = store_caitlyn.load()

        assert loaded_alex.spice_level == "hot"
        assert loaded_caitlyn.spice_level == "mild"
        assert loaded_alex.diet == "none"
        assert loaded_caitlyn.diet == "vegetarian"
        assert "mexican" in loaded_alex.preferred_cuisines
        assert "italian" in loaded_caitlyn.preferred_cuisines

    def test_guest_gets_defaults(self, temp_db):
        """Guest user gets default preferences."""
        store = ProfileStore(temp_db, username="guest")

        # Load without any saved profile
        profile = store.load()

        assert profile.spice_level == "medium"
        assert profile.diet == "none"
        assert profile.avoid_ingredients == []
        assert profile.preferred_cuisines == []

    def test_default_username_is_guest(self, temp_db):
        """Default username should be 'guest'."""
        store = ProfileStore(temp_db)

        assert store.user == "guest"

    def test_update_isolated_by_user(self, temp_db):
        """Updates should only affect the specified user."""
        store_alex = ProfileStore(temp_db, username="alex")
        store_caitlyn = ProfileStore(temp_db, username="caitlyn")

        # Set initial preferences for both
        store_alex.save(PreferenceProfile(spice_level="medium"))
        store_caitlyn.save(PreferenceProfile(spice_level="medium"))

        # Update only Alex's
        store_alex.update(spice_level="hot")

        # Verify only Alex changed
        assert store_alex.load().spice_level == "hot"
        assert store_caitlyn.load().spice_level == "medium"


class TestFeedbackIsolation:
    """Test that feedback is isolated per user."""

    def test_likes_isolated_by_user(self, temp_db):
        """Alex's likes don't show in Caitlyn's liked list."""
        store_alex = FeedbackStore(temp_db, username="alex")
        store_caitlyn = FeedbackStore(temp_db, username="caitlyn")

        # Alex likes recipe_1
        store_alex.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ))

        # Caitlyn likes recipe_2
        store_caitlyn.add_feedback(RecipeFeedback(
            recipe_id="recipe_2",
            feedback_type="like",
            session_id="session2"
        ))

        # Verify isolation
        alex_likes = store_alex.get_liked_recipe_ids()
        caitlyn_likes = store_caitlyn.get_liked_recipe_ids()

        assert "recipe_1" in alex_likes
        assert "recipe_1" not in caitlyn_likes
        assert "recipe_2" in caitlyn_likes
        assert "recipe_2" not in alex_likes

    def test_dislikes_isolated_by_user(self, temp_db):
        """Dislikes are user-specific."""
        store_alex = FeedbackStore(temp_db, username="alex")
        store_caitlyn = FeedbackStore(temp_db, username="caitlyn")

        # Alex dislikes recipe_3
        store_alex.add_feedback(RecipeFeedback(
            recipe_id="recipe_3",
            feedback_type="dislike",
            session_id="session1"
        ))

        # Verify Caitlyn doesn't see it
        alex_dislikes = store_alex.get_disliked_recipe_ids()
        caitlyn_dislikes = store_caitlyn.get_disliked_recipe_ids()

        assert "recipe_3" in alex_dislikes
        assert "recipe_3" not in caitlyn_dislikes

    def test_default_username_is_guest(self, temp_db):
        """Default username should be 'guest'."""
        store = FeedbackStore(temp_db)

        # Add feedback with default user
        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ))

        # Verify it's under guest
        assert store.user == "guest"
        guest_store = FeedbackStore(temp_db, username="guest")
        guest_likes = guest_store.get_liked_recipe_ids()
        assert "recipe_1" in guest_likes


class TestRecipeBoxIsolation:
    """Test that recipe box is isolated per user."""

    def test_saved_recipes_isolated(self, temp_db):
        """Alex's saved recipes don't appear in Caitlyn's box."""
        store_alex = RecipeBoxStore(temp_db, username="alex")
        store_caitlyn = RecipeBoxStore(temp_db, username="caitlyn")

        # Alex saves recipe_1
        store_alex.save_recipe("recipe_1", "Chicken Tacos")

        # Caitlyn saves recipe_2
        store_caitlyn.save_recipe("recipe_2", "Pasta Carbonara")

        # Verify isolation
        alex_box = store_alex.get_saved_recipes()
        caitlyn_box = store_caitlyn.get_saved_recipes()

        alex_ids = {r.recipe_id for r in alex_box}
        caitlyn_ids = {r.recipe_id for r in caitlyn_box}

        assert "recipe_1" in alex_ids
        assert "recipe_1" not in caitlyn_ids
        assert "recipe_2" in caitlyn_ids
        assert "recipe_2" not in alex_ids

    def test_same_recipe_different_users(self, temp_db):
        """Same recipe can be saved by multiple users without conflict."""
        store_alex = RecipeBoxStore(temp_db, username="alex")
        store_caitlyn = RecipeBoxStore(temp_db, username="caitlyn")

        # Both users save the same recipe
        store_alex.save_recipe("recipe_1", "Chicken Tacos")
        store_caitlyn.save_recipe("recipe_1", "Chicken Tacos")

        # Both should have it
        alex_box = store_alex.get_saved_recipes()
        caitlyn_box = store_caitlyn.get_saved_recipes()

        assert any(r.recipe_id == "recipe_1" for r in alex_box)
        assert any(r.recipe_id == "recipe_1" for r in caitlyn_box)

    def test_is_saved_isolated_by_user(self, temp_db):
        """is_saved checks are user-specific."""
        store_alex = RecipeBoxStore(temp_db, username="alex")
        store_caitlyn = RecipeBoxStore(temp_db, username="caitlyn")

        # Alex saves recipe_1
        store_alex.save_recipe("recipe_1", "Chicken Tacos")

        # Check isolation
        assert store_alex.is_saved("recipe_1") is True
        assert store_caitlyn.is_saved("recipe_1") is False


class TestCrossUserProtection:
    """Test that users can't affect each other's data."""

    def test_cannot_unsave_another_users_recipe(self, temp_db):
        """Alex can't unsave a recipe from Caitlyn's box."""
        store_alex = RecipeBoxStore(temp_db, username="alex")
        store_caitlyn = RecipeBoxStore(temp_db, username="caitlyn")

        # Caitlyn saves recipe_1
        store_caitlyn.save_recipe("recipe_1", "Chicken Tacos")

        # Alex tries to unsave it (should fail/return False)
        result = store_alex.remove_recipe("recipe_1")
        assert result is False

        # Caitlyn's saved recipe should still exist
        caitlyn_box = store_caitlyn.get_saved_recipes()
        assert any(r.recipe_id == "recipe_1" for r in caitlyn_box)

    def test_likes_dont_cross_users(self, temp_db):
        """Alex's like doesn't appear in Caitlyn's liked set."""
        store_alex = FeedbackStore(temp_db, username="alex")
        store_caitlyn = FeedbackStore(temp_db, username="caitlyn")

        # Alex likes many recipes
        for i in range(1, 4):
            store_alex.add_feedback(RecipeFeedback(
                recipe_id=f"recipe_{i}",
                feedback_type="like",
                session_id="session1"
            ))

        # Caitlyn's likes should be empty
        caitlyn_likes = store_caitlyn.get_liked_recipe_ids()
        assert len(caitlyn_likes) == 0


class TestHistoryIsolation:
    """Test that cooking history is isolated per user."""

    def test_history_isolated_by_user(self, temp_db):
        """Alex's cooking history doesn't appear in Caitlyn's."""
        store_alex = HistoryStore(temp_db, username="alex")
        store_caitlyn = HistoryStore(temp_db, username="caitlyn")

        # Alex cooked recipe_1
        store_alex.add_cooked("recipe_1")

        # Caitlyn cooked recipe_2
        store_caitlyn.add_cooked("recipe_2")

        # Verify isolation
        alex_history = store_alex.get_cooking_history()
        caitlyn_history = store_caitlyn.get_cooking_history()

        alex_ids = {h.recipe_id for h in alex_history}
        caitlyn_ids = {h.recipe_id for h in caitlyn_history}

        assert "recipe_1" in alex_ids
        assert "recipe_1" not in caitlyn_ids
        assert "recipe_2" in caitlyn_ids
        assert "recipe_2" not in alex_ids

    def test_recently_cooked_isolated(self, temp_db):
        """Recently cooked exclusion is user-specific."""
        store_alex = HistoryStore(temp_db, username="alex")
        store_caitlyn = HistoryStore(temp_db, username="caitlyn")

        # Alex cooked recipe_1
        store_alex.add_cooked("recipe_1")

        # Verify isolation in recently cooked
        alex_recent = store_alex.get_recently_cooked_ids(days=7)
        caitlyn_recent = store_caitlyn.get_recently_cooked_ids(days=7)

        assert "recipe_1" in alex_recent
        assert "recipe_1" not in caitlyn_recent

    def test_cooking_count_isolated(self, temp_db):
        """Cooking count is user-specific."""
        store_alex = HistoryStore(temp_db, username="alex")
        store_caitlyn = HistoryStore(temp_db, username="caitlyn")

        # Alex cooked recipe_1 twice
        store_alex.add_cooked("recipe_1")
        store_alex.add_cooked("recipe_1")

        # Caitlyn cooked recipe_1 once
        store_caitlyn.add_cooked("recipe_1")

        # Verify counts are isolated
        assert store_alex.get_cooking_count("recipe_1") == 2
        assert store_caitlyn.get_cooking_count("recipe_1") == 1


class TestSessionIsolation:
    """Test that sessions are isolated per user."""

    def test_sessions_per_user(self, temp_db):
        """Each user gets their own session."""
        store_alex = SessionStore(temp_db, username="alex")
        store_caitlyn = SessionStore(temp_db, username="caitlyn")

        # Create sessions for different users
        alex_session_id = store_alex.create()
        caitlyn_session_id = store_caitlyn.create()

        # Sessions should be different
        assert alex_session_id != caitlyn_session_id

    def test_get_or_create_current_per_user(self, temp_db):
        """get_or_create_current returns user-specific session."""
        store_alex = SessionStore(temp_db, username="alex")

        # Get or create for Alex
        alex_id1, _ = store_alex.get_or_create_current()
        alex_id2, _ = store_alex.get_or_create_current()

        # Should return same session for Alex
        assert alex_id1 == alex_id2

        # Get or create for Caitlyn
        store_caitlyn = SessionStore(temp_db, username="caitlyn")
        caitlyn_id, _ = store_caitlyn.get_or_create_current()

        # Should be different from Alex's
        assert caitlyn_id != alex_id1


class TestDefaultGuestFallback:
    """Test that default username is 'guest' across all stores."""

    def test_profile_default_is_guest(self, temp_db):
        """ProfileStore: default username is guest."""
        store = ProfileStore(temp_db)

        # Verify user property
        assert store.user == "guest"

        # Save with default store
        store.save(PreferenceProfile(spice_level="hot"))

        # Load with explicit guest should see it
        guest_store = ProfileStore(temp_db, username="guest")
        profile = guest_store.load()
        assert profile.spice_level == "hot"

    def test_feedback_default_is_guest(self, temp_db):
        """FeedbackStore: default username is guest."""
        store = FeedbackStore(temp_db)
        assert store.user == "guest"

        store.add_feedback(RecipeFeedback(
            recipe_id="recipe_1",
            feedback_type="like",
            session_id="session1"
        ))

        # Should appear under guest
        guest_store = FeedbackStore(temp_db, username="guest")
        likes = guest_store.get_liked_recipe_ids()
        assert "recipe_1" in likes

    def test_recipe_box_default_is_guest(self, temp_db):
        """RecipeBoxStore: default username is guest."""
        store = RecipeBoxStore(temp_db)
        assert store.user == "guest"

        store.save_recipe("recipe_1", "Chicken Tacos")

        # Should appear under guest
        guest_store = RecipeBoxStore(temp_db, username="guest")
        box = guest_store.get_saved_recipes()
        assert any(r.recipe_id == "recipe_1" for r in box)

    def test_history_default_is_guest(self, temp_db):
        """HistoryStore: default username is guest."""
        store = HistoryStore(temp_db)
        assert store.user == "guest"

        store.add_cooked("recipe_1")

        # Should appear under guest
        guest_store = HistoryStore(temp_db, username="guest")
        history = guest_store.get_cooking_history()
        assert any(h.recipe_id == "recipe_1" for h in history)

    def test_session_default_is_guest(self, temp_db):
        """SessionStore: default username is guest."""
        store = SessionStore(temp_db)
        assert store.user == "guest"

        session_id = store.create()

        # Should be retrievable for guest
        guest_store = SessionStore(temp_db, username="guest")
        session_id2, _ = guest_store.get_or_create_current()
        assert session_id == session_id2


class TestUserPropertyReadOnly:
    """Test that user property is read-only."""

    def test_profile_user_readonly(self, temp_db):
        """Cannot assign to ProfileStore.user after init."""
        store = ProfileStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_feedback_user_readonly(self, temp_db):
        """Cannot assign to FeedbackStore.user after init."""
        store = FeedbackStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_history_user_readonly(self, temp_db):
        """Cannot assign to HistoryStore.user after init."""
        store = HistoryStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_recipe_box_user_readonly(self, temp_db):
        """Cannot assign to RecipeBoxStore.user after init."""
        store = RecipeBoxStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_session_user_readonly(self, temp_db):
        """Cannot assign to SessionStore.user after init."""
        store = SessionStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_meal_plan_user_readonly(self, temp_db):
        """Cannot assign to MealPlanStore.user after init."""
        store = MealPlanStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"


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
        store = ProfileStore(db_path, username="guest")

        # Old data should be under 'guest'
        profile = store.load()
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
        store = RecipeBoxStore(db_path, username="guest")

        # Old data should be under 'guest'
        box = store.get_saved_recipes()
        assert any(r.recipe_id == "old_recipe" for r in box)
