"""Tests for StoreFactory and BaseUserBoundStore."""

import sqlite3
from pathlib import Path

import pytest

from src.memory.profile_store import ProfileStore
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
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
    conn.commit()
    conn.close()

    return db_path


class TestStoreFactory:
    """Test StoreFactory class."""

    def test_get_stores_creates_all_stores(self, temp_db):
        """Factory creates all store types."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        assert stores.profile is not None
        assert stores.feedback is not None
        assert stores.history is not None
        assert stores.recipe_box is not None
        assert stores.session is not None
        assert stores.meal_plan is not None

    def test_get_stores_returns_userstores_dataclass(self, temp_db):
        """Factory returns UserStores dataclass."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        assert isinstance(stores, UserStores)

    def test_stores_bound_to_correct_user(self, temp_db):
        """All stores are bound to the requested user."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        assert stores.profile.user == "alex"
        assert stores.feedback.user == "alex"
        assert stores.history.user == "alex"
        assert stores.recipe_box.user == "alex"
        assert stores.session.user == "alex"
        assert stores.meal_plan.user == "alex"

    def test_get_stores_caches_instances(self, temp_db):
        """Same username returns same store instances."""
        factory = StoreFactory(temp_db)

        stores1 = factory.get_stores("alex")
        stores2 = factory.get_stores("alex")

        # Should be exact same objects
        assert stores1 is stores2
        assert stores1.profile is stores2.profile
        assert stores1.feedback is stores2.feedback
        assert stores1.history is stores2.history
        assert stores1.recipe_box is stores2.recipe_box
        assert stores1.session is stores2.session
        assert stores1.meal_plan is stores2.meal_plan

    def test_different_users_different_instances(self, temp_db):
        """Different usernames get different store instances."""
        factory = StoreFactory(temp_db)

        alex_stores = factory.get_stores("alex")
        caitlyn_stores = factory.get_stores("caitlyn")

        # Should be different objects
        assert alex_stores is not caitlyn_stores
        assert alex_stores.profile is not caitlyn_stores.profile
        assert alex_stores.feedback is not caitlyn_stores.feedback
        assert alex_stores.history is not caitlyn_stores.history
        assert alex_stores.recipe_box is not caitlyn_stores.recipe_box
        assert alex_stores.session is not caitlyn_stores.session
        assert alex_stores.meal_plan is not caitlyn_stores.meal_plan

    def test_clear_cache_specific_user(self, temp_db):
        """clear_cache with username clears only that user."""
        factory = StoreFactory(temp_db)

        # Create caches for both users
        factory.get_stores("alex")
        factory.get_stores("caitlyn")

        # Clear only Alex's cache
        factory.clear_cache("alex")

        assert "alex" not in factory._cache
        assert "caitlyn" in factory._cache

    def test_clear_cache_all_users(self, temp_db):
        """clear_cache without username clears all users."""
        factory = StoreFactory(temp_db)

        # Create caches for both users
        factory.get_stores("alex")
        factory.get_stores("caitlyn")

        # Clear all caches
        factory.clear_cache()

        assert len(factory._cache) == 0

    def test_cleared_user_gets_new_instances(self, temp_db):
        """After clearing cache, user gets new store instances."""
        factory = StoreFactory(temp_db)

        stores1 = factory.get_stores("alex")
        factory.clear_cache("alex")
        stores2 = factory.get_stores("alex")

        # Should be different objects (new instances)
        assert stores1 is not stores2
        assert stores1.profile is not stores2.profile

    def test_factory_with_different_db_paths(self, tmp_path):
        """Different factories can use different database paths."""
        db_path1 = tmp_path / "test1.db"
        db_path2 = tmp_path / "test2.db"

        factory1 = StoreFactory(db_path1)
        factory2 = StoreFactory(db_path2)

        stores1 = factory1.get_stores("alex")
        stores2 = factory2.get_stores("alex")

        assert stores1.profile.db_path == db_path1
        assert stores2.profile.db_path == db_path2

    def test_stores_are_correct_types(self, temp_db):
        """Verify stores are the correct class types."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        assert isinstance(stores.profile, ProfileStore)
        assert isinstance(stores.feedback, FeedbackStore)
        assert isinstance(stores.history, HistoryStore)
        assert isinstance(stores.recipe_box, RecipeBoxStore)
        assert isinstance(stores.session, SessionStore)
        assert isinstance(stores.meal_plan, MealPlanStore)

    def test_factory_produces_isolated_stores_per_user(self, temp_db):
        """Verify factory produces isolated store instances per user (sanity check).

        This test verifies the key multi-user isolation property: different users
        get different store instances, and the same user always gets the same
        (cached) instance.
        """
        factory = StoreFactory(temp_db)
        alice_stores = factory.get_stores("alice")
        bob_stores = factory.get_stores("bob")

        # Different UserStores instances
        assert alice_stores is not bob_stores

        # Each store type is different instance
        assert alice_stores.profile is not bob_stores.profile
        assert alice_stores.feedback is not bob_stores.feedback
        assert alice_stores.history is not bob_stores.history
        assert alice_stores.recipe_box is not bob_stores.recipe_box
        assert alice_stores.session is not bob_stores.session
        assert alice_stores.meal_plan is not bob_stores.meal_plan

        # Each store is bound to correct user
        assert alice_stores.profile.user == "alice"
        assert bob_stores.profile.user == "bob"
        assert alice_stores.meal_plan.user == "alice"
        assert bob_stores.meal_plan.user == "bob"

        # But same user returns cached instance
        alice_stores_again = factory.get_stores("alice")
        assert alice_stores is alice_stores_again


class TestBaseUserBoundStore:
    """Test BaseUserBoundStore behavior via concrete implementations."""

    def test_user_property_returns_bound_username(self, temp_db):
        """user property returns the bound username."""
        store = ProfileStore(temp_db, username="alex")
        assert store.user == "alex"

    def test_user_property_is_readonly(self, temp_db):
        """Cannot assign to store.user after init."""
        store = ProfileStore(temp_db, username="alex")
        with pytest.raises(AttributeError):
            store.user = "caitlyn"

    def test_default_username_is_guest(self, temp_db):
        """Default username is 'guest' when not specified."""
        store = ProfileStore(temp_db)
        assert store.user == "guest"

    def test_db_path_is_stored(self, temp_db):
        """db_path is stored on the store instance."""
        store = ProfileStore(temp_db, username="alex")
        assert store.db_path == temp_db


class TestUserSwitchingWithFactory:
    """Test realistic user switching scenarios."""

    def test_user_switch_preserves_data(self, temp_db):
        """Switching users and back preserves each user's data."""
        from src.domain.models import PreferenceProfile

        factory = StoreFactory(temp_db)

        # Alex saves preferences
        alex_stores = factory.get_stores("alex")
        alex_stores.profile.save(PreferenceProfile(spice_level="hot"))

        # Switch to Caitlyn
        caitlyn_stores = factory.get_stores("caitlyn")
        caitlyn_stores.profile.save(PreferenceProfile(spice_level="mild"))

        # Switch back to Alex
        alex_stores_again = factory.get_stores("alex")

        # Alex's preferences should still be there
        alex_profile = alex_stores_again.profile.load()
        assert alex_profile.spice_level == "hot"

        # Caitlyn's should still be different
        caitlyn_profile = caitlyn_stores.profile.load()
        assert caitlyn_profile.spice_level == "mild"

    def test_rapid_user_switching(self, temp_db):
        """Rapid switching between users works correctly."""
        from src.domain.models import RecipeFeedback

        factory = StoreFactory(temp_db)

        users = ["alex", "caitlyn", "guest"]

        # Each user likes a different recipe
        for i, user in enumerate(users):
            stores = factory.get_stores(user)
            stores.feedback.add_feedback(RecipeFeedback(
                recipe_id=f"recipe_{i + 1}",
                feedback_type="like",
                session_id="session1"
            ))

        # Verify each user only sees their own likes
        for i, user in enumerate(users):
            stores = factory.get_stores(user)
            likes = stores.feedback.get_liked_recipe_ids()
            assert f"recipe_{i + 1}" in likes
            assert len(likes) == 1

    def test_stores_independent_after_factory_clear(self, temp_db):
        """After clearing factory cache, new stores are independent."""
        from src.domain.models import PreferenceProfile

        factory = StoreFactory(temp_db)

        # Create stores and save data
        stores = factory.get_stores("alex")
        stores.profile.save(PreferenceProfile(spice_level="hot"))

        # Clear cache
        factory.clear_cache("alex")

        # Get new stores
        new_stores = factory.get_stores("alex")

        # Data should still be in database
        profile = new_stores.profile.load()
        assert profile.spice_level == "hot"


class TestUserStoresDataclass:
    """Test UserStores dataclass."""

    def test_userstores_has_all_fields(self, temp_db):
        """UserStores has all expected fields."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        # Check all fields exist
        assert hasattr(stores, "profile")
        assert hasattr(stores, "feedback")
        assert hasattr(stores, "history")
        assert hasattr(stores, "recipe_box")
        assert hasattr(stores, "session")
        assert hasattr(stores, "meal_plan")

    def test_userstores_fields_accessible(self, temp_db):
        """UserStores fields are accessible as attributes."""
        factory = StoreFactory(temp_db)
        stores = factory.get_stores("alex")

        # Should be able to access without errors
        _ = stores.profile
        _ = stores.feedback
        _ = stores.history
        _ = stores.recipe_box
        _ = stores.session
        _ = stores.meal_plan
