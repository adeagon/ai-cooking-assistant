"""Tests for multi-user data isolation.

These tests verify that user data is properly isolated - one user's data
should never be visible to or affect another user.
"""

import pytest

from src.domain.models import PreferenceProfile, RecipeFeedback
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from src.memory.meal_plan_store import MealPlanStore
from src.memory.profile_store import ProfileStore
from src.memory.recipe_box_store import RecipeBoxStore
from src.memory.session_store import SessionStore


# Second test user ID for isolation tests
TEST_USER_2_ID = "test-user-00000000-0000-0000-0000-000000000002"


@pytest.fixture
def second_user_id(temp_db):
    """Create a second test user and return their ID."""
    import sqlite3

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (id, username, is_active) VALUES (?, ?, ?)",
        (TEST_USER_2_ID, "test_user_2", True)
    )
    conn.commit()
    conn.close()

    return TEST_USER_2_ID


class TestFeedbackIsolation:
    """Test that feedback data is isolated between users."""

    def test_liked_recipes_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's liked recipes should not appear for User 2."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 likes a recipe
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))

        # User 2 should not see it
        assert "123" in store1.get_liked_recipe_ids()
        assert "123" not in store2.get_liked_recipe_ids()

    def test_disliked_recipes_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's disliked recipes should not appear for User 2."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 dislikes a recipe
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="dislike"))

        assert "123" in store1.get_disliked_recipe_ids()
        assert "123" not in store2.get_disliked_recipe_ids()

    def test_ratings_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's ratings should not affect User 2."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 rates a recipe
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=5))

        # User 1 sees their rating (returns list of feedback)
        feedback1 = store1.get_feedback_for_recipe("123")
        assert len(feedback1) == 1
        assert feedback1[0].rating == 5

        # User 2 sees nothing (empty list)
        feedback2 = store2.get_feedback_for_recipe("123")
        assert len(feedback2) == 0

    def test_both_users_can_like_same_recipe(self, temp_db, test_user_id, second_user_id):
        """Both users can independently like the same recipe."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # Both users like same recipe
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))
        store2.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))

        # Both see it in their lists
        assert "123" in store1.get_liked_recipe_ids()
        assert "123" in store2.get_liked_recipe_ids()

    def test_preferred_cuisines_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's cuisine preferences don't affect User 2."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 likes multiple Italian recipes to meet min_count threshold
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))  # italian
        store1.add_feedback(RecipeFeedback(recipe_id="789", feedback_type="like"))  # italian, pasta
        store1.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="like"))  # mexican

        # User 1 can query cuisines (may need min_count=1 for test)
        prefs1 = store1.get_preferred_cuisines_from_likes(min_count=1)

        # User 2 sees nothing
        prefs2 = store2.get_preferred_cuisines_from_likes(min_count=1)
        assert len(prefs2) == 0


class TestHistoryIsolation:
    """Test that cooking history is isolated between users."""

    def test_cooked_recipes_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's cooked recipes should not appear for User 2."""
        store1 = HistoryStore(temp_db, test_user_id)
        store2 = HistoryStore(temp_db, second_user_id)

        # User 1 cooks a recipe
        store1.add_cooked("123")

        # User 1 sees it
        assert "123" in store1.get_recently_cooked_ids()

        # User 2 doesn't see it
        assert "123" not in store2.get_recently_cooked_ids()

    def test_cooking_history_isolated(self, temp_db, test_user_id, second_user_id):
        """Full cooking history is isolated."""
        store1 = HistoryStore(temp_db, test_user_id)
        store2 = HistoryStore(temp_db, second_user_id)

        # User 1 cooks multiple recipes
        store1.add_cooked("123", notes="Great!")
        store1.add_cooked("456")

        # User 1 sees 2 entries
        history1 = store1.get_cooking_history()
        assert len(history1) == 2

        # User 2 sees none
        history2 = store2.get_cooking_history()
        assert len(history2) == 0

    def test_both_users_can_cook_same_recipe(self, temp_db, test_user_id, second_user_id):
        """Both users can independently cook the same recipe."""
        store1 = HistoryStore(temp_db, test_user_id)
        store2 = HistoryStore(temp_db, second_user_id)

        # Both cook same recipe with different notes
        store1.add_cooked("123", notes="User 1 notes")
        store2.add_cooked("123", notes="User 2 notes")

        # Each sees their own entry
        history1 = store1.get_cooking_history()
        history2 = store2.get_cooking_history()

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].notes == "User 1 notes"
        assert history2[0].notes == "User 2 notes"


class TestRecipeBoxIsolation:
    """Test that saved recipes are isolated between users."""

    def test_saved_recipes_isolated(self, temp_db, test_user_id, second_user_id):
        """User 1's saved recipes should not appear for User 2."""
        store1 = RecipeBoxStore(temp_db, test_user_id)
        store2 = RecipeBoxStore(temp_db, second_user_id)

        # User 1 saves a recipe
        store1.save_recipe("123", "Test Recipe")

        # User 1 sees it
        saved1 = store1.get_saved_recipes()
        assert len(saved1) == 1
        assert saved1[0].recipe_id == "123"

        # User 2 doesn't see it
        saved2 = store2.get_saved_recipes()
        assert len(saved2) == 0

    def test_is_saved_isolated(self, temp_db, test_user_id, second_user_id):
        """is_saved returns correct value per user."""
        store1 = RecipeBoxStore(temp_db, test_user_id)
        store2 = RecipeBoxStore(temp_db, second_user_id)

        store1.save_recipe("123", "Test Recipe")

        assert store1.is_saved("123") is True
        assert store2.is_saved("123") is False

    def test_both_users_can_save_same_recipe(self, temp_db, test_user_id, second_user_id):
        """Both users can save the same recipe independently."""
        store1 = RecipeBoxStore(temp_db, test_user_id)
        store2 = RecipeBoxStore(temp_db, second_user_id)

        # Both save same recipe with different notes
        store1.save_recipe("123", "Test Recipe", notes="User 1 notes")
        store2.save_recipe("123", "Test Recipe", notes="User 2 notes")

        saved1 = store1.get_saved_recipes()
        saved2 = store2.get_saved_recipes()

        assert len(saved1) == 1
        assert len(saved2) == 1
        assert saved1[0].notes == "User 1 notes"
        assert saved2[0].notes == "User 2 notes"

    def test_remove_only_affects_own_saved(self, temp_db, test_user_id, second_user_id):
        """Removing a recipe only affects that user's saved list."""
        store1 = RecipeBoxStore(temp_db, test_user_id)
        store2 = RecipeBoxStore(temp_db, second_user_id)

        # Both save same recipe
        store1.save_recipe("123", "Test Recipe")
        store2.save_recipe("123", "Test Recipe")

        # User 1 removes it
        store1.remove_recipe("123")

        # User 1 no longer has it
        assert store1.is_saved("123") is False

        # User 2 still has it
        assert store2.is_saved("123") is True


class TestProfileIsolation:
    """Test that user preferences are isolated between users."""

    def test_profiles_isolated(self, temp_db, test_user_id, second_user_id):
        """Each user has their own profile."""
        store1 = ProfileStore(temp_db, test_user_id)
        store2 = ProfileStore(temp_db, second_user_id)

        # User 1 sets preferences
        profile1 = PreferenceProfile(
            spice_level="hot",
            diet="vegan",
            avoid_ingredients=["nuts"],
            preferred_cuisines=["mexican"],
        )
        store1.save(profile1)

        # User 2 gets default profile
        loaded1 = store1.load()
        loaded2 = store2.load()

        assert loaded1.spice_level == "hot"
        assert loaded1.diet == "vegan"

        # User 2 has default values
        assert loaded2.spice_level == "medium"
        assert loaded2.diet == "none"

    def test_profile_update_isolated(self, temp_db, test_user_id, second_user_id):
        """Profile updates don't affect other users."""
        store1 = ProfileStore(temp_db, test_user_id)
        store2 = ProfileStore(temp_db, second_user_id)

        # Both create profiles
        store1.save(PreferenceProfile(spice_level="mild"))
        store2.save(PreferenceProfile(spice_level="medium"))

        # User 1 updates
        store1.update(spice_level="hot")

        # Only User 1's profile changed
        assert store1.load().spice_level == "hot"
        assert store2.load().spice_level == "medium"


class TestSessionIsolation:
    """Test that sessions are isolated between users."""

    def test_sessions_isolated(self, temp_db, test_user_id, second_user_id):
        """Each user has their own session."""
        store1 = SessionStore(temp_db, test_user_id)
        store2 = SessionStore(temp_db, second_user_id)

        # User 1 creates and updates session
        session_id1 = store1.create()
        store1.update(session_id1, ingredients_on_hand=["chicken", "rice"])

        # User 2 creates session
        session_id2 = store2.create()

        # Sessions are different
        assert session_id1 != session_id2

        # User 1's session has ingredients
        session1 = store1.get(session_id1)
        assert session1.ingredients_on_hand == ["chicken", "rice"]

        # User 2's session is empty
        session2 = store2.get(session_id2)
        assert session2.ingredients_on_hand == []

    def test_get_or_create_current_isolated(self, temp_db, test_user_id, second_user_id):
        """get_or_create_current returns user-specific sessions."""
        store1 = SessionStore(temp_db, test_user_id)
        store2 = SessionStore(temp_db, second_user_id)

        # Each user gets their own current session
        session_id1, _ = store1.get_or_create_current()
        session_id2, _ = store2.get_or_create_current()

        # Different sessions
        assert session_id1 != session_id2


class TestMealPlanIsolation:
    """Test that meal plans are isolated between users."""

    def test_plans_isolated(self, temp_db, test_user_id, second_user_id):
        """Each user has their own meal plans."""
        from datetime import date, timedelta

        from src.domain.models import MealPlan, PlannedMeal

        store1 = MealPlanStore(temp_db, test_user_id)
        store2 = MealPlanStore(temp_db, second_user_id)

        # User 1 creates a plan
        plan = MealPlan(
            name="User 1 Plan",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=2),
            meal_types=["dinner"],
            status="draft",
            meals=[],
        )
        plan_id = store1.create_plan(plan)

        # User 1 sees their plan
        assert store1.get_plan(plan_id) is not None
        assert store1.get_plan_count() == 1

        # User 2 sees nothing
        assert store2.get_plan_count() == 0

    def test_active_plan_isolated(self, temp_db, test_user_id, second_user_id):
        """Active plan is user-specific."""
        from datetime import date, timedelta

        from src.domain.models import MealPlan

        store1 = MealPlanStore(temp_db, test_user_id)
        store2 = MealPlanStore(temp_db, second_user_id)

        # User 1 creates and activates a plan
        plan1 = MealPlan(
            name="User 1 Active",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=2),
            meal_types=["dinner"],
            status="active",
            meals=[],
        )
        plan_id1 = store1.create_plan(plan1)
        store1.update_plan_status(plan_id1, "active")

        # User 2 creates their own plan
        plan2 = MealPlan(
            name="User 2 Plan",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=2),
            meal_types=["dinner"],
            status="draft",
            meals=[],
        )
        plan_id2 = store2.create_plan(plan2)

        # Each user's active plan is different
        active1 = store1.get_active_plan()
        active2 = store2.get_active_plan()

        assert active1 is not None
        assert active1.name == "User 1 Active"
        assert active2 is None  # User 2's plan is draft, not active


class TestCrossUserDataAccess:
    """Test that users cannot access each other's data even with IDs."""

    def test_cannot_see_other_user_feedback_by_recipe(self, temp_db, test_user_id, second_user_id):
        """User cannot see another user's feedback for a specific recipe."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 rates a recipe
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=5))

        # User 2 queries for same recipe
        feedback = store2.get_feedback_for_recipe("123")

        # User 2 sees empty list (get_feedback_for_recipe returns list)
        assert len(feedback) == 0

    def test_users_have_independent_averages(self, temp_db, test_user_id, second_user_id):
        """Each user's average rating is calculated independently per recipe."""
        store1 = FeedbackStore(temp_db, test_user_id)
        store2 = FeedbackStore(temp_db, second_user_id)

        # User 1 rates recipe 123 highly
        store1.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=5))

        # User 2 rates recipe 123 lower
        store2.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=2))

        # Each user sees only their own rating
        avg1 = store1.get_average_rating("123")
        avg2 = store2.get_average_rating("123")

        assert avg1 == 5.0
        assert avg2 == 2.0
