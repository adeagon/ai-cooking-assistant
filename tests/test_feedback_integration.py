"""Integration tests for Phase 5 feedback functionality.

These tests verify the complete feedback workflow including CLI commands,
recipe reference resolution, exclusion filtering, and full recipe display.
"""

import sqlite3
from pathlib import Path
import pytest
from src.app.cli import resolve_recipe_reference, display_full_recipe
from src.domain.models import Recipe, RecipeCard, RecipeFeedback
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from rich.console import Console


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with test recipes."""
    db_path = tmp_path / "test.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create recipes table
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            ingredients_raw TEXT,
            ingredients_normalized TEXT,
            instructions TEXT,
            tags TEXT,
            rating_avg REAL,
            rating_count INTEGER,
            minutes INTEGER,
            n_steps INTEGER,
            n_ingredients INTEGER,
            source TEXT DEFAULT 'foodcom',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add test recipes
    import json
    test_recipes = [
        ("123", "Spicy Chicken Tacos", '["chicken", "taco shells", "salsa"]',
         '["chicken", "taco shells", "salsa"]', '["Cook chicken", "Assemble tacos"]',
         '["mexican", "spicy"]', 4.5, 100, 30, 2, 3),
        ("456", "Lemon Herb Chicken", '["chicken", "lemon", "herbs"]',
         '["chicken", "lemon", "herbs"]', '["Marinate chicken", "Grill chicken"]',
         '["healthy", "quick"]', 4.8, 200, 25, 2, 3),
        ("789", "BBQ Chicken Wings", '["chicken wings", "bbq sauce"]',
         '["chicken wings", "bbq sauce"]', '["Season wings", "Bake wings", "Coat with sauce"]',
         '["american", "comfort"]', 4.3, 150, 45, 3, 2),
    ]

    for recipe in test_recipes:
        cursor.execute("""
            INSERT INTO recipes
            (recipe_id, title, ingredients_raw, ingredients_normalized, instructions,
             tags, rating_avg, rating_count, minutes, n_steps, n_ingredients)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, recipe)

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def recipe_cards():
    """Create test recipe cards."""
    return [
        RecipeCard(
            recipe_id="123",
            title="Spicy Chicken Tacos",
            rating_avg=4.5,
            rating_count=100,
            tags=["mexican", "spicy"],
            time_total=30,
            key_ingredients=["chicken", "taco shells", "salsa"]
        ),
        RecipeCard(
            recipe_id="456",
            title="Lemon Herb Chicken",
            rating_avg=4.8,
            rating_count=200,
            tags=["healthy", "quick"],
            time_total=25,
            key_ingredients=["chicken", "lemon", "herbs"]
        ),
        RecipeCard(
            recipe_id="789",
            title="BBQ Chicken Wings",
            rating_avg=4.3,
            rating_count=150,
            tags=["american", "comfort"],
            time_total=45,
            key_ingredients=["chicken wings", "bbq sauce"]
        ),
    ]


class TestRecipeReferenceResolver:
    """Test recipe reference resolution (by number or name)."""

    def test_resolve_by_number(self, recipe_cards):
        """Test resolving recipe by number (1-indexed)."""
        result = resolve_recipe_reference("1", recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "123"
        assert title == "Spicy Chicken Tacos"

    def test_resolve_by_number_second_recipe(self, recipe_cards):
        """Test resolving second recipe by number."""
        result = resolve_recipe_reference("2", recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "456"
        assert title == "Lemon Herb Chicken"

    def test_resolve_by_number_out_of_range(self, recipe_cards):
        """Test resolving with out-of-range number."""
        result = resolve_recipe_reference("10", recipe_cards)
        assert result is None

    def test_resolve_by_exact_name(self, recipe_cards):
        """Test resolving by exact recipe name."""
        result = resolve_recipe_reference("Spicy Chicken Tacos", recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "123"

    def test_resolve_by_partial_name(self, recipe_cards):
        """Test resolving by partial recipe name (fuzzy match)."""
        result = resolve_recipe_reference("Lemon", recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "456"
        assert "Lemon" in title

    def test_resolve_by_name_case_insensitive(self, recipe_cards):
        """Test case-insensitive name matching."""
        result = resolve_recipe_reference("chicken tacos", recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "123"

    def test_resolve_with_quotes(self, recipe_cards):
        """Test resolving with quoted name."""
        result = resolve_recipe_reference('"Spicy Chicken Tacos"', recipe_cards)
        assert result is not None
        recipe_id, title = result
        assert recipe_id == "123"

    def test_resolve_empty_reference(self, recipe_cards):
        """Test resolving empty reference."""
        result = resolve_recipe_reference("", recipe_cards)
        assert result is None

    def test_resolve_no_match(self, recipe_cards):
        """Test resolving with no matching recipe."""
        result = resolve_recipe_reference("Pizza", recipe_cards)
        assert result is None


class TestExclusionFiltering:
    """Test that liked/disliked/cooked recipes are excluded from recommendations."""

    def test_liked_recipe_excluded(self, temp_db):
        """Test that liked recipe is excluded from search results."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Like recipe 123
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="like"
        ))

        # Compute exclusion set (like in CLI)
        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        # Verify recipe 123 is in exclusion set
        assert "123" in exclude_ids
        assert "456" not in exclude_ids
        assert "789" not in exclude_ids

    def test_disliked_recipe_excluded(self, temp_db):
        """Test that disliked recipe is excluded."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Dislike recipe 456
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="456",
            feedback_type="dislike"
        ))

        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        assert "456" in exclude_ids

    def test_cooked_recipe_excluded(self, temp_db):
        """Test that recently cooked recipe is excluded."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Mark recipe 789 as cooked
        history_store.add_cooked("789")

        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        assert "789" in exclude_ids

    def test_multiple_exclusions_combined(self, temp_db):
        """Test that all three exclusion types work together."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Like recipe 123
        feedback_store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))

        # Dislike recipe 456
        feedback_store.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="dislike"))

        # Cook recipe 789
        history_store.add_cooked("789")

        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        # All three should be excluded
        assert "123" in exclude_ids
        assert "456" in exclude_ids
        assert "789" in exclude_ids
        assert len(exclude_ids) == 3


class TestFeedbackWorkflow:
    """Test complete feedback workflow scenarios."""

    def test_like_and_rate_same_recipe(self, temp_db):
        """Test liking then rating the same recipe."""
        feedback_store = FeedbackStore(temp_db)

        # Like recipe
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="like"
        ))

        # Rate same recipe
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="rate",
            rating=5
        ))

        # Both feedbacks should be stored
        feedbacks = feedback_store.get_feedback_for_recipe("123")
        assert len(feedbacks) == 2

        # Recipe should still be in liked set
        liked_ids = feedback_store.get_liked_recipe_ids()
        assert "123" in liked_ids

    def test_like_then_dislike_recipe(self, temp_db):
        """Test changing mind from like to dislike."""
        feedback_store = FeedbackStore(temp_db)

        # Like recipe
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="like"
        ))

        # Later dislike it
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="dislike"
        ))

        # Both feedbacks stored (user changed mind)
        feedbacks = feedback_store.get_feedback_for_recipe("123")
        assert len(feedbacks) == 2

        # Recipe should be in BOTH liked and disliked sets
        # (Most recent action matters in practice)
        liked_ids = feedback_store.get_liked_recipe_ids()
        disliked_ids = feedback_store.get_disliked_recipe_ids()
        assert "123" in liked_ids
        assert "123" in disliked_ids

    def test_cook_then_like_recipe(self, temp_db):
        """Test cooking a recipe then liking it."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Cook recipe
        history_store.add_cooked("123")

        # Like it after cooking
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id="123",
            feedback_type="like"
        ))

        # Should be in both exclusion sets
        exclude_ids = (
            feedback_store.get_liked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )
        assert "123" in exclude_ids


class TestCookingHistory:
    """Test cooking history functionality."""

    def test_history_shows_most_recent_first(self, temp_db):
        """Test that cooking history is ordered by most recent."""
        history_store = HistoryStore(temp_db)

        # Cook recipes in order
        history_store.add_cooked("123")  # First
        history_store.add_cooked("456")  # Second
        history_store.add_cooked("789")  # Third (most recent)

        history = history_store.get_cooking_history(limit=10)

        # Most recent should be first
        assert len(history) == 3
        assert history[0].recipe_id == "789"
        assert history[1].recipe_id == "456"
        assert history[2].recipe_id == "123"

    def test_cook_same_recipe_multiple_times(self, temp_db):
        """Test cooking the same recipe multiple times."""
        history_store = HistoryStore(temp_db)

        # Cook same recipe three times
        history_store.add_cooked("123")
        history_store.add_cooked("123")
        history_store.add_cooked("123")

        # All three entries should be tracked
        history = history_store.get_cooking_history(limit=10)
        assert len(history) == 3
        assert all(entry.recipe_id == "123" for entry in history)

        # Cooking count should be 3
        count = history_store.get_cooking_count("123")
        assert count == 3


class TestFullRecipeDisplay:
    """Test full recipe display functionality."""

    def test_display_full_recipe_basic(self, temp_db):
        """Test that display_full_recipe works without errors."""
        from src.ingest.build_db import get_recipe_by_id
        from io import StringIO

        recipe = get_recipe_by_id(temp_db, "123")
        assert recipe is not None

        # Create console with StringIO to capture output
        output = StringIO()
        console = Console(file=output, force_terminal=True)

        # Should not raise an error
        display_full_recipe(recipe, console)

        # Verify something was printed
        output_text = output.getvalue()
        assert len(output_text) > 0
        assert "Spicy Chicken Tacos" in output_text

    def test_recipe_has_required_fields(self, temp_db):
        """Test that recipes have all fields needed for display."""
        from src.ingest.build_db import get_recipe_by_id

        recipe = get_recipe_by_id(temp_db, "123")

        assert recipe is not None
        assert recipe.title is not None
        assert recipe.ingredients is not None
        assert recipe.instructions is not None
        assert len(recipe.ingredients) > 0
        assert len(recipe.instructions) > 0


@pytest.mark.integration
class TestPhase5EndToEnd:
    """End-to-end integration tests for Phase 5 features.

    These tests verify the complete workflow would work in practice.
    They test storage and logic but not the actual CLI (which requires Ollama).
    """

    def test_complete_feedback_workflow(self, temp_db, recipe_cards):
        """Test complete workflow: search -> like -> search again -> verify excluded."""
        feedback_store = FeedbackStore(temp_db)
        history_store = HistoryStore(temp_db)

        # Step 1: User sees recommendations (recipe_cards)
        assert len(recipe_cards) == 3

        # Step 2: User likes recipe 1 using recipe reference resolver
        ref = "1"
        result = resolve_recipe_reference(ref, recipe_cards)
        assert result is not None
        recipe_id, title = result

        # Step 3: Store the feedback
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="like"
        ))

        # Step 4: Compute exclusion set for next search
        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        # Step 5: Verify liked recipe is excluded
        assert recipe_id in exclude_ids

        # Step 6: Filter recipe cards (like RetrievalRunnable does)
        filtered_cards = [card for card in recipe_cards if card.recipe_id not in exclude_ids]

        # Liked recipe should not appear in filtered results
        assert len(filtered_cards) == 2
        assert all(card.recipe_id != recipe_id for card in filtered_cards)

    def test_show_then_cook_workflow(self, temp_db, recipe_cards):
        """Test workflow: search -> show recipe -> cook it -> verify excluded."""
        from src.ingest.build_db import get_recipe_by_id
        history_store = HistoryStore(temp_db)

        # Step 1: User wants to see full recipe for item 1
        result = resolve_recipe_reference("1", recipe_cards)
        assert result is not None
        recipe_id, title = result

        # Step 2: Get full recipe from database
        full_recipe = get_recipe_by_id(temp_db, recipe_id)
        assert full_recipe is not None
        assert len(full_recipe.ingredients) > 0
        assert len(full_recipe.instructions) > 0

        # Step 3: User cooks it
        history_store.add_cooked(recipe_id)

        # Step 4: Verify it appears in history
        history = history_store.get_cooking_history(limit=10)
        assert len(history) == 1
        assert history[0].recipe_id == recipe_id

        # Step 5: Verify it's excluded from future searches
        recently_cooked = history_store.get_recently_cooked_ids(days=7)
        assert recipe_id in recently_cooked

    def test_dislike_permanently_excludes(self, temp_db, recipe_cards):
        """Test that disliked recipe never appears again."""
        feedback_store = FeedbackStore(temp_db)

        # User dislikes a recipe
        result = resolve_recipe_reference("2", recipe_cards)
        recipe_id, title = result

        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="dislike"
        ))

        # Disliked recipes should always be excluded (no time limit)
        disliked_ids = feedback_store.get_disliked_recipe_ids()
        assert recipe_id in disliked_ids

        # Verify exclusion set includes it
        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            HistoryStore(temp_db).get_recently_cooked_ids(days=7)
        )
        assert recipe_id in exclude_ids
