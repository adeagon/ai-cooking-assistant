"""Tests for FeedbackStore."""

import sqlite3
from datetime import datetime
from pathlib import Path
import pytest
from src.domain.models import RecipeFeedback
from src.memory.feedback_store import FeedbackStore


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"

    # Create recipes table (required for foreign key)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT,
            tags TEXT
        )
    """)
    # Add a test recipe
    cursor.execute("INSERT INTO recipes VALUES ('123', 'Test Recipe', '[\"italian\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('456', 'Another Recipe', '[\"mexican\"]')")
    cursor.execute("INSERT INTO recipes VALUES ('789', 'Third Recipe', '[\"italian\", \"pasta\"]')")
    conn.commit()
    conn.close()

    return db_path


class TestFeedbackStore:
    """Test FeedbackStore functionality."""

    def test_initialization_creates_table(self, temp_db):
        """Test that initialization creates the feedback table."""
        store = FeedbackStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='recipe_feedback'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_add_like_feedback(self, temp_db):
        """Test adding a like."""
        store = FeedbackStore(temp_db)

        feedback = RecipeFeedback(
            recipe_id="123",
            feedback_type="like",
            session_id="session1"
        )

        feedback_id = store.add_feedback(feedback)
        assert feedback_id > 0

    def test_add_dislike_feedback(self, temp_db):
        """Test adding a dislike."""
        store = FeedbackStore(temp_db)

        feedback = RecipeFeedback(
            recipe_id="123",
            feedback_type="dislike",
            session_id="session1"
        )

        feedback_id = store.add_feedback(feedback)
        assert feedback_id > 0

    def test_add_rating(self, temp_db):
        """Test adding a 1-5 rating."""
        store = FeedbackStore(temp_db)

        feedback = RecipeFeedback(
            recipe_id="123",
            feedback_type="rate",
            rating=4,
            session_id="session1"
        )

        feedback_id = store.add_feedback(feedback)
        assert feedback_id > 0

        # Verify rating was stored
        feedbacks = store.get_feedback_for_recipe("123")
        assert len(feedbacks) == 1
        assert feedbacks[0].rating == 4

    def test_get_liked_recipe_ids(self, temp_db):
        """Test retrieving liked IDs for exclusion."""
        store = FeedbackStore(temp_db)

        # Add some likes
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))
        store.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="like"))
        store.add_feedback(RecipeFeedback(recipe_id="789", feedback_type="dislike"))

        liked_ids = store.get_liked_recipe_ids()
        assert "123" in liked_ids
        assert "456" in liked_ids
        assert "789" not in liked_ids

    def test_get_liked_recipe_ids_with_limit(self, temp_db):
        """Test limit parameter for liked recipes."""
        store = FeedbackStore(temp_db)

        # Add multiple likes
        for i in range(5):
            store.add_feedback(RecipeFeedback(recipe_id=f"{i}", feedback_type="like"))

        liked_ids = store.get_liked_recipe_ids(limit=3)
        assert len(liked_ids) <= 3

    def test_get_disliked_recipe_ids(self, temp_db):
        """Test retrieving disliked IDs."""
        store = FeedbackStore(temp_db)

        # Add some dislikes
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="dislike"))
        store.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="like"))

        disliked_ids = store.get_disliked_recipe_ids()
        assert "123" in disliked_ids
        assert "456" not in disliked_ids

    def test_get_feedback_for_recipe(self, temp_db):
        """Test getting all feedback for a specific recipe."""
        store = FeedbackStore(temp_db)

        # Add multiple feedbacks for same recipe
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=5))
        store.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="like"))

        feedbacks = store.get_feedback_for_recipe("123")
        assert len(feedbacks) == 2

    def test_get_average_rating(self, temp_db):
        """Test calculating average rating for a recipe."""
        store = FeedbackStore(temp_db)

        # Add multiple ratings
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=4))
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=5))
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="rate", rating=3))

        avg = store.get_average_rating("123")
        assert avg == pytest.approx(4.0)

    def test_get_average_rating_no_ratings(self, temp_db):
        """Test average rating when no ratings exist."""
        store = FeedbackStore(temp_db)

        avg = store.get_average_rating("nonexistent")
        assert avg is None

    def test_get_preferred_cuisines_from_likes(self, temp_db):
        """Test learning cuisines from likes."""
        store = FeedbackStore(temp_db)

        # Like Italian recipes
        store.add_feedback(RecipeFeedback(recipe_id="123", feedback_type="like"))  # italian
        store.add_feedback(RecipeFeedback(recipe_id="789", feedback_type="like"))  # italian

        # Not enough for Mexican (only 1 like)
        store.add_feedback(RecipeFeedback(recipe_id="456", feedback_type="like"))  # mexican

        cuisines = store.get_preferred_cuisines_from_likes(min_count=2)
        assert "italian" in cuisines
        assert "mexican" not in cuisines

    def test_get_preferred_cuisines_empty(self, temp_db):
        """Test cuisine learning with no likes."""
        store = FeedbackStore(temp_db)

        cuisines = store.get_preferred_cuisines_from_likes()
        assert len(cuisines) == 0
