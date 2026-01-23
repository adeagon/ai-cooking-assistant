"""Tests for HistoryStore."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from src.domain.models import CookingHistoryEntry
from src.memory.history_store import HistoryStore


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
            title TEXT
        )
    """)
    # Add test recipes
    cursor.execute("INSERT INTO recipes VALUES ('123', 'Test Recipe')")
    cursor.execute("INSERT INTO recipes VALUES ('456', 'Another Recipe')")
    conn.commit()
    conn.close()

    return db_path


class TestHistoryStore:
    """Test HistoryStore functionality."""

    def test_initialization_creates_table(self, temp_db):
        """Test that initialization creates the cooking history table."""
        store = HistoryStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='cooking_history'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_add_cooked(self, temp_db):
        """Test recording a cooked recipe."""
        store = HistoryStore(temp_db)

        history_id = store.add_cooked("123")
        assert history_id > 0

    def test_add_cooked_with_notes(self, temp_db):
        """Test recording a cooked recipe with notes."""
        store = HistoryStore(temp_db)

        history_id = store.add_cooked("123", notes="Delicious!")
        assert history_id > 0

        # Verify notes were stored
        history = store.get_cooking_history(limit=1)
        assert len(history) == 1
        assert history[0].notes == "Delicious!"

    def test_get_recently_cooked_ids(self, temp_db):
        """Test getting recently cooked recipe IDs."""
        store = HistoryStore(temp_db)

        # Add some cooked recipes
        store.add_cooked("123")
        store.add_cooked("456")

        recently_cooked = store.get_recently_cooked_ids(days=7)
        assert "123" in recently_cooked
        assert "456" in recently_cooked

    def test_get_recently_cooked_ids_with_date_filter(self, temp_db):
        """Test filtering by date range."""
        store = HistoryStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Manually insert old cooked entry (15 days ago)
        old_date = datetime.now() - timedelta(days=15)
        cursor.execute("""
            INSERT INTO cooking_history (recipe_id, cooked_at, username)
            VALUES ('123', ?, 'guest')
        """, (old_date,))

        # Add recent entry (today)
        cursor.execute("""
            INSERT INTO cooking_history (recipe_id, cooked_at, username)
            VALUES ('456', ?, 'guest')
        """, (datetime.now(),))

        conn.commit()
        conn.close()

        # Get recipes cooked in last 7 days
        recently_cooked = store.get_recently_cooked_ids(days=7)
        assert "456" in recently_cooked
        assert "123" not in recently_cooked

        # Get recipes cooked in last 30 days
        all_cooked = store.get_recently_cooked_ids(days=30)
        assert "456" in all_cooked
        assert "123" in all_cooked

    def test_get_cooking_history(self, temp_db):
        """Test retrieving cooking history."""
        store = HistoryStore(temp_db)

        # Add multiple entries
        store.add_cooked("123")
        store.add_cooked("456")
        store.add_cooked("123")  # Cook same recipe again

        history = store.get_cooking_history(limit=10)
        assert len(history) == 3
        assert all(isinstance(entry, CookingHistoryEntry) for entry in history)

    def test_get_cooking_history_with_limit(self, temp_db):
        """Test limit parameter for cooking history."""
        store = HistoryStore(temp_db)

        # Add multiple entries
        for i in range(5):
            store.add_cooked(f"{i}")

        history = store.get_cooking_history(limit=3)
        assert len(history) == 3

    def test_get_cooking_history_order(self, temp_db):
        """Test that history is ordered by most recent first."""
        store = HistoryStore(temp_db)

        # Add entries in order
        id1 = store.add_cooked("123")
        id2 = store.add_cooked("456")

        history = store.get_cooking_history(limit=10)
        # Most recent should be first
        assert history[0].recipe_id == "456"
        assert history[1].recipe_id == "123"

    def test_get_cooking_count(self, temp_db):
        """Test getting count of times a recipe was cooked."""
        store = HistoryStore(temp_db)

        # Cook recipe multiple times
        store.add_cooked("123")
        store.add_cooked("123")
        store.add_cooked("456")

        count_123 = store.get_cooking_count("123")
        count_456 = store.get_cooking_count("456")
        count_nonexistent = store.get_cooking_count("nonexistent")

        assert count_123 == 2
        assert count_456 == 1
        assert count_nonexistent == 0

    def test_empty_history(self, temp_db):
        """Test behavior with no cooking history."""
        store = HistoryStore(temp_db)

        history = store.get_cooking_history()
        recently_cooked = store.get_recently_cooked_ids()

        assert len(history) == 0
        assert len(recently_cooked) == 0
