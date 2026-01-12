"""Unit tests for memory system (ProfileStore, SessionStore, RollingSummarizer)."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.domain.models import Constraints, PreferenceProfile, SessionState
from src.memory import ProfileStore, RollingSummarizer, SessionStore


class TestProfileStore:
    """Tests for ProfileStore."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    def test_profile_store_init_creates_table(self, temp_db):
        """Test that ProfileStore creates table on initialization."""
        store = ProfileStore(db_path=temp_db)

        # Check table exists
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='preferences'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_load_default_profile(self, temp_db):
        """Test loading default profile when none exists."""
        store = ProfileStore(db_path=temp_db)
        profile = store.load()

        assert isinstance(profile, PreferenceProfile)
        assert profile.spice_level == "medium"
        assert profile.diet == "none"
        assert profile.avoid_ingredients == []
        assert profile.preferred_cuisines == []

    def test_save_and_load_profile(self, temp_db):
        """Test saving and loading a profile."""
        store = ProfileStore(db_path=temp_db)

        profile = PreferenceProfile(
            spice_level="hot",
            diet="vegetarian",
            avoid_ingredients=["fish", "meat"],
            preferred_cuisines=["italian", "mexican"],
            time_limit_default_minutes=45,
        )

        store.save(profile)
        loaded = store.load()

        assert loaded.spice_level == "hot"
        assert loaded.diet == "vegetarian"
        assert loaded.avoid_ingredients == ["fish", "meat"]
        assert loaded.preferred_cuisines == ["italian", "mexican"]
        assert loaded.time_limit_default_minutes == 45

    def test_update_profile(self, temp_db):
        """Test updating specific profile fields."""
        store = ProfileStore(db_path=temp_db)

        # Save initial profile
        profile = PreferenceProfile(spice_level="mild")
        store.save(profile)

        # Update spice level
        updated = store.update(spice_level="hot", diet="vegan")

        assert updated.spice_level == "hot"
        assert updated.diet == "vegan"

        # Verify persistence
        loaded = store.load()
        assert loaded.spice_level == "hot"
        assert loaded.diet == "vegan"


class TestSessionStore:
    """Tests for SessionStore."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    def test_session_store_init_creates_table(self, temp_db):
        """Test that SessionStore creates table on initialization."""
        store = SessionStore(db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_create_session(self, temp_db):
        """Test creating a new session."""
        store = SessionStore(db_path=temp_db)

        session_id = store.create()

        assert session_id is not None
        assert len(session_id) > 0  # UUID string

        # Verify session exists in DB
        session = store.get(session_id)
        assert session is not None
        assert isinstance(session, SessionState)

    def test_get_nonexistent_session(self, temp_db):
        """Test getting a session that doesn't exist."""
        store = SessionStore(db_path=temp_db)

        session = store.get("nonexistent-id")

        assert session is None

    def test_update_session(self, temp_db):
        """Test updating session fields."""
        store = SessionStore(db_path=temp_db)

        session_id = store.create()

        # Update session
        updated = store.update(
            session_id,
            ingredients_on_hand=["chicken", "tomatoes"],
            goals=["quick", "healthy"],
            time_limit_minutes=30,
        )

        assert updated.ingredients_on_hand == ["chicken", "tomatoes"]
        assert updated.goals == ["quick", "healthy"]
        assert updated.time_limit_minutes == 30

        # Verify persistence
        loaded = store.get(session_id)
        assert loaded.ingredients_on_hand == ["chicken", "tomatoes"]

    def test_get_or_create_current(self, temp_db):
        """Test get_or_create_current method."""
        store = SessionStore(db_path=temp_db)

        session_id1, session1 = store.get_or_create_current()

        assert session_id1 is not None
        assert isinstance(session1, SessionState)

        # Second call should return same session
        session_id2, session2 = store.get_or_create_current()
        assert session_id2 == session_id1

    def test_update_and_get_summary(self, temp_db):
        """Test summary storage and retrieval."""
        store = SessionStore(db_path=temp_db)

        session_id = store.create()

        # Update summary
        store.update_summary(session_id, "ingredients: chicken, tomatoes; time: 30 min")

        # Retrieve summary
        summary = store.get_summary(session_id)

        assert summary == "ingredients: chicken, tomatoes; time: 30 min"


class TestRollingSummarizer:
    """Tests for RollingSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Create RollingSummarizer instance."""
        return RollingSummarizer()

    def test_update_summary_with_ingredients(self, summarizer):
        """Test updating summary with ingredient constraints."""
        constraints = Constraints(ingredients=["chicken", "tomatoes", "garlic"])

        summary = summarizer.update_summary("", constraints, "I have chicken and tomatoes")

        assert "ingredients:" in summary
        assert "chicken" in summary or "tomatoes" in summary

    def test_update_summary_with_time(self, summarizer):
        """Test updating summary with time constraint."""
        constraints = Constraints(time_limit=30)

        summary = summarizer.update_summary("", constraints, "Something quick")

        assert "time: 30 min" in summary

    def test_update_summary_with_dietary(self, summarizer):
        """Test updating summary with dietary constraint."""
        constraints = Constraints(dietary="vegetarian")

        summary = summarizer.update_summary("", constraints, "I'm vegetarian")

        assert "diet: vegetarian" in summary

    def test_update_summary_with_goals(self, summarizer):
        """Test updating summary with goals."""
        constraints = Constraints(goals=["healthy", "quick"])

        summary = summarizer.update_summary("", constraints, "Something healthy and quick")

        assert "goals:" in summary
        assert "healthy" in summary

    def test_update_summary_accumulates(self, summarizer):
        """Test that summary accumulates multiple turns."""
        constraints1 = Constraints(ingredients=["chicken"])
        summary1 = summarizer.update_summary("", constraints1, "I have chicken")

        constraints2 = Constraints(time_limit=30)
        summary2 = summarizer.update_summary(summary1, constraints2, "Under 30 minutes")

        # Both constraints should be present
        assert "ingredients:" in summary2 or "chicken" in summary2
        assert "time:" in summary2 or "30" in summary2

    def test_update_summary_deduplicates(self, summarizer):
        """Test that summary deduplicates categories."""
        constraints1 = Constraints(ingredients=["chicken"])
        summary1 = summarizer.update_summary("", constraints1, "I have chicken")

        constraints2 = Constraints(ingredients=["tomatoes"])
        summary2 = summarizer.update_summary(summary1, constraints2, "Also have tomatoes")

        # Should only have one "ingredients:" entry
        assert summary2.count("ingredients:") <= 1

    def test_update_summary_limits_points(self, summarizer):
        """Test that summary limits to MAX_POINTS."""
        summary = ""

        for i in range(5):
            constraints = Constraints(ingredients=[f"ingredient{i}"])
            summary = summarizer.update_summary(summary, constraints, f"Turn {i}")

        # Should have at most MAX_POINTS entries
        points = summary.split("; ")
        assert len(points) <= RollingSummarizer.MAX_POINTS

    def test_clear_summary(self, summarizer):
        """Test clearing summary."""
        summary = "ingredients: chicken; time: 30 min"

        cleared = summarizer.clear_summary()

        assert cleared == ""

    def test_format_for_prompt_empty(self, summarizer):
        """Test formatting empty summary for prompt."""
        formatted = summarizer.format_for_prompt("")

        assert formatted == ""

    def test_format_for_prompt_with_content(self, summarizer):
        """Test formatting non-empty summary for prompt."""
        formatted = summarizer.format_for_prompt("ingredients: chicken; time: 30 min")

        assert "Previous discussion points:" in formatted
        assert "ingredients: chicken" in formatted
