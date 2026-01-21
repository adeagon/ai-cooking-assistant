"""Tests for WebSessionStore."""

import json
import sqlite3
import time
from datetime import datetime, timedelta

import pytest

from src.memory.web_session_store import WebMessage, WebSession, WebSessionStore


@pytest.fixture
def web_store(temp_db, test_user_id):
    """Create a WebSessionStore instance."""
    return WebSessionStore(temp_db)


class TestWebSessionStoreInit:
    """Test store initialization."""

    def test_creates_tables(self, temp_db, test_user_id):
        """Store creates necessary tables on init."""
        store = WebSessionStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check web_sessions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='web_sessions'"
        )
        assert cursor.fetchone() is not None

        # Check web_messages table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='web_messages'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_creates_indexes(self, temp_db, test_user_id):
        """Store creates indexes on init."""
        store = WebSessionStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_web_sessions_user'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_web_sessions_expires'"
        )
        assert cursor.fetchone() is not None

        conn.close()


class TestSessionCreation:
    """Test session creation."""

    def test_create_session(self, web_store, test_user_id):
        """Can create a web session."""
        session_id = web_store.create(test_user_id)

        assert session_id is not None
        assert len(session_id) == 36  # UUID format

    def test_create_session_has_expiration(self, web_store, test_user_id):
        """Created session has valid expiration."""
        session_id = web_store.create(test_user_id)

        session = web_store.get(session_id)

        assert session is not None
        assert session.expires_at is not None
        # Should expire in ~24 hours
        expected_expiry = datetime.now() + timedelta(hours=23)
        assert session.expires_at > expected_expiry

    def test_create_multiple_sessions_for_user(self, web_store, test_user_id):
        """Plan B: User can have multiple active sessions."""
        session_1 = web_store.create(test_user_id)
        session_2 = web_store.create(test_user_id)

        # Both sessions should be different
        assert session_1 != session_2

        # Both should be valid
        assert web_store.get(session_1) is not None
        assert web_store.get(session_2) is not None

    def test_create_session_does_not_expire_others(self, web_store, test_user_id):
        """Creating new session doesn't expire other sessions."""
        session_1 = web_store.create(test_user_id)

        # Create second session
        session_2 = web_store.create(test_user_id)

        # First session should still be valid
        assert web_store.get(session_1) is not None
        assert web_store.get(session_2) is not None


class TestSessionRetrieval:
    """Test session retrieval."""

    def test_get_existing_session(self, web_store, test_user_id):
        """Can retrieve an existing session."""
        session_id = web_store.create(test_user_id)

        session = web_store.get(session_id)

        assert session is not None
        assert session.id == session_id
        assert session.user_id == test_user_id

    def test_get_nonexistent_session(self, web_store):
        """Returns None for nonexistent session."""
        session = web_store.get("nonexistent-session-id")

        assert session is None

    def test_get_expired_session_returns_none(self, web_store, test_user_id, temp_db):
        """Expired session returns None."""
        session_id = web_store.create(test_user_id)

        # Manually expire the session
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE web_sessions SET expires_at = datetime('now', '-1 hour') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()

        session = web_store.get(session_id)

        assert session is None


class TestSessionTTLRefresh:
    """Test TTL refresh functionality."""

    def test_touch_refreshes_ttl(self, web_store, test_user_id, temp_db):
        """touch() refreshes session TTL."""
        session_id = web_store.create(test_user_id)

        # Get initial expires_at
        session_before = web_store.get(session_id)
        initial_expires = session_before.expires_at

        # Small delay
        time.sleep(0.1)

        # Touch the session
        web_store.touch(session_id)

        # Verify TTL was refreshed
        session_after = web_store.get(session_id)
        assert session_after.expires_at >= initial_expires

    def test_update_refreshes_ttl(self, web_store, test_user_id):
        """update() also refreshes TTL."""
        session_id = web_store.create(test_user_id)

        session_before = web_store.get(session_id)
        initial_expires = session_before.expires_at

        time.sleep(0.1)

        web_store.update(session_id, rolling_summary="test summary")

        session_after = web_store.get(session_id)
        assert session_after.expires_at >= initial_expires


class TestSessionUpdate:
    """Test session state updates."""

    def test_update_rolling_summary(self, web_store, test_user_id):
        """Can update rolling summary."""
        session_id = web_store.create(test_user_id)

        web_store.update(session_id, rolling_summary="user wants chicken recipes")

        session = web_store.get(session_id)
        assert session.rolling_summary == "user wants chicken recipes"

    def test_update_last_cards_json(self, web_store, test_user_id):
        """Can update last cards JSON."""
        session_id = web_store.create(test_user_id)

        cards = [{"id": "123", "title": "Chicken"}]
        web_store.update(session_id, last_cards_json=json.dumps(cards))

        session = web_store.get(session_id)
        assert json.loads(session.last_cards_json) == cards

    def test_update_multiple_fields(self, web_store, test_user_id):
        """Can update multiple fields at once."""
        session_id = web_store.create(test_user_id)

        web_store.update(
            session_id,
            rolling_summary="test summary",
            last_cards_json='[{"id": "1"}]',
            chat_session_id="chat-123",
        )

        session = web_store.get(session_id)
        assert session.rolling_summary == "test summary"
        assert session.last_cards_json == '[{"id": "1"}]'
        assert session.chat_session_id == "chat-123"


class TestSessionDeletion:
    """Test session deletion."""

    def test_delete_session(self, web_store, test_user_id):
        """Can delete a session."""
        session_id = web_store.create(test_user_id)

        web_store.delete(session_id)

        assert web_store.get(session_id) is None

    def test_delete_cascades_messages(self, web_store, test_user_id, temp_db):
        """Deleting session cascades to messages."""
        session_id = web_store.create(test_user_id)
        web_store.add_message(session_id, "user", "hello")
        web_store.add_message(session_id, "assistant", "hi there")

        web_store.delete(session_id)

        # Check messages are gone
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM web_messages WHERE web_session_id = ?", (session_id,))
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0


class TestMessageOperations:
    """Test message add/get operations."""

    def test_add_message(self, web_store, test_user_id):
        """Can add a message."""
        session_id = web_store.create(test_user_id)

        web_store.add_message(session_id, "user", "hello world")

        messages = web_store.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hello world"

    def test_add_multiple_messages(self, web_store, test_user_id):
        """Can add multiple messages."""
        session_id = web_store.create(test_user_id)

        web_store.add_message(session_id, "user", "hello")
        web_store.add_message(session_id, "assistant", "hi there")
        web_store.add_message(session_id, "user", "how are you")

        messages = web_store.get_messages(session_id)
        assert len(messages) == 3

    def test_messages_ordered_by_id_ascending(self, web_store, test_user_id):
        """Messages returned oldest first for UI display."""
        session_id = web_store.create(test_user_id)

        web_store.add_message(session_id, "user", "first")
        web_store.add_message(session_id, "assistant", "second")
        web_store.add_message(session_id, "user", "third")

        messages = web_store.get_messages(session_id)

        assert messages[0].content == "first"
        assert messages[1].content == "second"
        assert messages[2].content == "third"

    def test_messages_bounded_to_max(self, web_store, test_user_id):
        """Messages are pruned to MAX_MESSAGES."""
        session_id = web_store.create(test_user_id)

        # Add more than MAX_MESSAGES
        for i in range(web_store.MAX_MESSAGES + 10):
            web_store.add_message(session_id, "user", f"message {i}")

        messages = web_store.get_messages(session_id)

        # Should only have MAX_MESSAGES
        assert len(messages) <= web_store.MAX_MESSAGES

    def test_clear_messages(self, web_store, test_user_id):
        """Can clear all messages."""
        session_id = web_store.create(test_user_id)

        web_store.add_message(session_id, "user", "hello")
        web_store.add_message(session_id, "assistant", "hi")

        web_store.clear_messages(session_id)

        messages = web_store.get_messages(session_id)
        assert len(messages) == 0


class TestAppendExchange:
    """Test append_exchange atomic operation."""

    def test_append_exchange_adds_both_messages(self, web_store, test_user_id):
        """append_exchange adds user and assistant messages."""
        session_id = web_store.create(test_user_id)

        web_store.append_exchange(
            session_id,
            user_text="what can I cook",
            assistant_text="here are some recipes",
        )

        messages = web_store.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "what can I cook"
        assert messages[1].role == "assistant"
        assert messages[1].content == "here are some recipes"

    def test_append_exchange_updates_state(self, web_store, test_user_id):
        """append_exchange updates session state."""
        session_id = web_store.create(test_user_id)

        web_store.append_exchange(
            session_id,
            user_text="hello",
            assistant_text="hi",
            rolling_summary="user said hello",
            last_cards_json='[{"id": "1"}]',
        )

        session = web_store.get(session_id)
        assert session.rolling_summary == "user said hello"
        assert session.last_cards_json == '[{"id": "1"}]'

    def test_append_exchange_refreshes_ttl(self, web_store, test_user_id):
        """append_exchange refreshes TTL."""
        session_id = web_store.create(test_user_id)

        session_before = web_store.get(session_id)
        initial_expires = session_before.expires_at

        time.sleep(0.1)

        web_store.append_exchange(
            session_id,
            user_text="hello",
            assistant_text="hi",
        )

        session_after = web_store.get(session_id)
        assert session_after.expires_at >= initial_expires


class TestGetLastCards:
    """Test get_last_cards helper."""

    def test_get_last_cards_empty(self, web_store, test_user_id):
        """Returns empty list when no cards."""
        session_id = web_store.create(test_user_id)

        cards = web_store.get_last_cards(session_id)

        assert cards == []

    def test_get_last_cards_with_data(self, web_store, test_user_id):
        """Returns parsed cards when present."""
        session_id = web_store.create(test_user_id)

        cards_data = [{"id": "123", "title": "Recipe"}]
        web_store.update(session_id, last_cards_json=json.dumps(cards_data))

        cards = web_store.get_last_cards(session_id)

        assert cards == cards_data

    def test_get_last_cards_invalid_json(self, web_store, test_user_id, temp_db):
        """Returns empty list on invalid JSON."""
        session_id = web_store.create(test_user_id)

        # Set invalid JSON directly
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE web_sessions SET last_cards_json = 'invalid{json' WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()

        cards = web_store.get_last_cards(session_id)

        assert cards == []


class TestCleanupExpired:
    """Test expired session cleanup."""

    def test_cleanup_removes_expired(self, web_store, test_user_id, temp_db):
        """cleanup_expired removes expired sessions."""
        session_id = web_store.create(test_user_id)

        # Manually expire
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE web_sessions SET expires_at = datetime('now', '-1 hour') WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()

        count = web_store.cleanup_expired()

        assert count == 1
        assert web_store.get(session_id) is None

    def test_cleanup_keeps_valid(self, web_store, test_user_id):
        """cleanup_expired keeps valid sessions."""
        session_id = web_store.create(test_user_id)

        count = web_store.cleanup_expired()

        assert count == 0
        assert web_store.get(session_id) is not None


class TestGetSessionsForUser:
    """Test getting all sessions for a user."""

    def test_get_sessions_for_user(self, web_store, test_user_id):
        """Can get all sessions for a user."""
        session_1 = web_store.create(test_user_id)
        session_2 = web_store.create(test_user_id)

        sessions = web_store.get_sessions_for_user(test_user_id)

        assert len(sessions) == 2
        session_ids = [s.id for s in sessions]
        assert session_1 in session_ids
        assert session_2 in session_ids

    def test_get_sessions_excludes_expired(self, web_store, test_user_id, temp_db):
        """Expired sessions not included."""
        valid_session = web_store.create(test_user_id)
        expired_session = web_store.create(test_user_id)

        # Expire one session
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE web_sessions SET expires_at = datetime('now', '-1 hour') WHERE id = ?",
            (expired_session,),
        )
        conn.commit()
        conn.close()

        sessions = web_store.get_sessions_for_user(test_user_id)

        assert len(sessions) == 1
        assert sessions[0].id == valid_session

    def test_get_sessions_ordered_by_updated_at(self, web_store, test_user_id):
        """Sessions ordered by most recently updated first."""
        session_1 = web_store.create(test_user_id)
        time.sleep(0.1)
        session_2 = web_store.create(test_user_id)

        sessions = web_store.get_sessions_for_user(test_user_id)

        # Most recently created (session_2) should be first
        assert sessions[0].id == session_2
        assert sessions[1].id == session_1
