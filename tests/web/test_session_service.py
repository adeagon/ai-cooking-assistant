"""Tests for SessionService."""

import time
from datetime import datetime, timedelta

import pytest

from src.web.db import get_db_connection
from src.web.services.session_service import SessionService
from src.web.services.user_service import UserService


class TestSessionService:
    """Tests for SessionService."""

    def test_create_session(self, session_service: SessionService, test_user):
        """Test creating a new session."""
        session = session_service.create(
            user_id=test_user.id,
            user_agent="Mozilla/5.0",
            ip="192.168.1.1"
        )

        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID
        assert session.user_id == test_user.id
        assert session.user_agent == "Mozilla/5.0"
        assert session.ip == "192.168.1.1"
        assert session.revoked_at is None

    def test_create_session_without_optional_fields(
        self, session_service: SessionService, test_user
    ):
        """Test creating session without user agent and IP."""
        session = session_service.create(user_id=test_user.id)

        assert session.session_id is not None
        assert session.user_agent is None
        assert session.ip is None

    def test_get_valid_session(self, session_service: SessionService, test_user):
        """Test getting a valid session."""
        created = session_service.create(user_id=test_user.id)
        fetched = session_service.get_valid(created.session_id)

        assert fetched is not None
        assert fetched.session_id == created.session_id
        assert fetched.user_id == test_user.id

    def test_get_valid_session_not_found(self, session_service: SessionService):
        """Test getting non-existent session returns None."""
        result = session_service.get_valid("nonexistent-session-id")
        assert result is None

    def test_get_valid_session_revoked(
        self, session_service: SessionService, test_user
    ):
        """Test that revoked sessions are not returned as valid."""
        session = session_service.create(user_id=test_user.id)
        session_service.revoke(session.session_id)

        result = session_service.get_valid(session.session_id)
        assert result is None

    def test_touch_session(self, session_service: SessionService, test_user):
        """Test updating session's last_seen_at."""
        session = session_service.create(user_id=test_user.id)
        original_last_seen = session.last_seen_at

        # Small delay to ensure time difference
        time.sleep(0.01)

        result = session_service.touch(session.session_id)
        assert result is True

        updated = session_service.get_valid(session.session_id)
        assert updated.last_seen_at >= original_last_seen

    def test_touch_session_not_found(self, session_service: SessionService):
        """Test touching non-existent session returns False."""
        result = session_service.touch("nonexistent-session-id")
        assert result is False

    def test_touch_revoked_session(self, session_service: SessionService, test_user):
        """Test that touching a revoked session returns False."""
        session = session_service.create(user_id=test_user.id)
        session_service.revoke(session.session_id)

        result = session_service.touch(session.session_id)
        assert result is False

    def test_revoke_session(self, session_service: SessionService, test_user):
        """Test revoking a session."""
        session = session_service.create(user_id=test_user.id)
        result = session_service.revoke(session.session_id)

        assert result is True

        # Session should no longer be valid
        fetched = session_service.get_valid(session.session_id)
        assert fetched is None

    def test_revoke_session_not_found(self, session_service: SessionService):
        """Test revoking non-existent session returns False."""
        result = session_service.revoke("nonexistent-session-id")
        assert result is False

    def test_revoke_already_revoked(self, session_service: SessionService, test_user):
        """Test revoking an already revoked session returns False."""
        session = session_service.create(user_id=test_user.id)
        session_service.revoke(session.session_id)

        result = session_service.revoke(session.session_id)
        assert result is False

    def test_revoke_all_for_user(self, session_service: SessionService, test_user):
        """Test revoking all sessions for a user."""
        # Create multiple sessions
        s1 = session_service.create(user_id=test_user.id)
        s2 = session_service.create(user_id=test_user.id)
        s3 = session_service.create(user_id=test_user.id)

        count = session_service.revoke_all_for_user(test_user.id)

        assert count == 3

        # All sessions should be invalid
        assert session_service.get_valid(s1.session_id) is None
        assert session_service.get_valid(s2.session_id) is None
        assert session_service.get_valid(s3.session_id) is None

    def test_get_active_sessions_for_user(
        self, session_service: SessionService, test_user
    ):
        """Test getting all active sessions for a user."""
        s1 = session_service.create(user_id=test_user.id)
        s2 = session_service.create(user_id=test_user.id)
        s3 = session_service.create(user_id=test_user.id)

        # Revoke one
        session_service.revoke(s2.session_id)

        active = session_service.get_active_sessions_for_user(test_user.id)

        assert len(active) == 2
        session_ids = [s.session_id for s in active]
        assert s1.session_id in session_ids
        assert s3.session_id in session_ids
        assert s2.session_id not in session_ids

    def test_cleanup_expired_sessions(self, temp_db, test_user):
        """Test cleanup of expired sessions."""
        # Create session service with 1 day max age
        session_service = SessionService(temp_db, max_age_days=1)

        session = session_service.create(user_id=test_user.id)

        # Manually set last_seen_at to 2 days ago
        with get_db_connection(temp_db) as conn:
            old_time = (datetime.now() - timedelta(days=2)).isoformat()
            conn.execute(
                "UPDATE web_sessions SET last_seen_at = ? WHERE session_id = ?",
                (old_time, session.session_id)
            )
            conn.commit()

        # Cleanup should remove the expired session
        count = session_service.cleanup_expired()

        assert count == 1
        assert session_service.get_valid(session.session_id) is None

    def test_multiple_users_sessions_isolated(
        self, session_service: SessionService, user_service: UserService
    ):
        """Test that sessions from different users are isolated."""
        user1 = user_service.create(username="user1")
        user2 = user_service.create(username="user2")

        s1 = session_service.create(user_id=user1.id)
        s2 = session_service.create(user_id=user2.id)

        # Revoke all for user1
        session_service.revoke_all_for_user(user1.id)

        # User2's session should still be valid
        assert session_service.get_valid(s1.session_id) is None
        assert session_service.get_valid(s2.session_id) is not None
