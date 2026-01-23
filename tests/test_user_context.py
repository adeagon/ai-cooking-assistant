"""Tests for user context and registry."""

import pytest
from src.app.user_context import UserContext, UserRegistry


class TestUserRegistry:
    """Tests for UserRegistry."""

    def test_allowed_users_not_empty(self):
        """Allowed users list should not be empty."""
        assert len(UserRegistry.ALLOWED_USERS) > 0

    def test_allowed_users_lowercase(self):
        """All allowed usernames should be lowercase."""
        for user in UserRegistry.ALLOWED_USERS:
            assert user == user.lower()

    def test_guest_in_allowed(self):
        """Guest user should always be in allowed list."""
        assert "guest" in UserRegistry.ALLOWED_USERS

    def test_default_user_is_guest(self):
        """Default user should be guest."""
        assert UserRegistry.DEFAULT_USER == "guest"

    def test_normalize_valid(self):
        """Valid usernames should be normalized to lowercase."""
        assert UserRegistry.normalize("alex") == "alex"
        assert UserRegistry.normalize("Alex") == "alex"
        assert UserRegistry.normalize("  ALEX  ") == "alex"

    def test_normalize_invalid(self):
        """Invalid usernames should return None."""
        assert UserRegistry.normalize("unknown") is None
        assert UserRegistry.normalize("") is None
        assert UserRegistry.normalize("   ") is None

    def test_is_valid(self):
        """is_valid should correctly identify valid/invalid users."""
        assert UserRegistry.is_valid("guest") is True
        assert UserRegistry.is_valid("alex") is True
        assert UserRegistry.is_valid("ALEX") is True
        assert UserRegistry.is_valid("unknown") is False
        assert UserRegistry.is_valid("") is False

    def test_get_all_returns_copy(self):
        """get_all should return a copy, not the original list."""
        users = UserRegistry.get_all()
        original_length = len(UserRegistry.ALLOWED_USERS)
        users.append("hacker")
        assert len(UserRegistry.ALLOWED_USERS) == original_length


class TestUserContext:
    """Tests for UserContext."""

    def test_default_user_is_guest(self):
        """New context should default to guest."""
        ctx = UserContext()
        assert ctx.current_user == "guest"

    def test_custom_initial_user(self):
        """Context can be initialized with a specific user."""
        ctx = UserContext(current_user="alex")
        assert ctx.current_user == "alex"

    def test_login_valid_user(self):
        """Login with valid user should succeed."""
        ctx = UserContext()
        success, msg = ctx.login("alex")
        assert success is True
        assert ctx.current_user == "alex"
        assert "alex" in msg

    def test_login_case_insensitive(self):
        """Login should be case-insensitive."""
        ctx = UserContext()
        success, msg = ctx.login("ALEX")
        assert success is True
        assert ctx.current_user == "alex"

    def test_login_with_whitespace(self):
        """Login should handle whitespace in username."""
        ctx = UserContext()
        success, msg = ctx.login("  alex  ")
        assert success is True
        assert ctx.current_user == "alex"

    def test_login_invalid_user(self):
        """Login with invalid user should fail and preserve state."""
        ctx = UserContext()
        success, msg = ctx.login("unknown")
        assert success is False
        assert ctx.current_user == "guest"  # unchanged
        assert "Unknown user" in msg

    def test_logout_returns_to_guest(self):
        """Logout should return to guest."""
        ctx = UserContext(current_user="alex")
        msg = ctx.logout()
        assert ctx.current_user == "guest"
        assert "guest" in msg
        assert "alex" in msg

    def test_logout_when_guest(self):
        """Logout when already guest should indicate so."""
        ctx = UserContext()
        msg = ctx.logout()
        assert "Already" in msg
        assert ctx.current_user == "guest"

    def test_whoami(self):
        """whoami should return current user info."""
        ctx = UserContext(current_user="alex")
        result = ctx.whoami()
        assert "alex" in result
        assert "Logged in as" in result

    def test_on_user_change_callback_on_login(self):
        """Login should trigger user change callback."""
        callback_users = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: callback_users.append(u))

        ctx.login("alex")
        assert callback_users == ["alex"]

    def test_on_user_change_callback_on_logout(self):
        """Logout should trigger user change callback."""
        callback_users = []
        ctx = UserContext(current_user="alex")
        ctx.set_on_user_change(lambda u: callback_users.append(u))

        ctx.logout()
        assert callback_users == ["guest"]

    def test_no_callback_when_none_set(self):
        """Operations should work without callback set."""
        ctx = UserContext()
        # Should not raise even without callback
        ctx.login("alex")
        ctx.logout()
        assert ctx.current_user == "guest"

    def test_callback_not_called_on_failed_login(self):
        """Failed login should not trigger callback."""
        callback_users = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: callback_users.append(u))

        ctx.login("invalid_user")
        assert callback_users == []

    def test_callback_not_called_on_logout_when_guest(self):
        """Logout when already guest should not trigger callback."""
        callback_users = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: callback_users.append(u))

        ctx.logout()
        assert callback_users == []  # No change, no callback
