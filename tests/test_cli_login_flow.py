"""Golden tests for CLI login flow."""

import pytest
from src.app.user_context import UserContext, UserRegistry


class TestLoginFlowGolden:
    """End-to-end golden tests for login flow."""

    def test_full_login_cycle(self):
        """Golden test: guest -> login alex -> verify -> logout -> verify guest."""
        # Track state changes
        state_changes = []

        ctx = UserContext()
        ctx.set_on_user_change(lambda u: state_changes.append(u))

        # Start as guest
        assert ctx.current_user == "guest"
        assert ctx.whoami() == "Logged in as: guest"

        # Login as alex
        success, msg = ctx.login("alex")
        assert success is True
        assert ctx.current_user == "alex"
        assert "alex" in ctx.whoami()

        # Logout returns to guest
        msg = ctx.logout()
        assert ctx.current_user == "guest"
        assert "guest" in ctx.whoami()

        # Verify callback sequence
        assert state_changes == ["alex", "guest"]

    def test_user_switch_without_logout(self):
        """Golden test: login alex -> login caitlyn (implicit switch)."""
        state_changes = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: state_changes.append(u))

        ctx.login("alex")
        assert ctx.current_user == "alex"

        # Direct switch without logout
        ctx.login("caitlyn")
        assert ctx.current_user == "caitlyn"

        # Both changes should have triggered callback
        assert state_changes == ["alex", "caitlyn"]

    def test_invalid_login_preserves_state(self):
        """Golden test: login alex -> attempt invalid -> still alex."""
        ctx = UserContext()
        ctx.login("alex")

        success, _ = ctx.login("invalid_user")
        assert success is False
        assert ctx.current_user == "alex"  # unchanged

    def test_case_insensitive_login(self):
        """Golden test: login ALEX works."""
        ctx = UserContext()
        success, _ = ctx.login("ALEX")
        assert success is True
        assert ctx.current_user == "alex"

    def test_all_allowed_users_can_login(self):
        """Golden test: all registered users can login."""
        for username in UserRegistry.get_all():
            ctx = UserContext()
            success, msg = ctx.login(username)
            assert success is True, f"Failed to login as {username}: {msg}"
            assert ctx.current_user == username

    def test_multiple_logout_attempts(self):
        """Golden test: multiple logout when guest doesn't cause errors."""
        ctx = UserContext()

        # First logout (already guest)
        msg1 = ctx.logout()
        assert "Already" in msg1

        # Second logout (still guest)
        msg2 = ctx.logout()
        assert "Already" in msg2

        assert ctx.current_user == "guest"

    def test_login_logout_login_cycle(self):
        """Golden test: complex login/logout/login cycle."""
        state_changes = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: state_changes.append(u))

        # Login as alex
        ctx.login("alex")
        assert ctx.current_user == "alex"

        # Logout to guest
        ctx.logout()
        assert ctx.current_user == "guest"

        # Login as caitlyn
        ctx.login("caitlyn")
        assert ctx.current_user == "caitlyn"

        # Login as family (without logout)
        ctx.login("family")
        assert ctx.current_user == "family"

        # Final logout
        ctx.logout()
        assert ctx.current_user == "guest"

        # Verify full sequence
        assert state_changes == ["alex", "guest", "caitlyn", "family", "guest"]

    def test_state_reset_callback_receives_correct_user(self):
        """Golden test: callback receives the new user correctly."""
        received_users = []

        def track_user(user: str):
            received_users.append(user)

        ctx = UserContext()
        ctx.set_on_user_change(track_user)

        ctx.login("test")
        assert received_users[-1] == "test"

        ctx.login("alex")
        assert received_users[-1] == "alex"

        ctx.logout()
        assert received_users[-1] == "guest"

    def test_whoami_format_consistency(self):
        """Golden test: whoami format is consistent."""
        ctx = UserContext()

        # Check format for guest
        assert ctx.whoami() == "Logged in as: guest"

        # Check format for other users
        ctx.login("alex")
        assert ctx.whoami() == "Logged in as: alex"

        ctx.login("caitlyn")
        assert ctx.whoami() == "Logged in as: caitlyn"

    def test_login_message_format(self):
        """Golden test: login success message format."""
        ctx = UserContext()
        success, msg = ctx.login("alex")

        assert success is True
        assert msg == "Logged in as: alex"

    def test_logout_message_format(self):
        """Golden test: logout message format."""
        ctx = UserContext(current_user="alex")
        msg = ctx.logout()

        assert "Logged out from: alex" in msg
        assert "guest" in msg

    def test_redundant_login_no_op(self):
        """Golden test: login as same user returns already-logged-in message."""
        ctx = UserContext()
        ctx.login("alex")

        # Login again as alex (redundant)
        success, msg = ctx.login("alex")
        assert success is True
        assert "Already logged in as alex" in msg
        assert ctx.current_user == "alex"

    def test_redundant_login_callback_not_triggered(self):
        """Golden test: redundant login doesn't trigger callback."""
        callback_calls = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: callback_calls.append(u))

        # First login triggers callback
        ctx.login("alex")
        assert callback_calls == ["alex"]

        # Redundant login should NOT trigger callback
        ctx.login("alex")
        assert callback_calls == ["alex"]  # Still only one call

    def test_redundant_login_guest(self):
        """Golden test: login as guest when already guest."""
        ctx = UserContext()
        assert ctx.current_user == "guest"

        success, msg = ctx.login("guest")
        assert success is True
        assert "Already logged in as guest" in msg

    def test_redundant_login_after_switch(self):
        """Golden test: redundant login after user switch."""
        callback_calls = []
        ctx = UserContext()
        ctx.set_on_user_change(lambda u: callback_calls.append(u))

        # Login as alex
        ctx.login("alex")
        assert callback_calls == ["alex"]

        # Switch to caitlyn
        ctx.login("caitlyn")
        assert callback_calls == ["alex", "caitlyn"]

        # Redundant login as caitlyn (should NOT trigger)
        success, msg = ctx.login("caitlyn")
        assert success is True
        assert "Already logged in" in msg
        assert callback_calls == ["alex", "caitlyn"]  # No additional call
