"""User context management for CLI and future web support."""

from dataclasses import dataclass, field
from typing import Callable

from src.app.logging_config import get_logger

logger = get_logger(__name__)


class UserRegistry:
    """Registry of allowed users. Centralizes user validation logic."""

    ALLOWED_USERS: list[str] = ["alex", "caitlyn", "family", "guest", "test"]
    DEFAULT_USER: str = "guest"

    @classmethod
    def is_valid(cls, username: str) -> bool:
        """Check if username is in allowed list."""
        return username.strip().lower() in cls.ALLOWED_USERS

    @classmethod
    def normalize(cls, username: str) -> str | None:
        """Normalize and validate username. Returns None if invalid."""
        normalized = username.strip().lower()
        return normalized if normalized in cls.ALLOWED_USERS else None

    @classmethod
    def get_all(cls) -> list[str]:
        """Get all allowed usernames."""
        return cls.ALLOWED_USERS.copy()


@dataclass
class UserContext:
    """Encapsulates current user state and login/logout logic.

    There is always a logged-in user (default: guest). This prevents
    null-user state bugs and simplifies command handling.
    """

    current_user: str = field(default=UserRegistry.DEFAULT_USER)
    session_id: str | None = None

    # Callbacks for state reset (set by CLI during initialization)
    _on_user_change: Callable[[str], None] | None = field(default=None, repr=False)

    def login(self, username: str) -> tuple[bool, str]:
        """Login as a user. Returns (success, message)."""
        validated = UserRegistry.normalize(username)
        if validated is None:
            users = ", ".join(UserRegistry.get_all())
            return False, f"Unknown user: {username}. Available: {users}"

        # Check for redundant login (already logged in as this user)
        if validated == self.current_user:
            return True, f"Already logged in as {validated}"

        old_user = self.current_user
        self.current_user = validated

        # Trigger state reset callback
        if self._on_user_change:
            self._on_user_change(validated)

        logger.info("User logged in", user=validated, previous_user=old_user)
        return True, f"Logged in as: {validated}"

    def logout(self) -> str:
        """Logout and switch back to guest. Returns message."""
        if self.current_user == UserRegistry.DEFAULT_USER:
            return "Already logged in as guest"

        old_user = self.current_user
        self.current_user = UserRegistry.DEFAULT_USER

        # Trigger state reset callback
        if self._on_user_change:
            self._on_user_change(UserRegistry.DEFAULT_USER)

        logger.info("User logged out", user=old_user, new_user=UserRegistry.DEFAULT_USER)
        return f"Logged out from: {old_user}. Now logged in as: guest"

    def whoami(self) -> str:
        """Get current user display string."""
        return f"Logged in as: {self.current_user}"

    def set_on_user_change(self, callback: Callable[[str], None]) -> None:
        """Set callback for user change events (for state reset)."""
        self._on_user_change = callback
