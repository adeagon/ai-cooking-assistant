"""Base class for user-bound stores."""

import sqlite3
from pathlib import Path

from src.app.logging_config import get_logger

logger = get_logger(__name__)


class BaseUserBoundStore:
    """Abstract base class for user-scoped stores.

    All stores bind to a specific username at instantiation.
    The user cannot be changed after initialization.
    """

    def __init__(self, db_path: Path, username: str = "guest"):
        """Initialize store bound to a specific user.

        Args:
            db_path: Path to SQLite database file
            username: Username this store is bound to (default: "guest")
        """
        self.db_path = db_path
        self._user = username
        self._ensure_table()

    @property
    def user(self) -> str:
        """Read-only access to bound username.

        Returns:
            The username this store is bound to.

        Note:
            Raises AttributeError if assignment is attempted.
        """
        return self._user

    def _ensure_table(self) -> None:
        """Create table if not exists. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _ensure_table")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        Returns:
            A new SQLite connection to the database.
        """
        return sqlite3.connect(self.db_path)
