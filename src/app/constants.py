"""Application constants for multi-user support."""

# Fixed UUID for CLI backward compatibility (never changes)
# This UUID is used by the CLI to maintain single-user mode
# and ensure all existing data is associated with this default user
DEFAULT_USER_UUID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_USERNAME = "default_user"

# Schema version for database migrations
CURRENT_SCHEMA_VERSION = 2  # Multi-user schema

# Predefined user accounts (UUIDs generated at migration time, not hardcoded)
# Usernames are case-insensitive (COLLATE NOCASE in SQLite)
PREDEFINED_USERNAMES = [
    "Alex",
    "Caitlyn",
    "Family",
    "Guest",
    "Test",
]

# Session configuration
WEB_SESSION_TTL_HOURS = 24
MAX_MESSAGES_PER_SESSION = 50
