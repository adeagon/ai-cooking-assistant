"""Database connection and schema management for web application."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.app.logging_config import get_logger

logger = get_logger(__name__)


# Schema definition
SCHEMA_SQL = """
-- Enable foreign keys (must be run per connection via PRAGMA)

-- Users table (UUID id, username unique)
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Browser sessions (with revoke semantics)
CREATE TABLE IF NOT EXISTS web_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP,
    user_agent TEXT,
    ip TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Conversations (chat threads per user)
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP,
    archived_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Messages (with flexible meta_json)
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_last
    ON conversations(user_id, last_message_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
"""

# Default seed users
DEFAULT_USERS = [
    ("alex", "Alex"),
    ("jordan", "Jordan"),
    ("taylor", "Taylor"),
    ("casey", "Casey"),
]


@contextmanager
def get_db_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with proper settings.

    Enables foreign keys, WAL mode, and sets row factory.

    Args:
        db_path: Path to the SQLite database file

    Yields:
        Configured SQLite connection
    """
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path, seed_users: bool = True) -> None:
    """Initialize the database schema and optionally seed users.

    Args:
        db_path: Path to the SQLite database file
        seed_users: Whether to create default users if none exist
    """
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection(db_path) as conn:
        # Create tables
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database schema initialized", db_path=str(db_path))

        # Seed default users if requested and none exist
        if seed_users:
            existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if existing == 0:
                import uuid
                for username, display_name in DEFAULT_USERS:
                    user_id = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO users (id, username, display_name) VALUES (?, ?, ?)",
                        (user_id, username, display_name)
                    )
                conn.commit()
                logger.info("Seeded default users", count=len(DEFAULT_USERS))


def cleanup_expired_sessions(db_path: Path, max_age_days: int = 30) -> int:
    """Remove sessions older than max_age_days.

    Args:
        db_path: Path to the SQLite database file
        max_age_days: Maximum age in days for sessions

    Returns:
        Number of sessions cleaned up
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM web_sessions
            WHERE last_seen_at < datetime('now', ? || ' days')
            """,
            (f"-{max_age_days}",)
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info("Cleaned up expired sessions", count=count)
        return count
