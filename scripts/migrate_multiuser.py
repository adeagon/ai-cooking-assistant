#!/usr/bin/env python3
"""Database migration script for multi-user support.

This script migrates the database from single-user (v1) to multi-user (v2) schema.
It's idempotent - running it multiple times is safe.

Usage:
    python scripts/migrate_multiuser.py [--dry-run] [--no-backup]

The migration:
1. Creates schema_version table for version tracking
2. Creates users table with predefined accounts
3. Adds user_id columns to all relevant tables
4. Migrates existing data to default_user UUID
5. Creates web_sessions and web_messages tables for web UI
"""

import argparse
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app.constants import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_USER_USERNAME,
    DEFAULT_USER_UUID,
    PREDEFINED_USERNAMES,
)

# Target schema version for this migration
TARGET_VERSION = 2


def get_db_path() -> Path:
    """Get database path from settings."""
    from src.app.settings import settings
    return settings.sqlite_db_path


def backup_database(db_path: Path) -> Path:
    """Create a timestamped backup of the database.

    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    print(f"  Created backup: {backup_path}")
    return backup_path


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version (0 if no version table)."""
    cursor = conn.cursor()

    # Check if schema_version table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_version'
    """)
    if not cursor.fetchone():
        return 0

    # Get max version
    cursor.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    return cursor.fetchone()[0]


def acquire_migration_lock(conn: sqlite3.Connection) -> bool:
    """Prevent two migrations from running simultaneously.

    Uses BEGIN IMMEDIATE to acquire write lock early.

    Returns:
        True if lock acquired, False if another migration is running
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
        return True
    except sqlite3.OperationalError:
        return False


def create_schema_version_table(conn: sqlite3.Connection) -> None:
    """Create schema_version table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    print("  Created schema_version table")


def create_users_table(conn: sqlite3.Connection) -> None:
    """Create users table with predefined accounts."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

    # Insert default_user with fixed UUID (for CLI backward compatibility)
    conn.execute("""
        INSERT OR IGNORE INTO users (id, username, password_hash, is_active)
        VALUES (?, ?, NULL, 1)
    """, (DEFAULT_USER_UUID, DEFAULT_USER_USERNAME))

    # Insert predefined users with generated UUIDs
    for username in PREDEFINED_USERNAMES:
        user_uuid = str(uuid.uuid4())
        conn.execute("""
            INSERT OR IGNORE INTO users (id, username, password_hash, is_active)
            VALUES (?, ?, NULL, 1)
        """, (user_uuid, username))

    print("  Created users table with predefined accounts")


def create_web_sessions_table(conn: sqlite3.Connection) -> None:
    """Create web_sessions table for server-side session state."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            chat_session_id TEXT,
            rolling_summary TEXT,
            last_cards_json TEXT,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_sessions_expires ON web_sessions(expires_at)")

    print("  Created web_sessions table")


def create_web_messages_table(conn: sqlite3.Connection) -> None:
    """Create web_messages table for conversation history."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            web_session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (web_session_id) REFERENCES web_sessions(id) ON DELETE CASCADE
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_messages_session_id ON web_messages(web_session_id, id)")

    print("  Created web_messages table")


def migrate_preferences_table(conn: sqlite3.Connection) -> None:
    """Migrate preferences table: change PK from id=1 to user_id."""
    cursor = conn.cursor()

    # Check if migration needed (does user_id column exist?)
    cursor.execute("PRAGMA table_info(preferences)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" in columns:
        print("  preferences table already migrated")
        return

    # Check if preferences table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='preferences'
    """)
    if not cursor.fetchone():
        # Create new table with user_id
        cursor.execute("""
            CREATE TABLE preferences (
                user_id TEXT PRIMARY KEY,
                spice_level TEXT DEFAULT 'medium',
                diet TEXT DEFAULT 'none',
                avoid_ingredients TEXT,
                preferred_cuisines TEXT,
                time_limit_default INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        print("  Created preferences table with user_id")
        return

    # Table rebuild pattern for preferences
    # 1. Create new table with user_id as PK
    cursor.execute("""
        CREATE TABLE preferences_new (
            user_id TEXT PRIMARY KEY,
            spice_level TEXT DEFAULT 'medium',
            diet TEXT DEFAULT 'none',
            avoid_ingredients TEXT,
            preferred_cuisines TEXT,
            time_limit_default INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # 2. Migrate existing row to default_user
    cursor.execute("""
        INSERT INTO preferences_new (
            user_id, spice_level, diet, avoid_ingredients,
            preferred_cuisines, time_limit_default, created_at, updated_at
        )
        SELECT
            ?, spice_level, diet, avoid_ingredients,
            preferred_cuisines, time_limit_default, created_at, updated_at
        FROM preferences
        WHERE id = 1
    """, (DEFAULT_USER_UUID,))

    # 3. Drop old table and rename
    cursor.execute("DROP TABLE preferences")
    cursor.execute("ALTER TABLE preferences_new RENAME TO preferences")

    print("  Migrated preferences table to use user_id")


def migrate_sessions_table(conn: sqlite3.Connection) -> None:
    """Add user_id column to sessions table."""
    cursor = conn.cursor()

    # Check if migration needed
    cursor.execute("PRAGMA table_info(sessions)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" in columns:
        print("  sessions table already migrated")
        return

    # Check if sessions table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='sessions'
    """)
    if not cursor.fetchone():
        # Create new table with user_id
        cursor.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                ingredients_on_hand TEXT,
                avoid_tonight TEXT,
                goals TEXT,
                time_limit INTEGER,
                servings INTEGER,
                rolling_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        print("  Created sessions table with user_id")
        return

    # Add user_id column
    cursor.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")

    # Set all existing sessions to default_user
    cursor.execute("UPDATE sessions SET user_id = ?", (DEFAULT_USER_UUID,))

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

    print("  Migrated sessions table to use user_id")


def migrate_recipe_feedback_table(conn: sqlite3.Connection) -> None:
    """Add user_id column to recipe_feedback table."""
    cursor = conn.cursor()

    # Check if migration needed
    cursor.execute("PRAGMA table_info(recipe_feedback)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" in columns:
        print("  recipe_feedback table already migrated")
        return

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='recipe_feedback'
    """)
    if not cursor.fetchone():
        # Create new table with user_id
        cursor.execute("""
            CREATE TABLE recipe_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_recipe ON recipe_feedback(user_id, recipe_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON recipe_feedback(feedback_type)")
        print("  Created recipe_feedback table with user_id")
        return

    # Add user_id column
    cursor.execute("ALTER TABLE recipe_feedback ADD COLUMN user_id TEXT")

    # Set all existing feedback to default_user
    cursor.execute("UPDATE recipe_feedback SET user_id = ?", (DEFAULT_USER_UUID,))

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_recipe ON recipe_feedback(user_id, recipe_id)")

    print("  Migrated recipe_feedback table to use user_id")


def migrate_cooking_history_table(conn: sqlite3.Connection) -> None:
    """Add user_id column to cooking_history table."""
    cursor = conn.cursor()

    # Check if migration needed
    cursor.execute("PRAGMA table_info(cooking_history)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" in columns:
        print("  cooking_history table already migrated")
        return

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='cooking_history'
    """)
    if not cursor.fetchone():
        # Create new table with user_id
        cursor.execute("""
            CREATE TABLE cooking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                cooked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_date ON cooking_history(user_id, cooked_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_recipe ON cooking_history(recipe_id)")
        print("  Created cooking_history table with user_id")
        return

    # Add user_id column
    cursor.execute("ALTER TABLE cooking_history ADD COLUMN user_id TEXT")

    # Set all existing history to default_user
    cursor.execute("UPDATE cooking_history SET user_id = ?", (DEFAULT_USER_UUID,))

    # Create index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_date ON cooking_history(user_id, cooked_at)")

    print("  Migrated cooking_history table to use user_id")


def migrate_saved_recipes_table(conn: sqlite3.Connection) -> None:
    """Migrate saved_recipes table: add user_id and change UNIQUE constraint."""
    cursor = conn.cursor()

    # Check if migration needed
    cursor.execute("PRAGMA table_info(saved_recipes)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" in columns:
        print("  saved_recipes table already migrated")
        return

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='saved_recipes'
    """)
    if not cursor.fetchone():
        # Create new table with user_id
        cursor.execute("""
            CREATE TABLE saved_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                title TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                UNIQUE(user_id, recipe_id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_user ON saved_recipes(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_recipe ON saved_recipes(recipe_id)")
        print("  Created saved_recipes table with user_id")
        return

    # Table rebuild pattern for saved_recipes (UNIQUE constraint change)
    # 1. Create new table with user_id and UNIQUE(user_id, recipe_id)
    cursor.execute("""
        CREATE TABLE saved_recipes_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            title TEXT NOT NULL,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE(user_id, recipe_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
        )
    """)

    # 2. Migrate existing data to default_user
    cursor.execute("""
        INSERT INTO saved_recipes_new (id, user_id, recipe_id, title, saved_at, notes)
        SELECT id, ?, recipe_id, title, saved_at, notes
        FROM saved_recipes
    """, (DEFAULT_USER_UUID,))

    # 3. Drop old table and rename
    cursor.execute("DROP TABLE saved_recipes")
    cursor.execute("ALTER TABLE saved_recipes_new RENAME TO saved_recipes")

    # 4. Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_user ON saved_recipes(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_recipe ON saved_recipes(recipe_id)")

    print("  Migrated saved_recipes table to use user_id")


def migrate_meal_plans_table(conn: sqlite3.Connection) -> None:
    """Ensure meal_plans table has user_id and update existing data."""
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='meal_plans'
    """)
    if not cursor.fetchone():
        print("  meal_plans table doesn't exist (will be created when needed)")
        return

    # Check if user_id column exists
    cursor.execute("PRAGMA table_info(meal_plans)")
    columns = {row[1] for row in cursor.fetchall()}

    if "user_id" not in columns:
        cursor.execute("ALTER TABLE meal_plans ADD COLUMN user_id TEXT")

    # Update any NULL user_ids to default_user
    cursor.execute("""
        UPDATE meal_plans
        SET user_id = ?
        WHERE user_id IS NULL OR user_id = ''
    """, (DEFAULT_USER_UUID,))

    # Ensure index exists
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_meal_plans_user ON meal_plans(user_id)")

    print("  Migrated meal_plans table to use user_id")


def enable_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable WAL mode for better concurrent access."""
    # WAL mode is database-level, not per-connection
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    result = cursor.fetchone()

    if result and result[0].lower() == "wal":
        print("  WAL mode enabled")
    else:
        print(f"  Warning: Could not enable WAL mode, got: {result}")


def validate_foreign_keys(conn: sqlite3.Connection) -> list[str]:
    """Validate all foreign key constraints.

    Returns:
        List of validation errors (empty if all OK)
    """
    errors = []
    cursor = conn.cursor()

    # Enable FK checking
    cursor.execute("PRAGMA foreign_keys=ON")

    # Check for FK violations
    cursor.execute("PRAGMA foreign_key_check")
    violations = cursor.fetchall()

    for violation in violations:
        table, rowid, parent, fkid = violation
        errors.append(f"FK violation in {table} row {rowid} -> {parent}")

    return errors


def run_migration(dry_run: bool = False, no_backup: bool = False) -> bool:
    """Run the database migration.

    Args:
        dry_run: If True, show what would be done without making changes
        no_backup: If True, skip creating a backup

    Returns:
        True if migration succeeded, False otherwise
    """
    db_path = get_db_path()

    print(f"\n{'=' * 60}")
    print("Multi-User Database Migration")
    print(f"{'=' * 60}")
    print(f"\nDatabase: {db_path}")
    print(f"Target version: {TARGET_VERSION}")

    if not db_path.exists():
        print(f"\nDatabase does not exist at {db_path}")
        print("Run 'python -m src.app.cli ingest process' first to create the database")
        return False

    # Create backup
    if not no_backup and not dry_run:
        print("\nStep 1: Creating backup...")
        backup_path = backup_database(db_path)
    else:
        print("\nStep 1: Skipping backup (--no-backup or --dry-run)")

    # Connect to database
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=5000")

    try:
        # Check current version
        current_version = get_schema_version(conn)
        print(f"\nCurrent schema version: {current_version}")

        if current_version >= TARGET_VERSION:
            print(f"\nDatabase already at version {current_version} (>= {TARGET_VERSION})")
            print("Migration is a no-op.")
            return True

        if dry_run:
            print("\n[DRY RUN] Would perform the following migrations:")
            print("  - Create schema_version table")
            print("  - Create users table with predefined accounts")
            print("  - Create web_sessions table")
            print("  - Create web_messages table")
            print("  - Migrate preferences table")
            print("  - Migrate sessions table")
            print("  - Migrate recipe_feedback table")
            print("  - Migrate cooking_history table")
            print("  - Migrate saved_recipes table")
            print("  - Migrate meal_plans table")
            print("  - Enable WAL mode")
            print(f"  - Update schema version to {TARGET_VERSION}")
            return True

        # Acquire migration lock
        print("\nStep 2: Acquiring migration lock...")
        if not acquire_migration_lock(conn):
            print("  ERROR: Could not acquire migration lock")
            print("  Another migration may be in progress")
            return False
        print("  Lock acquired")

        # Run migrations
        print("\nStep 3: Running migrations...")

        create_schema_version_table(conn)
        create_users_table(conn)
        create_web_sessions_table(conn)
        create_web_messages_table(conn)
        migrate_preferences_table(conn)
        migrate_sessions_table(conn)
        migrate_recipe_feedback_table(conn)
        migrate_cooking_history_table(conn)
        migrate_saved_recipes_table(conn)
        migrate_meal_plans_table(conn)

        # Enable WAL mode
        print("\nStep 4: Enabling WAL mode...")
        enable_wal_mode(conn)

        # Validate foreign keys
        print("\nStep 5: Validating foreign keys...")
        errors = validate_foreign_keys(conn)
        if errors:
            print("  WARNING: Foreign key violations found:")
            for error in errors[:10]:  # Show first 10
                print(f"    {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")
        else:
            print("  All foreign keys valid")

        # Record schema version
        print(f"\nStep 6: Recording schema version {TARGET_VERSION}...")
        conn.execute("""
            INSERT INTO schema_version (version, description)
            VALUES (?, ?)
        """, (TARGET_VERSION, "Multi-user support with web sessions"))

        # Commit transaction
        conn.commit()
        print("  Migration committed")

        print(f"\n{'=' * 60}")
        print("Migration completed successfully!")
        print(f"Schema version: {TARGET_VERSION}")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\nERROR: Migration failed: {e}")
        conn.rollback()
        print("Transaction rolled back")

        if not no_backup and not dry_run:
            print(f"\nTo restore from backup:")
            print(f"  cp {backup_path} {db_path}")

        raise

    finally:
        conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate database to multi-user schema"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup (not recommended)"
    )

    args = parser.parse_args()

    try:
        success = run_migration(dry_run=args.dry_run, no_backup=args.no_backup)
        sys.exit(0 if success else 1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
