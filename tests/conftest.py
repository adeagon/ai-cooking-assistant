"""Pytest configuration and fixtures."""

import sqlite3
from pathlib import Path

import pytest

from src.app.settings import Settings
from src.memory._table_init import reset_initialized_tables

# Test user ID for store tests (stable UUID for deterministic tests)
TEST_USER_ID = "test-user-00000000-0000-0000-0000-000000000001"


@pytest.fixture
def test_user_id():
    """Provide a test user ID for store tests."""
    return TEST_USER_ID


@pytest.fixture
def test_settings():
    """Create test settings with overrides."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.3:70b",
        log_level="DEBUG"
    )


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with users table for testing.

    This fixture creates the minimal schema needed for multi-user stores.
    """
    # Reset table initialization tracking for each test
    reset_initialized_tables()

    db_path = tmp_path / "test.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create users table (required for foreign keys)
    cursor.execute("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    # Insert test user
    cursor.execute(
        "INSERT INTO users (id, username, is_active) VALUES (?, ?, ?)",
        (TEST_USER_ID, "test_user", True)
    )

    # Create recipes table with full schema (required for get_recipe_by_id)
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT,
            tags TEXT,
            ingredients_raw TEXT,
            ingredients_normalized TEXT,
            instructions TEXT,
            rating_avg REAL DEFAULT 4.0,
            rating_count INTEGER DEFAULT 10,
            minutes INTEGER DEFAULT 30,
            n_steps INTEGER DEFAULT 5,
            n_ingredients INTEGER DEFAULT 8,
            source TEXT DEFAULT 'test'
        )
    """)

    # Add test recipes with full data
    cursor.execute("""
        INSERT INTO recipes VALUES (
            '123', 'Test Recipe', '["italian"]',
            '["1 lb pasta", "2 cups tomato sauce", "1/2 cup parmesan"]',
            '["pasta", "tomato sauce", "parmesan"]',
            '["Boil pasta", "Heat sauce", "Combine and serve"]',
            4.5, 25, 30, 3, 3, 'test'
        )
    """)
    cursor.execute("""
        INSERT INTO recipes VALUES (
            '456', 'Another Recipe', '["mexican"]',
            '["1 lb ground beef", "1 packet taco seasoning", "8 taco shells"]',
            '["ground beef", "taco seasoning", "taco shells"]',
            '["Brown beef", "Add seasoning", "Serve in shells"]',
            4.2, 18, 25, 3, 3, 'test'
        )
    """)
    cursor.execute("""
        INSERT INTO recipes VALUES (
            '789', 'Third Recipe', '["italian", "pasta"]',
            '["1 lb spaghetti", "4 eggs", "1 cup pancetta"]',
            '["spaghetti", "eggs", "pancetta"]',
            '["Cook pasta", "Fry pancetta", "Mix with eggs"]',
            4.8, 42, 35, 3, 3, 'test'
        )
    """)

    # Add additional test recipes for tests that iterate (0-9)
    for i in range(10):
        cursor.execute("""
            INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(i), f"Recipe {i}", "[]",
            '["ingredient 1", "ingredient 2"]',
            '["ingredient1", "ingredient2"]',
            '["Step 1", "Step 2"]',
            4.0, 10, 30, 2, 2, "test"
        ))

    conn.commit()
    conn.close()

    return db_path
