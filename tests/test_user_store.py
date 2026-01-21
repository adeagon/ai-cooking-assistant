"""Tests for UserStore."""

import sqlite3

import pytest

from src.app.constants import DEFAULT_USER_USERNAME
from src.memory.user_store import User, UserStore


class TestUserStore:
    """Test UserStore functionality."""

    def test_initialization_creates_table(self, temp_db):
        """Test that initialization creates the users table."""
        store = UserStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='users'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_create_user(self, temp_db):
        """Test creating a new user."""
        store = UserStore(temp_db)

        user = store.create_user("newuser", "password123")

        assert user is not None
        assert user.username == "newuser"
        assert user.password_hash is not None
        assert user.is_active is True
        assert user.id is not None

    def test_create_user_without_password(self, temp_db):
        """Test creating a user without password (set on first login)."""
        store = UserStore(temp_db)

        user = store.create_user("newuser")

        assert user is not None
        assert user.username == "newuser"
        assert user.password_hash is None
        assert user.is_active is True

    def test_create_duplicate_user_raises_error(self, temp_db):
        """Test that creating duplicate user raises IntegrityError."""
        store = UserStore(temp_db)

        store.create_user("duplicateuser")

        with pytest.raises(sqlite3.IntegrityError):
            store.create_user("duplicateuser")

    def test_create_user_case_insensitive(self, temp_db):
        """Test that username is case-insensitive."""
        store = UserStore(temp_db)

        store.create_user("MyUser")

        # Same username with different case should fail
        with pytest.raises(sqlite3.IntegrityError):
            store.create_user("myuser")

    def test_create_user_if_not_exists_new_user(self, temp_db):
        """Test create_user_if_not_exists with new user."""
        store = UserStore(temp_db)

        user = store.create_user_if_not_exists("newuser")

        assert user is not None
        assert user.username == "newuser"

    def test_create_user_if_not_exists_existing_user(self, temp_db, test_user_id):
        """Test create_user_if_not_exists with existing user."""
        store = UserStore(temp_db)

        # test_user already exists from fixture
        user = store.create_user_if_not_exists("test_user")

        assert user is None  # Returns None if user exists

    def test_get_user_by_id(self, temp_db, test_user_id):
        """Test getting user by ID."""
        store = UserStore(temp_db)

        user = store.get_user_by_id(test_user_id)

        assert user is not None
        assert user.id == test_user_id
        assert user.username == "test_user"

    def test_get_user_by_id_not_found(self, temp_db):
        """Test getting non-existent user by ID."""
        store = UserStore(temp_db)

        user = store.get_user_by_id("nonexistent-id")

        assert user is None

    def test_get_user_by_username(self, temp_db, test_user_id):
        """Test getting user by username."""
        store = UserStore(temp_db)

        user = store.get_user_by_username("test_user")

        assert user is not None
        assert user.username == "test_user"
        assert user.id == test_user_id

    def test_get_user_by_username_case_insensitive(self, temp_db, test_user_id):
        """Test that username lookup is case-insensitive."""
        store = UserStore(temp_db)

        user = store.get_user_by_username("TEST_USER")

        assert user is not None
        assert user.id == test_user_id

    def test_get_user_by_username_not_found(self, temp_db):
        """Test getting non-existent user by username."""
        store = UserStore(temp_db)

        user = store.get_user_by_username("nonexistent")

        assert user is None


class TestPasswordVerification:
    """Test password verification functionality."""

    def test_verify_password_valid(self, temp_db):
        """Test verifying valid password."""
        store = UserStore(temp_db)

        store.create_user("authuser", "correctpassword")

        user = store.verify_password("authuser", "correctpassword")

        assert user is not None
        assert user.username == "authuser"

    def test_verify_password_invalid(self, temp_db):
        """Test verifying invalid password."""
        store = UserStore(temp_db)

        store.create_user("authuser", "correctpassword")

        user = store.verify_password("authuser", "wrongpassword")

        assert user is None

    def test_verify_password_no_password_set(self, temp_db):
        """Test verifying when user has no password set."""
        store = UserStore(temp_db)

        store.create_user("nopassuser")  # No password

        user = store.verify_password("nopassuser", "anypassword")

        assert user is None

    def test_verify_password_user_not_found(self, temp_db):
        """Test verifying password for non-existent user."""
        store = UserStore(temp_db)

        user = store.verify_password("nonexistent", "password")

        assert user is None

    def test_verify_password_inactive_user(self, temp_db):
        """Test that inactive users cannot login."""
        store = UserStore(temp_db)

        user = store.create_user("inactiveuser", "password")
        store.set_active(user.id, False)

        result = store.verify_password("inactiveuser", "password")

        assert result is None

    def test_verify_password_blocks_default_user(self, temp_db):
        """Test that default_user cannot login via password."""
        store = UserStore(temp_db)

        # Create default_user with a password (should still be blocked)
        store.create_user_if_not_exists(DEFAULT_USER_USERNAME, "hashed_password")

        user = store.verify_password(DEFAULT_USER_USERNAME, "anypassword")

        assert user is None


class TestPasswordManagement:
    """Test password set/update functionality."""

    def test_set_password_new(self, temp_db):
        """Test setting password for user without one."""
        store = UserStore(temp_db)

        user = store.create_user("newuser")
        assert user.password_hash is None

        result = store.set_password(user.id, "newpassword")

        assert result is True

        # Verify password works
        verified = store.verify_password("newuser", "newpassword")
        assert verified is not None

    def test_set_password_update(self, temp_db):
        """Test updating existing password."""
        store = UserStore(temp_db)

        user = store.create_user("existinguser", "oldpassword")

        result = store.set_password(user.id, "newpassword")

        assert result is True

        # Old password should fail
        assert store.verify_password("existinguser", "oldpassword") is None
        # New password should work
        assert store.verify_password("existinguser", "newpassword") is not None

    def test_set_password_user_not_found(self, temp_db):
        """Test setting password for non-existent user."""
        store = UserStore(temp_db)

        result = store.set_password("nonexistent-id", "password")

        assert result is False

    def test_needs_password_setup_true(self, temp_db):
        """Test needs_password_setup returns True when no password."""
        store = UserStore(temp_db)

        store.create_user("nopassuser")

        assert store.needs_password_setup("nopassuser") is True

    def test_needs_password_setup_false(self, temp_db):
        """Test needs_password_setup returns False when password set."""
        store = UserStore(temp_db)

        store.create_user("withpassuser", "password")

        assert store.needs_password_setup("withpassuser") is False

    def test_needs_password_setup_nonexistent(self, temp_db):
        """Test needs_password_setup returns False for non-existent user."""
        store = UserStore(temp_db)

        assert store.needs_password_setup("nonexistent") is False


class TestUserActiveStatus:
    """Test user active/inactive status."""

    def test_set_active_disable(self, temp_db):
        """Test disabling a user."""
        store = UserStore(temp_db)

        user = store.create_user("activeuser")
        assert user.is_active is True

        result = store.set_active(user.id, False)

        assert result is True

        updated = store.get_user_by_id(user.id)
        assert updated.is_active is False

    def test_set_active_enable(self, temp_db):
        """Test re-enabling a user."""
        store = UserStore(temp_db)

        user = store.create_user("tempuser")
        store.set_active(user.id, False)
        store.set_active(user.id, True)

        updated = store.get_user_by_id(user.id)
        assert updated.is_active is True

    def test_set_active_nonexistent(self, temp_db):
        """Test set_active for non-existent user."""
        store = UserStore(temp_db)

        result = store.set_active("nonexistent-id", False)

        assert result is False


class TestListUsers:
    """Test user listing functionality."""

    def test_list_users_active_only(self, temp_db, test_user_id):
        """Test listing only active users."""
        store = UserStore(temp_db)

        # Create additional users
        user1 = store.create_user("user1")
        user2 = store.create_user("user2")
        store.set_active(user2.id, False)

        users = store.list_users(include_inactive=False)

        usernames = [u.username for u in users]
        assert "test_user" in usernames
        assert "user1" in usernames
        assert "user2" not in usernames

    def test_list_users_include_inactive(self, temp_db, test_user_id):
        """Test listing all users including inactive."""
        store = UserStore(temp_db)

        user1 = store.create_user("user1")
        user2 = store.create_user("user2")
        store.set_active(user2.id, False)

        users = store.list_users(include_inactive=True)

        usernames = [u.username for u in users]
        assert "test_user" in usernames
        assert "user1" in usernames
        assert "user2" in usernames

    def test_list_users_ordered(self, temp_db):
        """Test that users are ordered by username."""
        store = UserStore(temp_db)

        store.create_user("zzzuser")
        store.create_user("aaauser")
        store.create_user("mmmuser")

        users = store.list_users()

        # Should be ordered by username (case-insensitive)
        usernames = [u.username for u in users]
        assert usernames == sorted(usernames, key=str.lower)
