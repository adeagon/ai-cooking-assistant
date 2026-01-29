"""Tests for UserService."""

import pytest

from src.web.services.user_service import UserService


class TestUserService:
    """Tests for UserService."""

    def test_create_user(self, user_service: UserService):
        """Test creating a new user."""
        user = user_service.create(username="alice", display_name="Alice Smith")

        assert user.username == "alice"
        assert user.display_name == "Alice Smith"
        assert user.id is not None
        assert len(user.id) == 36  # UUID length

    def test_create_user_without_display_name(self, user_service: UserService):
        """Test creating a user without display name."""
        user = user_service.create(username="bob")

        assert user.username == "bob"
        assert user.display_name is None

    def test_create_duplicate_username_raises(self, user_service: UserService):
        """Test that duplicate username raises ValueError."""
        user_service.create(username="charlie")

        with pytest.raises(ValueError, match="already exists"):
            user_service.create(username="charlie")

    def test_get_by_id(self, user_service: UserService):
        """Test getting user by ID."""
        created = user_service.create(username="david")
        fetched = user_service.get_by_id(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.username == "david"

    def test_get_by_id_not_found(self, user_service: UserService):
        """Test getting non-existent user by ID returns None."""
        result = user_service.get_by_id("nonexistent-uuid")
        assert result is None

    def test_get_by_username(self, user_service: UserService):
        """Test getting user by username."""
        created = user_service.create(username="eve")
        fetched = user_service.get_by_username("eve")

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.username == "eve"

    def test_get_by_username_not_found(self, user_service: UserService):
        """Test getting non-existent user by username returns None."""
        result = user_service.get_by_username("nonexistent")
        assert result is None

    def test_get_username(self, user_service: UserService):
        """Test UUID to username mapping."""
        user = user_service.create(username="frank")
        username = user_service.get_username(user.id)

        assert username == "frank"

    def test_get_username_not_found(self, user_service: UserService):
        """Test UUID to username mapping for non-existent user."""
        result = user_service.get_username("nonexistent-uuid")
        assert result is None

    def test_get_all(self, user_service: UserService):
        """Test getting all users."""
        user_service.create(username="user1")
        user_service.create(username="user2")
        user_service.create(username="user3")

        users = user_service.get_all()

        assert len(users) == 3
        usernames = [u.username for u in users]
        assert "user1" in usernames
        assert "user2" in usernames
        assert "user3" in usernames

    def test_get_all_sorted_by_username(self, user_service: UserService):
        """Test that get_all returns users sorted by username."""
        user_service.create(username="zebra")
        user_service.create(username="apple")
        user_service.create(username="mango")

        users = user_service.get_all()
        usernames = [u.username for u in users]

        assert usernames == ["apple", "mango", "zebra"]

    def test_update_display_name(self, user_service: UserService):
        """Test updating user's display name."""
        user = user_service.create(username="george", display_name="George")
        result = user_service.update_display_name(user.id, "George Washington")

        assert result is True

        updated = user_service.get_by_id(user.id)
        assert updated.display_name == "George Washington"

    def test_update_display_name_not_found(self, user_service: UserService):
        """Test updating non-existent user returns False."""
        result = user_service.update_display_name("nonexistent", "New Name")
        assert result is False
