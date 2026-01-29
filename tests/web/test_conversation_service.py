"""Tests for ConversationService."""

import time

import pytest

from src.web.services.conversation_service import ConversationService
from src.web.services.user_service import UserService


class TestConversationService:
    """Tests for ConversationService."""

    def test_create_conversation(
        self, conversation_service: ConversationService, test_user
    ):
        """Test creating a new conversation."""
        conv_id = conversation_service.create(test_user.id, title="Test Chat")

        assert conv_id is not None
        assert len(conv_id) == 36  # UUID

    def test_create_conversation_without_title(
        self, conversation_service: ConversationService, test_user
    ):
        """Test creating conversation without title."""
        conv_id = conversation_service.create(test_user.id)

        conv = conversation_service.get(conv_id, test_user.id)
        assert conv.title is None

    def test_get_conversation(
        self, conversation_service: ConversationService, test_user
    ):
        """Test getting a conversation."""
        conv_id = conversation_service.create(test_user.id, title="My Chat")
        conv = conversation_service.get(conv_id, test_user.id)

        assert conv is not None
        assert conv.id == conv_id
        assert conv.title == "My Chat"

    def test_get_conversation_not_found(
        self, conversation_service: ConversationService, test_user
    ):
        """Test getting non-existent conversation returns None."""
        result = conversation_service.get("nonexistent-id", test_user.id)
        assert result is None

    def test_get_conversation_wrong_user(
        self, conversation_service: ConversationService, user_service: UserService
    ):
        """Test that users can't access other users' conversations."""
        user1 = user_service.create(username="user1")
        user2 = user_service.create(username="user2")

        conv_id = conversation_service.create(user1.id, title="User1's Chat")

        # User2 should not be able to access user1's conversation
        result = conversation_service.get(conv_id, user2.id)
        assert result is None

    def test_list_for_user(
        self, conversation_service: ConversationService, test_user
    ):
        """Test listing conversations for a user."""
        conversation_service.create(test_user.id, title="Chat 1")
        conversation_service.create(test_user.id, title="Chat 2")
        conversation_service.create(test_user.id, title="Chat 3")

        convs = conversation_service.list_for_user(test_user.id)

        assert len(convs) == 3
        titles = [c.title for c in convs]
        assert "Chat 1" in titles
        assert "Chat 2" in titles
        assert "Chat 3" in titles

    def test_list_for_user_sorted_by_last_message(
        self, conversation_service: ConversationService, test_user
    ):
        """Test that conversations are sorted by last_message_at desc."""
        c1 = conversation_service.create(test_user.id, title="First")
        time.sleep(0.01)
        c2 = conversation_service.create(test_user.id, title="Second")
        time.sleep(0.01)
        c3 = conversation_service.create(test_user.id, title="Third")

        # Add message to first conversation to make it most recent
        conversation_service.add_message(c1, test_user.id, "user", "Hello")

        convs = conversation_service.list_for_user(test_user.id)

        # First should be at top (most recent message)
        assert convs[0].id == c1
        # Then Third (created last, no messages)
        assert convs[1].id == c3
        # Then Second
        assert convs[2].id == c2

    def test_list_for_user_excludes_archived(
        self, conversation_service: ConversationService, test_user
    ):
        """Test that archived conversations are excluded from list."""
        c1 = conversation_service.create(test_user.id, title="Active")
        c2 = conversation_service.create(test_user.id, title="Archived")

        conversation_service.archive(c2, test_user.id)

        convs = conversation_service.list_for_user(test_user.id)

        assert len(convs) == 1
        assert convs[0].id == c1

    def test_list_for_user_with_limit(
        self, conversation_service: ConversationService, test_user
    ):
        """Test limiting number of returned conversations."""
        for i in range(5):
            conversation_service.create(test_user.id, title=f"Chat {i}")

        convs = conversation_service.list_for_user(test_user.id, limit=3)

        assert len(convs) == 3

    def test_update_title(
        self, conversation_service: ConversationService, test_user
    ):
        """Test updating conversation title."""
        conv_id = conversation_service.create(test_user.id, title="Old Title")
        result = conversation_service.update_title(conv_id, test_user.id, "New Title")

        assert result is True

        conv = conversation_service.get(conv_id, test_user.id)
        assert conv.title == "New Title"

    def test_update_title_not_found(
        self, conversation_service: ConversationService, test_user
    ):
        """Test updating non-existent conversation returns False."""
        result = conversation_service.update_title("nonexistent", test_user.id, "Title")
        assert result is False

    def test_archive_conversation(
        self, conversation_service: ConversationService, test_user
    ):
        """Test archiving a conversation."""
        conv_id = conversation_service.create(test_user.id, title="To Archive")
        result = conversation_service.archive(conv_id, test_user.id)

        assert result is True

        # Should no longer be accessible via get
        conv = conversation_service.get(conv_id, test_user.id)
        assert conv is None

    def test_archive_conversation_not_found(
        self, conversation_service: ConversationService, test_user
    ):
        """Test archiving non-existent conversation returns False."""
        result = conversation_service.archive("nonexistent", test_user.id)
        assert result is False

    def test_add_message(
        self, conversation_service: ConversationService, test_user
    ):
        """Test adding a message to a conversation."""
        conv_id = conversation_service.create(test_user.id)
        msg_id = conversation_service.add_message(
            conv_id, test_user.id, "user", "Hello, world!"
        )

        assert msg_id is not None
        assert len(msg_id) == 36  # UUID

    def test_add_message_updates_last_message_at(
        self, conversation_service: ConversationService, test_user
    ):
        """Test that adding message updates conversation's last_message_at."""
        conv_id = conversation_service.create(test_user.id)
        conv_before = conversation_service.get(conv_id, test_user.id)
        assert conv_before.last_message_at is None

        conversation_service.add_message(conv_id, test_user.id, "user", "Hello")

        conv_after = conversation_service.get(conv_id, test_user.id)
        assert conv_after.last_message_at is not None

    def test_add_message_with_meta(
        self, conversation_service: ConversationService, test_user
    ):
        """Test adding message with metadata."""
        conv_id = conversation_service.create(test_user.id)
        meta = {
            "recipe_cards": [{"recipe_id": "123", "title": "Test Recipe"}],
            "intent": "recommend"
        }
        msg_id = conversation_service.add_message(
            conv_id, test_user.id, "assistant", "Here's a recipe", meta
        )

        messages = conversation_service.get_messages(conv_id, test_user.id)
        assert len(messages) == 1
        assert messages[0].meta.recipe_cards[0].recipe_id == "123"
        assert messages[0].meta.intent == "recommend"

    def test_get_messages(
        self, conversation_service: ConversationService, test_user
    ):
        """Test getting messages from a conversation."""
        conv_id = conversation_service.create(test_user.id)
        conversation_service.add_message(conv_id, test_user.id, "user", "Hello")
        conversation_service.add_message(conv_id, test_user.id, "assistant", "Hi!")
        conversation_service.add_message(conv_id, test_user.id, "user", "How are you?")

        messages = conversation_service.get_messages(conv_id, test_user.id)

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi!"
        assert messages[2].role == "user"
        assert messages[2].content == "How are you?"

    def test_get_messages_ordered_oldest_first(
        self, conversation_service: ConversationService, test_user
    ):
        """Test that messages are returned oldest first."""
        conv_id = conversation_service.create(test_user.id)
        conversation_service.add_message(conv_id, test_user.id, "user", "First")
        time.sleep(0.01)
        conversation_service.add_message(conv_id, test_user.id, "user", "Second")
        time.sleep(0.01)
        conversation_service.add_message(conv_id, test_user.id, "user", "Third")

        messages = conversation_service.get_messages(conv_id, test_user.id)

        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    def test_get_messages_wrong_user(
        self, conversation_service: ConversationService, user_service: UserService
    ):
        """Test that users can't access other users' messages."""
        user1 = user_service.create(username="user1")
        user2 = user_service.create(username="user2")

        conv_id = conversation_service.create(user1.id)
        conversation_service.add_message(conv_id, user1.id, "user", "Secret message")

        # User2 should get empty list
        messages = conversation_service.get_messages(conv_id, user2.id)
        assert messages == []

    def test_get_messages_with_limit(
        self, conversation_service: ConversationService, test_user
    ):
        """Test limiting number of returned messages."""
        conv_id = conversation_service.create(test_user.id)
        for i in range(10):
            conversation_service.add_message(conv_id, test_user.id, "user", f"Msg {i}")

        messages = conversation_service.get_messages(conv_id, test_user.id, limit=5)

        assert len(messages) == 5

    def test_get_recent_messages(
        self, conversation_service: ConversationService, test_user
    ):
        """Test getting recent messages for context building."""
        conv_id = conversation_service.create(test_user.id)
        for i in range(5):
            conversation_service.add_message(conv_id, test_user.id, "user", f"Msg {i}")

        messages = conversation_service.get_recent_messages(conv_id, limit=3)

        assert len(messages) == 3
        # Should be oldest first of the 3 most recent
        assert messages[0].content == "Msg 2"
        assert messages[1].content == "Msg 3"
        assert messages[2].content == "Msg 4"
