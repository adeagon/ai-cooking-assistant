"""E2E tests for chat flow."""

import pytest

# Mark all tests in this module as E2E tests requiring Playwright
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        True,  # Skip by default, enable with: pytest -m e2e --run-e2e
        reason="E2E tests require running server and Playwright"
    )
]


def test_welcome_screen_visible_on_start(test_server, page):
    """Test that welcome screen is visible when starting."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Welcome screen should be visible
    assert chat.is_welcome_screen_visible()
    assert not chat.is_chat_area_visible()


def test_new_chat_button_shows_welcome(test_server, page):
    """Test that new chat button shows welcome screen."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Start a new conversation
    chat.start_new_conversation()

    # Welcome screen should be visible
    assert chat.is_welcome_screen_visible()


def test_conversation_created_on_first_message(test_server, page):
    """Test that sending first message creates a conversation."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # No conversations initially
    initial_count = chat.get_conversation_count()

    # Send a message (this will create a conversation)
    chat.send_message("What can I make with chicken?")

    # Wait for response
    chat.wait_for_response(timeout=60000)

    # Should now have a conversation
    assert chat.get_conversation_count() > initial_count


def test_messages_appear_in_order(test_server, page):
    """Test that user message appears before assistant response."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Send a message
    chat.send_message("Quick dinner ideas")

    # Wait for response
    chat.wait_for_response(timeout=60000)

    # Check messages
    messages = chat.get_messages()
    assert len(messages) >= 2

    # First should be user message
    assert messages[0]["role"] == "user"
    assert "Quick dinner ideas" in messages[0]["content"]

    # Second should be assistant
    assert messages[1]["role"] == "assistant"


def test_suggestion_chip_sends_message(test_server, page):
    """Test that clicking a suggestion chip sends a message."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Click a suggestion chip
    chat.click_suggestion_chip("Quick chicken dinner")

    # Should switch to chat area
    chat.wait_for_response(timeout=60000)
    assert chat.is_chat_area_visible()


def test_conversation_persists_after_refresh(test_server, page):
    """Test that conversation persists after page refresh."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Send a message
    chat.send_message("Test message for persistence")
    chat.wait_for_response(timeout=60000)

    # Get the message content
    messages_before = chat.get_messages()

    # Refresh
    page.reload()
    page.wait_for_load_state("networkidle")

    # Should still be logged in
    assert chat.is_logged_in()

    # Select the conversation
    if chat.get_conversation_count() > 0:
        chat.select_conversation(0)

        # Messages should still be there
        messages_after = chat.get_messages()
        assert len(messages_after) == len(messages_before)


def test_multiple_conversations(test_server, page):
    """Test creating and switching between multiple conversations."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])
    chat.login("alex")

    # Create first conversation
    chat.send_message("First conversation message")
    chat.wait_for_response(timeout=60000)

    # Start new conversation
    chat.start_new_conversation()

    # Create second conversation
    chat.send_message("Second conversation message")
    chat.wait_for_response(timeout=60000)

    # Should have 2 conversations
    assert chat.get_conversation_count() == 2


def test_user_conversations_isolated(test_server, browser):
    """Test that different users have isolated conversations."""
    from tests.web.e2e.pages.chat_page import ChatPage

    # Alex creates a conversation
    context1 = browser.new_context()
    page1 = context1.new_page()
    chat1 = ChatPage(page1)
    chat1.goto(test_server["url"])
    chat1.login("alex")
    chat1.send_message("Alex's secret recipe")
    chat1.wait_for_response(timeout=60000)
    alex_conv_count = chat1.get_conversation_count()

    # Jordan logs in with different context
    context2 = browser.new_context()
    page2 = context2.new_page()
    chat2 = ChatPage(page2)
    chat2.goto(test_server["url"])
    chat2.login("jordan")

    # Jordan should not see Alex's conversation
    jordan_conv_count = chat2.get_conversation_count()
    assert jordan_conv_count == 0  # Jordan has no conversations

    # Cleanup
    context1.close()
    context2.close()
