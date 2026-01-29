"""E2E tests for login flow."""

import pytest

# Mark all tests in this module as E2E tests requiring Playwright
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        True,  # Skip by default, enable with: pytest -m e2e --run-e2e
        reason="E2E tests require running server and Playwright"
    )
]


def test_login_shows_user_info(test_server, page):
    """Test that logging in shows user info."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])

    # Initially should show login form
    assert not chat.is_logged_in()
    assert chat.login_form.is_visible()

    # Login
    chat.login("alex")

    # Should now show user info
    assert chat.is_logged_in()
    assert "Alex" in chat.get_current_user()


def test_logout_clears_session(test_server, page):
    """Test that logging out clears the session."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])

    # Login and then logout
    chat.login("jordan")
    assert chat.is_logged_in()

    chat.logout()

    # Should be back to login form
    assert not chat.is_logged_in()
    assert chat.login_form.is_visible()


def test_login_enables_chat_input(test_server, page):
    """Test that chat input is disabled until login."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])

    # Input should be disabled before login
    assert not chat.is_input_enabled()

    # Login
    chat.login("taylor")

    # Input should now be enabled
    assert chat.is_input_enabled()


def test_session_persists_on_refresh(test_server, page):
    """Test that session persists after page refresh."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])

    # Login
    chat.login("casey")
    assert chat.is_logged_in()

    # Refresh the page
    page.reload()
    page.wait_for_load_state("networkidle")

    # Should still be logged in
    assert chat.is_logged_in()
    assert "Casey" in chat.get_current_user()


def test_user_switch(test_server, page):
    """Test switching between users."""
    from tests.web.e2e.pages.chat_page import ChatPage

    chat = ChatPage(page)
    chat.goto(test_server["url"])

    # Login as alex
    chat.login("alex")
    assert "Alex" in chat.get_current_user()

    # Logout and login as jordan
    chat.logout()
    chat.login("jordan")
    assert "Jordan" in chat.get_current_user()
