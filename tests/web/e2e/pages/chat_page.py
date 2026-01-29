"""Page Object Model for the chat interface."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page


class ChatPage:
    """Page object for the chat interface."""

    def __init__(self, page: "Page"):
        self.page = page

        # Selectors
        self.user_select = page.locator("#userSelect")
        self.login_btn = page.locator("#loginBtn")
        self.logout_btn = page.locator("#logoutBtn")
        self.user_name = page.locator("#userName")
        self.user_info = page.locator("#userInfo")
        self.login_form = page.locator("#loginForm")

        self.new_chat_btn = page.locator("#newChatBtn")
        self.conversations_list = page.locator("#conversationsList")

        self.welcome_screen = page.locator("#welcomeScreen")
        self.chat_messages = page.locator("#chatMessages")
        self.chat_input = page.locator("#chatInput")
        self.send_btn = page.locator("#sendBtn")
        self.chat_form = page.locator("#chatForm")

    def goto(self, base_url: str):
        """Navigate to the chat page."""
        self.page.goto(base_url)
        self.page.wait_for_load_state("networkidle")

    def login(self, username: str):
        """Login with the given username."""
        self.user_select.select_option(username)
        self.login_btn.click()
        # Wait for login to complete
        self.user_info.wait_for(state="visible", timeout=5000)

    def logout(self):
        """Logout the current user."""
        self.logout_btn.click()
        # Wait for logout to complete
        self.login_form.wait_for(state="visible", timeout=5000)

    def is_logged_in(self) -> bool:
        """Check if a user is currently logged in."""
        return self.user_info.is_visible()

    def get_current_user(self) -> str:
        """Get the name of the currently logged in user."""
        return self.user_name.text_content() or ""

    def send_message(self, message: str):
        """Send a chat message."""
        self.chat_input.fill(message)
        self.send_btn.click()

    def wait_for_response(self, timeout: int = 30000):
        """Wait for assistant response to appear."""
        # Wait for a message with class 'assistant' to appear
        self.page.locator(".message.assistant").last.wait_for(
            state="visible", timeout=timeout
        )

    def get_messages(self) -> list[dict]:
        """Get all messages in the current conversation."""
        messages = []
        for msg in self.page.locator(".message").all():
            role = "user" if "user" in (msg.get_attribute("class") or "") else "assistant"
            content = msg.locator(".message-text").text_content() or ""
            messages.append({"role": role, "content": content})
        return messages

    def start_new_conversation(self):
        """Start a new conversation."""
        self.new_chat_btn.click()
        # Wait for welcome screen to appear
        self.welcome_screen.wait_for(state="visible", timeout=5000)

    def get_conversation_count(self) -> int:
        """Get the number of conversations in the sidebar."""
        return self.conversations_list.locator(".conversation-item").count()

    def select_conversation(self, index: int):
        """Select a conversation by index (0-based)."""
        items = self.conversations_list.locator(".conversation-item").all()
        if index < len(items):
            items[index].click()
            # Wait for messages to load
            self.chat_messages.wait_for(state="visible", timeout=5000)

    def click_suggestion_chip(self, text: str):
        """Click a suggestion chip by its text content."""
        chip = self.page.locator(f".chip:has-text('{text}')")
        chip.click()

    def get_recipe_cards(self) -> list[dict]:
        """Get all recipe cards in the latest message."""
        cards = []
        for card in self.page.locator(".message.assistant").last.locator(".recipe-card").all():
            title = card.locator(".recipe-title").text_content() or ""
            summary = card.locator(".recipe-summary").text_content() or ""
            cards.append({"title": title, "summary": summary})
        return cards

    def is_input_enabled(self) -> bool:
        """Check if the chat input is enabled."""
        return self.chat_input.is_enabled()

    def is_welcome_screen_visible(self) -> bool:
        """Check if the welcome screen is visible."""
        return self.welcome_screen.is_visible()

    def is_chat_area_visible(self) -> bool:
        """Check if the chat messages area is visible."""
        return self.chat_messages.is_visible()
