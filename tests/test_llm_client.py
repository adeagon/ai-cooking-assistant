"""Tests for OllamaLLMClient - unit tests with mocked HTTP client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.llm.ollama_client import OllamaLLMClient
from src.llm.base import LLMClient


class TestOllamaLLMClientInit:
    """Tests for OllamaLLMClient initialization."""

    def test_default_initialization(self):
        """Test client initializes with default values."""
        client = OllamaLLMClient()

        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen3:14b"
        assert client.timeout == 300.0
        assert client.disable_thinking is True
        assert client.client is not None

    def test_custom_initialization(self):
        """Test client initializes with custom values."""
        client = OllamaLLMClient(
            base_url="http://custom:8080/",
            model="llama3:8b",
            timeout=60.0,
            disable_thinking=False
        )

        assert client.base_url == "http://custom:8080"  # Trailing slash stripped
        assert client.model == "llama3:8b"
        assert client.timeout == 60.0
        assert client.disable_thinking is False

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slashes are stripped from base_url."""
        client = OllamaLLMClient(base_url="http://localhost:11434///")
        assert client.base_url == "http://localhost:11434"

    def test_implements_llm_client_interface(self):
        """Test that OllamaLLMClient implements LLMClient interface."""
        client = OllamaLLMClient()
        assert isinstance(client, LLMClient)


class TestOllamaLLMClientChat:
    """Tests for the chat method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return OllamaLLMClient()

    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        """Test successful chat request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello! How can I help you?"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = await client.chat(messages)

            assert result == "Hello! How can I help you?"
            mock_post.assert_called_once()

            # Verify the URL and payload
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/chat"
            payload = call_args[1]["json"]
            assert payload["model"] == "qwen3:14b"
            assert payload["messages"] == messages
            assert payload["stream"] is False
            assert payload["think"] is False  # disable_thinking=True

    @pytest.mark.asyncio
    async def test_chat_with_thinking_enabled(self):
        """Test chat with thinking mode enabled."""
        client = OllamaLLMClient(disable_thinking=False)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response with thinking"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.chat([{"role": "user", "content": "Test"}])

            # When disable_thinking=False, 'think' should NOT be in payload
            payload = mock_post.call_args[1]["json"]
            assert "think" not in payload

    @pytest.mark.asyncio
    async def test_chat_with_explicit_think_kwarg(self, client):
        """Test that explicit think kwarg overrides disable_thinking."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Pass explicit think=True even though disable_thinking=True
            await client.chat([{"role": "user", "content": "Test"}], think=True)

            payload = mock_post.call_args[1]["json"]
            assert payload["think"] is True

    @pytest.mark.asyncio
    async def test_chat_with_extra_kwargs(self, client):
        """Test that extra kwargs are passed to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.chat(
                [{"role": "user", "content": "Test"}],
                temperature=0.7,
                max_tokens=1024
            )

            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_chat_empty_response(self, client):
        """Test handling of empty response content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.chat([{"role": "user", "content": "Test"}])

            assert result == ""

    @pytest.mark.asyncio
    async def test_chat_missing_message_key(self, client):
        """Test handling of response without message key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.chat([{"role": "user", "content": "Test"}])

            assert result == ""

    @pytest.mark.asyncio
    async def test_chat_http_error(self, client):
        """Test that HTTP errors are propagated."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500)
            )

            with pytest.raises(httpx.HTTPError):
                await client.chat([{"role": "user", "content": "Test"}])

    @pytest.mark.asyncio
    async def test_chat_connection_error(self, client):
        """Test handling of connection errors."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(httpx.HTTPError):
                await client.chat([{"role": "user", "content": "Test"}])

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, client):
        """Test handling of timeout errors."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(httpx.HTTPError):
                await client.chat([{"role": "user", "content": "Test"}])


class TestOllamaLLMClientGenerate:
    """Tests for the generate method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return OllamaLLMClient()

    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful generate request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Generated text response"
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.generate("Complete this: Hello")

            assert result == "Generated text response"
            mock_post.assert_called_once()

            # Verify the URL and payload
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"
            payload = call_args[1]["json"]
            assert payload["model"] == "qwen3:14b"
            assert payload["prompt"] == "Complete this: Hello"
            assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, client):
        """Test generate with extra parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Response"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.generate(
                "Test prompt",
                temperature=0.5,
                num_predict=256
            )

            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.5
            assert payload["num_predict"] == 256

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, client):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.generate("Test")

            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_http_error(self, client):
        """Test that HTTP errors are propagated."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=MagicMock(status_code=404)
            )

            with pytest.raises(httpx.HTTPError):
                await client.generate("Test")


class TestOllamaLLMClientContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self):
        """Test that context manager works correctly."""
        async with OllamaLLMClient() as client:
            assert isinstance(client, OllamaLLMClient)
            # Client should be usable
            assert client.client is not None

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test that close method closes the HTTP client."""
        client = OllamaLLMClient()

        with patch.object(client.client, 'aclose', new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()


class TestOllamaLLMClientMultipleMessages:
    """Tests for chat with multiple messages (conversation history)."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return OllamaLLMClient()

    @pytest.mark.asyncio
    async def test_chat_with_conversation_history(self, client):
        """Test chat with multiple messages in conversation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Based on our conversation..."}
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What can you help with?"}
            ]

            result = await client.chat(messages)

            assert result == "Based on our conversation..."
            payload = mock_post.call_args[1]["json"]
            assert len(payload["messages"]) == 4
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_with_empty_messages(self, client):
        """Test chat with empty message list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.chat([])

            payload = mock_post.call_args[1]["json"]
            assert payload["messages"] == []


class TestOllamaLLMClientModels:
    """Tests for different model configurations."""

    @pytest.mark.asyncio
    async def test_chat_with_different_model(self):
        """Test that different models are used correctly."""
        client = OllamaLLMClient(model="cooking-assistant")

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response"}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.chat([{"role": "user", "content": "Test"}])

            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "cooking-assistant"

    @pytest.mark.asyncio
    async def test_generate_with_different_model(self):
        """Test generate with different model."""
        client = OllamaLLMClient(model="intent-classifier")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "classified intent"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await client.generate("Classify: I love this recipe")

            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "intent-classifier"
