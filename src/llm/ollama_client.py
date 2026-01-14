"""Ollama LLM client implementation."""

from typing import Any
import httpx
from src.llm.base import LLMClient
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class OllamaLLMClient(LLMClient):
    """LLM client implementation for Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        timeout: float = 300.0,
        disable_thinking: bool = True
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
            disable_thinking: Disable Qwen3 thinking mode for faster responses
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(
            "Initialized Ollama client",
            base_url=base_url,
            model=model,
            disable_thinking=disable_thinking
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> str:
        """Send chat messages to Ollama and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)

        Returns:
            The assistant's response text

        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        }

        # Disable thinking mode for Qwen3 models if configured
        if self.disable_thinking and "think" not in kwargs:
            payload["think"] = False

        logger.debug(
            "Sending chat request to Ollama",
            url=url,
            num_messages=len(messages)
        )

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data.get("message", {}).get("content", "")

            logger.debug(
                "Received chat response from Ollama",
                response_length=len(content)
            )

            return content

        except httpx.HTTPError as e:
            logger.error(
                "Ollama chat request failed",
                error=str(e),
                url=url
            )
            raise

    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate completion for a single prompt.

        Args:
            prompt: The text prompt
            **kwargs: Additional model parameters

        Returns:
            The generated text

        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        logger.debug(
            "Sending generate request to Ollama",
            url=url,
            prompt_length=len(prompt)
        )

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "")

            logger.debug(
                "Received generate response from Ollama",
                response_length=len(content)
            )

            return content

        except httpx.HTTPError as e:
            logger.error(
                "Ollama generate request failed",
                error=str(e),
                url=url
            )
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        logger.debug("Closed Ollama client")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
