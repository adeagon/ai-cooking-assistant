"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM client implementations.

    This interface allows swapping between different LLM runtimes
    (Ollama, LM Studio, llama.cpp, vLLM, etc.) without changing application code.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> str:
        """Send chat messages and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]
            **kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            The assistant's response text
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Generate completion for a single prompt.

        Args:
            prompt: The text prompt
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text
        """
        pass
