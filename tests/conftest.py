"""Pytest configuration and fixtures."""

import pytest
from src.app.settings import Settings


@pytest.fixture
def test_settings():
    """Create test settings with overrides."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.3:70b",
        log_level="DEBUG"
    )
