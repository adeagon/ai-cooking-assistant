"""Tests for application settings configuration."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_default_ollama_base_url(self):
        """Test default Ollama base URL."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_default_ollama_model(self):
        """Test default Ollama model."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.ollama_model == "qwen3:14b"

    def test_default_ollama_intent_model(self):
        """Test default intent classifier model."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.ollama_intent_model == "qwen3:14b"

    def test_default_ollama_disable_thinking(self):
        """Test default thinking mode is disabled."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.ollama_disable_thinking is True

    def test_default_embedding_model(self):
        """Test default embedding model."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.embedding_model == "all-mpnet-base-v2"

    def test_default_reranker_model(self):
        """Test default reranker model."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_default_data_paths(self):
        """Test default data paths."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.chroma_persist_dir == Path("data/chroma")
        assert settings.sqlite_db_path == Path("data/sqlite/recipes.db")

    def test_default_retrieval_parameters(self):
        """Test default retrieval parameters."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.k_retrieve == 100
        assert settings.k_rerank == 20
        assert settings.k_context == 6

    def test_default_llm_parameters(self):
        """Test default LLM parameters."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.context_length == 8192
        assert settings.llm_temperature == 0.3
        assert settings.llm_max_tokens == 1024
        assert settings.ollama_timeout == 300.0

    def test_default_log_level(self):
        """Test default log level."""
        from src.app.settings import Settings
        settings = Settings()
        assert settings.log_level == "INFO"


class TestSettingsEnvironmentVariables:
    """Tests for loading settings from environment variables."""

    def test_ollama_base_url_from_env(self):
        """Test loading Ollama base URL from environment."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:9000"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.ollama_base_url == "http://custom:9000"

    def test_ollama_model_from_env(self):
        """Test loading Ollama model from environment."""
        with patch.dict(os.environ, {"OLLAMA_MODEL": "cooking-assistant"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.ollama_model == "cooking-assistant"

    def test_k_retrieve_from_env(self):
        """Test loading k_retrieve from environment."""
        with patch.dict(os.environ, {"K_RETRIEVE": "200"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.k_retrieve == 200

    def test_llm_temperature_from_env(self):
        """Test loading LLM temperature from environment."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.7"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.llm_temperature == 0.7

    def test_log_level_from_env(self):
        """Test loading log level from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case-insensitive."""
        with patch.dict(os.environ, {"ollama_model": "test-model"}):
            from src.app.settings import Settings
            settings = Settings()
            assert settings.ollama_model == "test-model"


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_path_fields_are_path_objects(self):
        """Test that path fields are Path objects."""
        from src.app.settings import Settings
        settings = Settings()
        assert isinstance(settings.chroma_persist_dir, Path)
        assert isinstance(settings.sqlite_db_path, Path)

    def test_numeric_fields_are_correct_types(self):
        """Test that numeric fields have correct types."""
        from src.app.settings import Settings
        settings = Settings()
        assert isinstance(settings.k_retrieve, int)
        assert isinstance(settings.k_rerank, int)
        assert isinstance(settings.k_context, int)
        assert isinstance(settings.context_length, int)
        assert isinstance(settings.llm_temperature, float)
        assert isinstance(settings.llm_max_tokens, int)
        assert isinstance(settings.ollama_timeout, float)

    def test_boolean_fields_are_boolean(self):
        """Test that boolean fields are boolean."""
        from src.app.settings import Settings
        settings = Settings()
        assert isinstance(settings.ollama_disable_thinking, bool)


class TestGlobalSettingsInstance:
    """Tests for the global settings instance."""

    def test_global_settings_exists(self):
        """Test that global settings instance is available."""
        from src.app.settings import settings
        assert settings is not None

    def test_global_settings_is_settings_instance(self):
        """Test that global settings is a Settings instance."""
        from src.app.settings import settings, Settings
        assert isinstance(settings, Settings)
