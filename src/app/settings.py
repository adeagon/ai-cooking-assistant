"""Application settings loaded from environment variables."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="qwen2.5:14b",
        description="Ollama model name"
    )

    # Embedding and reranking models
    embedding_model: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence transformers embedding model"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model"
    )

    # Data paths
    chroma_persist_dir: Path = Field(
        default=Path("data/chroma"),
        description="ChromaDB persistence directory"
    )
    sqlite_db_path: Path = Field(
        default=Path("data/sqlite/recipes.db"),
        description="SQLite database path"
    )

    # Retrieval parameters
    k_retrieve: int = Field(
        default=100,
        description="Number of candidates to retrieve from vector store"
    )
    k_rerank: int = Field(
        default=20,
        description="Number of candidates after reranking"
    )
    k_context: int = Field(
        default=6,
        description="Number of recipe cards to pass to LLM"
    )

    # LLM parameters
    context_length: int = Field(
        default=8192,
        description="Target context window size"
    )
    llm_temperature: float = Field(
        default=0.3,
        description="LLM temperature for generation (0.0 = deterministic, 1.0 = creative)"
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for LLM response"
    )
    ollama_timeout: float = Field(
        default=300.0,
        description="Ollama API timeout in seconds"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


# Global settings instance
settings = Settings()
