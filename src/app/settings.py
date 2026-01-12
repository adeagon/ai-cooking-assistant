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
        default="llama3.3:70b",
        description="Ollama model name"
    )

    # Embedding and reranking models
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
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
        default=Path("data/sqlite/app.db"),
        description="SQLite database path"
    )

    # Retrieval parameters
    k_retrieve: int = Field(
        default=30,
        description="Number of candidates to retrieve from vector store"
    )
    k_rerank: int = Field(
        default=10,
        description="Number of candidates after reranking"
    )
    k_context: int = Field(
        default=4,
        description="Number of recipe cards to pass to LLM"
    )

    # LLM parameters
    context_length: int = Field(
        default=8192,
        description="Target context window size"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


# Global settings instance
settings = Settings()
