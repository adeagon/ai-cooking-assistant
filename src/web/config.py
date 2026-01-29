"""Web-specific configuration settings."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """Web application configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="WEB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind server to"
    )
    port: int = Field(
        default=8000,
        description="Port to bind server to"
    )

    # Database
    db_path: Path = Field(
        default=Path("data/sqlite/app.db"),
        description="SQLite database path for web app"
    )

    # Session settings
    session_cookie_name: str = Field(
        default="aca_session",
        description="Name of the session cookie"
    )
    session_max_age_days: int = Field(
        default=30,
        description="Session expiry in days"
    )

    # CORS settings
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )


# Global settings instance
web_settings = WebSettings()
