"""Flask application configuration."""

import os
from datetime import timedelta
from pathlib import Path


class Config:
    """Base configuration."""

    # Flask core
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # Database
    SQLITE_DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", "data/sqlite/recipes.db"))

    # Rate limiting (per-worker in-memory)
    RATELIMIT_STORAGE_URI = "memory://"
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_HEADERS_ENABLED = True

    # Setup token for first-login password setup
    INITIAL_SETUP_TOKEN = os.environ.get("INITIAL_SETUP_TOKEN")

    # Guest login
    ALLOW_GUEST_LOGIN = os.environ.get("ALLOW_GUEST_LOGIN", "0") == "1"


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development

    # Relaxed rate limits for development
    RATELIMIT_DEFAULT = "1000/hour"

    # Allow first-login without setup token in localhost development
    REQUIRE_SETUP_TOKEN = False


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    SESSION_COOKIE_SECURE = True  # Require HTTPS

    # Stricter rate limits
    RATELIMIT_DEFAULT = "100/hour"

    # Require setup token for first-login password setup
    REQUIRE_SETUP_TOKEN = True

    @classmethod
    def validate(cls):
        """Validate production configuration."""
        if cls.SECRET_KEY == "dev-secret-key-change-in-production":
            raise RuntimeError(
                "SECRET_KEY must be set in production. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

        if not cls.INITIAL_SETUP_TOKEN:
            raise RuntimeError(
                "INITIAL_SETUP_TOKEN must be set in production for first-login password setup."
            )


class LANConfig(Config):
    """LAN-exposed configuration (ALLOW_LAN=1)."""

    DEBUG = False
    SESSION_COOKIE_SECURE = False  # May not have HTTPS on LAN

    # Require setup token when exposed on LAN
    REQUIRE_SETUP_TOKEN = True

    @classmethod
    def validate(cls):
        """Validate LAN configuration."""
        if cls.SECRET_KEY == "dev-secret-key-change-in-production":
            raise RuntimeError(
                "SECRET_KEY must be set when ALLOW_LAN=1. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

        if not cls.INITIAL_SETUP_TOKEN:
            raise RuntimeError(
                "INITIAL_SETUP_TOKEN must be set when ALLOW_LAN=1 for first-login password setup."
            )


def get_config():
    """Get configuration based on environment variables.

    Environment variables:
        ENV: development | production (default: development)
        ALLOW_LAN: 0 | 1 (default: 0)
    """
    env = os.environ.get("ENV", "development").lower()
    allow_lan = os.environ.get("ALLOW_LAN", "0") == "1"

    if env == "production":
        ProductionConfig.validate()
        return ProductionConfig

    if allow_lan:
        LANConfig.validate()
        return LANConfig

    return DevelopmentConfig


def get_bind_host() -> str:
    """Get host to bind to based on ALLOW_LAN setting."""
    allow_lan = os.environ.get("ALLOW_LAN", "0") == "1"
    return "0.0.0.0" if allow_lan else "127.0.0.1"


def get_port() -> int:
    """Get port to bind to."""
    return int(os.environ.get("PORT", "5000"))
