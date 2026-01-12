"""Smoke tests for CLI application."""

from typer.testing import CliRunner
from src.app.cli import app
from src.app.settings import settings

runner = CliRunner()


def test_cli_help():
    """Test that --help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "recipe-assistant" in result.stdout.lower()


def test_version_command():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()


def test_config_command():
    """Test config command."""
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "configuration" in result.stdout.lower()
    assert settings.ollama_base_url in result.stdout


def test_settings_load():
    """Test that settings load correctly."""
    assert settings.ollama_base_url is not None
    assert settings.ollama_model is not None
    assert settings.k_retrieve > 0
    assert settings.k_rerank > 0
    assert settings.k_context > 0
