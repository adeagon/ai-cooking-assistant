"""Playwright E2E test fixtures."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure browser context for tests."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def test_server():
    """Start a test server for E2E tests.

    This fixture is session-scoped, meaning it starts once for all tests.
    """
    # Skip if Playwright not installed
    pytest.importorskip("playwright")

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Set environment variables for test server
    env = os.environ.copy()
    env["WEB_DB_PATH"] = str(db_path)
    env["WEB_PORT"] = "8765"  # Use different port for tests

    # Start server
    server_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.web.app:app",
            "--host", "127.0.0.1",
            "--port", "8765",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    yield {
        "url": "http://127.0.0.1:8765",
        "db_path": db_path,
    }

    # Cleanup
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

    # Remove temp database
    try:
        os.unlink(db_path)
        for suffix in ["-wal", "-shm"]:
            p = Path(str(db_path) + suffix)
            if p.exists():
                os.unlink(p)
    except Exception:
        pass
