"""Pytest fixtures for web application tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.web.db import init_db, get_db_connection
from src.web.services.user_service import UserService
from src.web.services.session_service import SessionService
from src.web.services.conversation_service import ConversationService


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Initialize database schema
    init_db(db_path, seed_users=False)

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
        # Also remove WAL and SHM files if they exist
        wal_path = Path(str(db_path) + "-wal")
        shm_path = Path(str(db_path) + "-shm")
        if wal_path.exists():
            os.unlink(wal_path)
        if shm_path.exists():
            os.unlink(shm_path)
    except Exception:
        pass


@pytest.fixture
def user_service(temp_db: Path) -> UserService:
    """Create a UserService with test database."""
    return UserService(temp_db)


@pytest.fixture
def session_service(temp_db: Path) -> SessionService:
    """Create a SessionService with test database."""
    return SessionService(temp_db, max_age_days=30)


@pytest.fixture
def conversation_service(temp_db: Path) -> ConversationService:
    """Create a ConversationService with test database."""
    return ConversationService(temp_db)


@pytest.fixture
def test_user(user_service: UserService):
    """Create a test user."""
    return user_service.create(username="testuser", display_name="Test User")


@pytest.fixture
def test_session(session_service: SessionService, test_user):
    """Create a test session for the test user."""
    return session_service.create(
        user_id=test_user.id,
        user_agent="pytest-test-agent",
        ip="127.0.0.1"
    )


@pytest.fixture
def test_conversation(conversation_service: ConversationService, test_user):
    """Create a test conversation."""
    conv_id = conversation_service.create(test_user.id, title="Test Conversation")
    return conversation_service.get(conv_id, test_user.id)
