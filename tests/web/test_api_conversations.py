"""Tests for conversations API endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.web.app import create_app
from src.web.config import web_settings
from src.web.db import init_db


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    init_db(db_path, seed_users=True)

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
        for suffix in ["-wal", "-shm"]:
            p = Path(str(db_path) + suffix)
            if p.exists():
                os.unlink(p)
    except Exception:
        pass


@pytest.fixture
def client(test_db_path):
    """Create a test client with patched database path."""
    with patch.object(web_settings, 'db_path', test_db_path):
        # Clear cached dependencies
        from src.web import dependencies
        dependencies.get_db_path.cache_clear()
        dependencies.get_user_service.cache_clear()
        dependencies.get_session_service.cache_clear()
        dependencies.get_conversation_service.cache_clear()

        app = create_app()
        with TestClient(app) as client:
            yield client


@pytest.fixture
def authenticated_client(client):
    """Client that is already logged in."""
    client.post("/api/auth/login", json={"username": "alex"})
    return client


class TestListConversations:
    """Tests for GET /api/conversations."""

    def test_list_empty(self, authenticated_client):
        """Test listing conversations when none exist."""
        response = authenticated_client.get("/api/conversations")

        assert response.status_code == 200
        data = response.json()
        assert data["conversations"] == []

    def test_list_conversations(self, authenticated_client):
        """Test listing user's conversations."""
        # Create some conversations
        authenticated_client.post(
            "/api/conversations",
            json={"title": "First Chat"}
        )
        authenticated_client.post(
            "/api/conversations",
            json={"title": "Second Chat"}
        )

        response = authenticated_client.get("/api/conversations")

        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) == 2

    def test_list_requires_auth(self, client):
        """Test that listing conversations requires authentication."""
        response = client.get("/api/conversations")

        assert response.status_code == 401


class TestCreateConversation:
    """Tests for POST /api/conversations."""

    def test_create_with_title(self, authenticated_client):
        """Test creating conversation with title."""
        response = authenticated_client.post(
            "/api/conversations",
            json={"title": "My New Chat"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "My New Chat"
        assert "id" in data
        assert "created_at" in data

    def test_create_without_title(self, authenticated_client):
        """Test creating conversation without title."""
        response = authenticated_client.post(
            "/api/conversations",
            json={}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] is None

    def test_create_requires_auth(self, client):
        """Test that creating conversation requires authentication."""
        response = client.post(
            "/api/conversations",
            json={"title": "Test"}
        )

        assert response.status_code == 401


class TestGetConversation:
    """Tests for GET /api/conversations/{id}."""

    def test_get_conversation(self, authenticated_client):
        """Test getting a specific conversation."""
        # Create first
        create_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "Test Chat"}
        )
        conv_id = create_resp.json()["id"]

        # Get it
        response = authenticated_client.get(f"/api/conversations/{conv_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv_id
        assert data["title"] == "Test Chat"

    def test_get_not_found(self, authenticated_client):
        """Test getting non-existent conversation."""
        response = authenticated_client.get("/api/conversations/nonexistent-id")

        assert response.status_code == 404

    def test_get_other_users_conversation(self, test_db_path):
        """Test that users can't access other users' conversations."""
        with patch.object(web_settings, 'db_path', test_db_path):
            from src.web import dependencies
            dependencies.get_db_path.cache_clear()
            dependencies.get_user_service.cache_clear()
            dependencies.get_session_service.cache_clear()
            dependencies.get_conversation_service.cache_clear()

            app = create_app()

            with TestClient(app) as client1:
                # Alex creates a conversation
                client1.post("/api/auth/login", json={"username": "alex"})
                create_resp = client1.post(
                    "/api/conversations",
                    json={"title": "Alex's Chat"}
                )
                conv_id = create_resp.json()["id"]

            with TestClient(app) as client2:
                # Jordan tries to access it
                client2.post("/api/auth/login", json={"username": "jordan"})
                response = client2.get(f"/api/conversations/{conv_id}")

                assert response.status_code == 404


class TestGetMessages:
    """Tests for GET /api/conversations/{id}/messages."""

    def test_get_messages_empty(self, authenticated_client):
        """Test getting messages from empty conversation."""
        create_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "Empty Chat"}
        )
        conv_id = create_resp.json()["id"]

        response = authenticated_client.get(
            f"/api/conversations/{conv_id}/messages"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conv_id
        assert data["messages"] == []

    def test_get_messages_not_found(self, authenticated_client):
        """Test getting messages from non-existent conversation."""
        response = authenticated_client.get(
            "/api/conversations/nonexistent-id/messages"
        )

        assert response.status_code == 404


class TestArchiveConversation:
    """Tests for POST /api/conversations/{id}/archive."""

    def test_archive_conversation(self, authenticated_client):
        """Test archiving a conversation."""
        # Create
        create_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "To Archive"}
        )
        conv_id = create_resp.json()["id"]

        # Archive
        response = authenticated_client.post(
            f"/api/conversations/{conv_id}/archive"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "archived"

        # Should no longer be accessible
        get_resp = authenticated_client.get(f"/api/conversations/{conv_id}")
        assert get_resp.status_code == 404

    def test_archive_not_found(self, authenticated_client):
        """Test archiving non-existent conversation."""
        response = authenticated_client.post(
            "/api/conversations/nonexistent-id/archive"
        )

        assert response.status_code == 404

    def test_archive_removes_from_list(self, authenticated_client):
        """Test that archived conversations don't appear in list."""
        # Create two conversations
        c1_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "Keep Me"}
        )
        c2_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "Archive Me"}
        )
        c2_id = c2_resp.json()["id"]

        # Archive one
        authenticated_client.post(f"/api/conversations/{c2_id}/archive")

        # List should only have one
        list_resp = authenticated_client.get("/api/conversations")
        convs = list_resp.json()["conversations"]

        assert len(convs) == 1
        assert convs[0]["title"] == "Keep Me"


class TestUpdateTitle:
    """Tests for PATCH /api/conversations/{id}/title."""

    def test_update_title(self, authenticated_client):
        """Test updating conversation title."""
        create_resp = authenticated_client.post(
            "/api/conversations",
            json={"title": "Old Title"}
        )
        conv_id = create_resp.json()["id"]

        response = authenticated_client.patch(
            f"/api/conversations/{conv_id}/title",
            params={"title": "New Title"}
        )

        assert response.status_code == 200
        assert response.json()["title"] == "New Title"

        # Verify it was updated
        get_resp = authenticated_client.get(f"/api/conversations/{conv_id}")
        assert get_resp.json()["title"] == "New Title"

    def test_update_title_not_found(self, authenticated_client):
        """Test updating non-existent conversation."""
        response = authenticated_client.patch(
            "/api/conversations/nonexistent-id/title",
            params={"title": "New Title"}
        )

        assert response.status_code == 404
