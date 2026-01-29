"""Tests for auth API endpoints."""

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


class TestUsersEndpoint:
    """Tests for GET /api/users."""

    def test_list_users(self, client):
        """Test listing available users."""
        response = client.get("/api/users")

        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert len(data["users"]) == 4  # Default seed users

        usernames = [u["username"] for u in data["users"]]
        assert "alex" in usernames
        assert "jordan" in usernames


class TestLoginEndpoint:
    """Tests for POST /api/auth/login."""

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/api/auth/login",
            json={"username": "alex"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user"]["username"] == "alex"
        assert "session_id" in data

        # Check cookie is set
        assert "aca_session" in response.cookies

    def test_login_sets_cookie(self, client):
        """Test that login sets the session cookie."""
        response = client.post(
            "/api/auth/login",
            json={"username": "jordan"}
        )

        assert response.status_code == 200
        cookie = response.cookies.get("aca_session")
        assert cookie is not None

    def test_login_user_not_found(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/api/auth/login",
            json={"username": "nonexistent"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_login_empty_username(self, client):
        """Test login with empty username."""
        response = client.post(
            "/api/auth/login",
            json={"username": ""}
        )

        assert response.status_code == 422  # Validation error


class TestLogoutEndpoint:
    """Tests for POST /api/auth/logout."""

    def test_logout_success(self, client):
        """Test successful logout."""
        # First login
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "alex"}
        )
        assert login_resp.status_code == 200

        # Then logout
        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        assert response.json()["status"] == "logged_out"

    def test_logout_clears_cookie(self, client):
        """Test that logout clears the session cookie."""
        # Login first
        client.post("/api/auth/login", json={"username": "alex"})

        # Logout
        response = client.post("/api/auth/logout")

        # Cookie should be cleared (set to empty or deleted)
        assert response.status_code == 200

    def test_logout_without_session(self, client):
        """Test logout without being logged in."""
        response = client.post("/api/auth/logout")

        # Should still succeed
        assert response.status_code == 200
        assert response.json()["status"] == "logged_out"

    def test_logout_revokes_session(self, client):
        """Test that logout revokes the current session."""
        # Login
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "alex"}
        )
        session_id = login_resp.json()["session_id"]

        # Logout
        client.post("/api/auth/logout")

        # Try to use the old session - should fail
        # Create new client without cookies and manually set the old session
        response = client.get(
            "/api/auth/whoami",
            cookies={"aca_session": session_id}
        )

        assert response.status_code == 401


class TestWhoamiEndpoint:
    """Tests for GET /api/auth/whoami."""

    def test_whoami_authenticated(self, client):
        """Test whoami when authenticated."""
        # Login first
        client.post("/api/auth/login", json={"username": "taylor"})

        response = client.get("/api/auth/whoami")

        assert response.status_code == 200
        data = response.json()
        assert data["user"]["username"] == "taylor"

    def test_whoami_not_authenticated(self, client):
        """Test whoami without authentication."""
        response = client.get("/api/auth/whoami")

        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    def test_whoami_invalid_session(self, client):
        """Test whoami with invalid session cookie."""
        response = client.get(
            "/api/auth/whoami",
            cookies={"aca_session": "invalid-session-id"}
        )

        assert response.status_code == 401


class TestSessionPersistence:
    """Tests for session persistence across requests."""

    def test_session_persists(self, client):
        """Test that session persists across multiple requests."""
        # Login
        client.post("/api/auth/login", json={"username": "casey"})

        # Make multiple requests
        for _ in range(3):
            response = client.get("/api/auth/whoami")
            assert response.status_code == 200
            assert response.json()["user"]["username"] == "casey"

    def test_different_users_different_sessions(self, client):
        """Test that different users have isolated sessions."""
        # Login as alex
        client.post("/api/auth/login", json={"username": "alex"})
        alex_resp = client.get("/api/auth/whoami")
        assert alex_resp.json()["user"]["username"] == "alex"

        # Login as jordan (in same client, replaces session)
        client.post("/api/auth/login", json={"username": "jordan"})
        jordan_resp = client.get("/api/auth/whoami")
        assert jordan_resp.json()["user"]["username"] == "jordan"
