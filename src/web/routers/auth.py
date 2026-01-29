"""Authentication endpoints."""

from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status

from src.app.logging_config import get_logger
from src.web.config import web_settings
from src.web.dependencies import (
    CurrentUser,
    SessionServiceDep,
    UserServiceDep,
)
from src.web.models import (
    LoginRequest,
    LoginResponse,
    User,
    UserListResponse,
    WhoamiResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["auth"])


@router.get("/users", response_model=UserListResponse)
async def list_users(user_service: UserServiceDep) -> UserListResponse:
    """List all available users for login selection."""
    users = user_service.get_all()
    return UserListResponse(users=users)


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    user_service: UserServiceDep,
    session_service: SessionServiceDep,
) -> LoginResponse:
    """Login with username.

    Creates a new session and sets the session cookie.
    """
    # Find user by username
    user = user_service.get_by_username(body.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{body.username}' not found"
        )

    # Get client info for debugging
    user_agent = request.headers.get("user-agent")
    client_ip = request.client.host if request.client else None

    # Create session
    session = session_service.create(
        user_id=user.id,
        user_agent=user_agent,
        ip=client_ip
    )

    # Set session cookie
    response.set_cookie(
        key=web_settings.session_cookie_name,
        value=session.session_id,
        max_age=web_settings.session_max_age_days * 86400,  # days to seconds
        path="/",
        samesite="lax",
        httponly=True,
        secure=False,  # HTTP on LAN
    )

    logger.info(
        "User logged in",
        username=user.username,
        session_id=session.session_id[:8]
    )

    return LoginResponse(user=user, session_id=session.session_id)


@router.post("/auth/logout")
async def logout(
    response: Response,
    aca_session: Annotated[str | None, Cookie()] = None,
    session_service: SessionServiceDep = None,
) -> dict:
    """Logout current session.

    Revokes the current session (sets revoked_at) and clears the cookie.
    Other sessions for the same user remain active.
    """
    if aca_session:
        session_service.revoke(aca_session)
        logger.info("Session revoked", session_id=aca_session[:8])

    # Clear cookie
    response.delete_cookie(
        key=web_settings.session_cookie_name,
        path="/"
    )

    return {"status": "logged_out"}


@router.get("/auth/whoami", response_model=WhoamiResponse)
async def whoami(user: CurrentUser) -> WhoamiResponse:
    """Get current authenticated user."""
    return WhoamiResponse(user=user)
