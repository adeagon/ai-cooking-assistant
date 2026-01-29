"""FastAPI dependencies for the web application."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, Request, status

from src.app.logging_config import get_logger
from src.app.settings import settings
from src.memory.store_factory import StoreFactory
from src.web.config import web_settings
from src.web.db import init_db
from src.web.models import User
from src.web.services.chat_service import ChatService
from src.web.services.conversation_service import ConversationService
from src.web.services.session_service import SessionService
from src.web.services.user_service import UserService

logger = get_logger(__name__)


@lru_cache()
def get_db_path() -> Path:
    """Get database path, ensuring it exists."""
    db_path = web_settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path, seed_users=True)
    return db_path


@lru_cache()
def get_user_service() -> UserService:
    """Get singleton UserService instance."""
    return UserService(get_db_path())


@lru_cache()
def get_session_service() -> SessionService:
    """Get singleton SessionService instance."""
    return SessionService(
        get_db_path(),
        max_age_days=web_settings.session_max_age_days
    )


@lru_cache()
def get_conversation_service() -> ConversationService:
    """Get singleton ConversationService instance."""
    return ConversationService(get_db_path())


@lru_cache()
def get_store_factory() -> StoreFactory:
    """Get singleton StoreFactory instance.

    Uses the recipe SQLite database (separate from web DB).
    """
    return StoreFactory(db_path=settings.sqlite_db_path)


@lru_cache()
def get_chat_service() -> ChatService:
    """Get singleton ChatService instance."""
    return ChatService(
        store_factory=get_store_factory(),
        chroma_dir=Path(settings.chroma_persist_dir),
        sqlite_db_path=settings.sqlite_db_path,
    )


async def get_current_user(
    request: Request,
    aca_session: Annotated[str | None, Cookie()] = None,
    session_service: SessionService = Depends(get_session_service),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Get current authenticated user from session cookie.

    Raises:
        HTTPException: 401 if not authenticated or session invalid
    """
    if not aca_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    # Validate session
    session = session_service.get_valid(aca_session)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session"
        )

    # Get user
    user = user_service.get_by_id(session.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Touch session to update last_seen_at
    session_service.touch(aca_session)

    return user


async def get_optional_user(
    request: Request,
    aca_session: Annotated[str | None, Cookie()] = None,
    session_service: SessionService = Depends(get_session_service),
    user_service: UserService = Depends(get_user_service),
) -> User | None:
    """Get current user if authenticated, None otherwise."""
    if not aca_session:
        return None

    session = session_service.get_valid(aca_session)
    if not session:
        return None

    user = user_service.get_by_id(session.user_id)
    if user:
        session_service.touch(aca_session)

    return user


# Type aliases for cleaner route signatures
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
