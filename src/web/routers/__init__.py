"""Web API routers."""

from src.web.routers.auth import router as auth_router
from src.web.routers.chat import router as chat_router
from src.web.routers.conversations import router as conversations_router
from src.web.routers.health import router as health_router

__all__ = [
    "auth_router",
    "chat_router",
    "conversations_router",
    "health_router",
]
