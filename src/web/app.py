"""FastAPI application factory for the cooking assistant web app."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.app.logging_config import configure_logging, get_logger
from src.app.settings import settings
from src.web.config import web_settings
from src.web.db import cleanup_expired_sessions, init_db
from src.web.routers import (
    auth_router,
    chat_router,
    conversations_router,
    health_router,
)

# Configure logging
configure_logging(settings.log_level)
logger = get_logger(__name__)

# Paths
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting web application")

    # Initialize database
    db_path = web_settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path, seed_users=True)
    logger.info("Database initialized", db_path=str(db_path))

    # Cleanup expired sessions
    cleaned = cleanup_expired_sessions(
        db_path,
        max_age_days=web_settings.session_max_age_days
    )
    if cleaned:
        logger.info("Cleaned up expired sessions", count=cleaned)

    yield

    # Shutdown
    logger.info("Shutting down web application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Cooking Assistant",
        description="Local recipe recommendation powered by RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=web_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.include_router(chat_router)

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Setup templates
    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=TEMPLATES_DIR)

        @app.get("/")
        async def index(request):
            """Serve the main chat page."""
            return templates.TemplateResponse("index.html", {"request": request})

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web.app:app",
        host=web_settings.host,
        port=web_settings.port,
        reload=True,
    )
