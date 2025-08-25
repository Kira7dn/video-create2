import os
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, APIRouter
import uvicorn
from contextlib import asynccontextmanager

from app.core.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from app.presentation.api.v1.routers import video
from app.presentation.api.v1.routers import health
from app.core.config import settings


# Configure logging: both to console and to file
log_handlers = [
    logging.StreamHandler(),
    RotatingFileHandler(
        "data/app.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
    ),
]
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    datefmt=settings.log_date_format,
    handlers=log_handlers,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Video Creation API...")
    yield
    logger.info("Shutting down Video Creation API...")


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Add CORS middleware
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=settings.cors_origins,
    #     allow_credentials=settings.cors_allow_credentials,
    #     allow_methods=settings.cors_allow_methods,
    #     allow_headers=settings.cors_allow_headers,
    # )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.max_concurrent_requests,
        period=60,
    )

    # Add exception handlers
    # app.add_exception_handler(RequestValidationError, validation_exception_handler)
    # app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    # app.add_exception_handler(VideoProcessingError, video_processing_exception_handler)
    # app.add_exception_handler(FileValidationError, file_validation_exception_handler)
    # app.add_exception_handler(Exception, general_exception_handler)

    # Include API routers under versioned prefix
    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(video.router, tags=["users"])
    api_v1.include_router(health.router)
    app.include_router(api_v1)

    return app


# Create application instance
app = create_application()

if __name__ == "__main__":
    dev_mode = os.getenv("DEV_MODE", "true").lower() == "true"
    uvicorn.run(
        "app.presentation.main:app",
        host=settings.host,
        port=settings.port,
        reload=dev_mode,
    )
