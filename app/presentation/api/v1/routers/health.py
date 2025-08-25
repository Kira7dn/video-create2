"""
Health check API endpoints
"""

from fastapi import APIRouter
from app.core.monitoring import health_checker, SystemHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=SystemHealth)
async def health_check():
    """
    Health check endpoint that returns system status and metrics
    """
    return health_checker.get_system_health()


@router.get("/")
async def root():
    """
    Root endpoint
    """
    return {"message": "Video Creation API is running", "status": "healthy"}
