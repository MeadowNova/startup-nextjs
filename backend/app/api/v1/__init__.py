"""
API v1 router configuration for ResumeForge MVP
Simple no-auth API with health checks and resume processing
"""

from fastapi import APIRouter

from .health import router as health_router
from .resume import router as resume_router

# Create main API router
api_router = APIRouter()

# Include health check router
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["health"]
)

# Include resume processing router
api_router.include_router(
    resume_router,
    prefix="/resume",
    tags=["resume"]
)
