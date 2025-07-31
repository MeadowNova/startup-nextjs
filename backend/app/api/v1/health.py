"""
Health check endpoints for ResumeForge API
Simple status monitoring without authentication
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session, get_database_info
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    Returns application status and version
    """
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


@router.get("/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Detailed health check with database and service status
    """
    # Check database connection
    db_info = await get_database_info()
    
    # Check OpenAI configuration
    openai_configured = bool(settings.openai_api_key and settings.openai_api_key.startswith("sk-"))
    
    # Check blob storage configuration
    blob_configured = bool(settings.blob_read_write_token)
    
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "services": {
            "database": db_info,
            "openai": {
                "configured": openai_configured,
                "model_text": settings.openai_model_text,
                "model_vision": settings.openai_model_vision
            },
            "blob_storage": {
                "configured": blob_configured,
                "fallback_enabled": settings.blob_storage_fallback
            }
        },
        "configuration": {
            "max_file_size_mb": settings.max_file_size_mb,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "job_timeout_minutes": settings.job_timeout_minutes,
            "debug": settings.debug
        }
    }
