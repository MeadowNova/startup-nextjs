"""
Database connection and session management for ResumeForge
Handles SQLAlchemy engine, session factory, and dependency injection
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from app.core.config import settings

logger = structlog.get_logger()

# Create the declarative base for models
Base = declarative_base()

# Database engines and session makers
engine: Optional[create_async_engine] = None
async_session_maker: Optional[async_sessionmaker] = None
sync_engine: Optional[create_engine] = None
sync_session_maker: Optional[sessionmaker] = None


def get_database_url(async_mode: bool = True) -> str:
    """
    Get database URL with appropriate driver for async/sync mode
    
    Args:
        async_mode: Whether to return async-compatible URL
        
    Returns:
        str: Database URL with correct driver
    """
    url = settings.database_url
    
    if async_mode:
        # Convert to async driver
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    else:
        # Ensure sync driver
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    
    return url


def create_database_engine():
    """
    Create database engines and session makers
    Sets up both async and sync engines for different use cases
    """
    global engine, async_session_maker, sync_engine, sync_session_maker
    
    try:
        # Async engine for main application
        async_url = get_database_url(async_mode=True)

        # Configure engine parameters based on database type
        engine_kwargs = {
            "echo": settings.debug,
        }

        if "sqlite" in async_url:
            # SQLite-specific configuration
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False}
            })
        else:
            # PostgreSQL-specific configuration
            engine_kwargs.update({
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_pre_ping": True
            })

        engine = create_async_engine(async_url, **engine_kwargs)
        
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        # Sync engine for migrations and admin tasks
        sync_url = get_database_url(async_mode=False)

        # Configure sync engine parameters based on database type
        sync_engine_kwargs = {
            "echo": settings.debug,
        }

        if "sqlite" in sync_url:
            # SQLite-specific configuration
            sync_engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False}
            })
        else:
            # PostgreSQL-specific configuration
            sync_engine_kwargs.update({
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_pre_ping": True
            })

        sync_engine = create_engine(sync_url, **sync_engine_kwargs)
        
        sync_session_maker = sessionmaker(
            sync_engine,
            autoflush=True,
            autocommit=False
        )
        
        logger.info(
            "Database engines created successfully",
            async_url=async_url.split("@")[-1] if "@" in async_url else async_url,  # Hide credentials
            sync_url=sync_url.split("@")[-1] if "@" in sync_url else sync_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow
        )
        
    except Exception as e:
        logger.error("Failed to create database engines", error=str(e))
        raise


async def create_tables():
    """
    Create all database tables
    Used during application startup
    """
    if not engine:
        create_database_engine()
    
    try:
        # Import models to ensure they're registered
        from app.models.database import ProcessingJob
        
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created/verified successfully")
        
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


async def check_database_connection() -> bool:
    """
    Check if database connection is working
    
    Returns:
        bool: True if connection is successful
    """
    if not engine:
        create_database_engine()
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
        
        logger.info("Database connection check successful")
        return True
        
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions
    
    Yields:
        AsyncSession: Database session
    """
    if not async_session_maker:
        create_database_engine()
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions
    
    Yields:
        AsyncSession: Database session for dependency injection
    """
    if not async_session_maker:
        create_database_engine()
    
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """
    Get synchronous database session
    Used for migrations and admin tasks
    
    Returns:
        Session: Synchronous database session
    """
    if not sync_session_maker:
        create_database_engine()
    
    return sync_session_maker()


async def close_database_connections():
    """
    Close all database connections
    Used during application shutdown
    """
    global engine, sync_engine
    
    try:
        if engine:
            await engine.dispose()
            logger.info("Async database engine disposed")
        
        if sync_engine:
            sync_engine.dispose()
            logger.info("Sync database engine disposed")
            
    except Exception as e:
        logger.error("Error closing database connections", error=str(e))


# Database health check utilities
async def get_database_info() -> dict:
    """
    Get database information for health checks
    
    Returns:
        dict: Database connection information
    """
    if not engine:
        return {"status": "not_initialized"}
    
    try:
        async with engine.begin() as conn:
            # Get database version
            result = await conn.execute(text("SELECT version()"))
            version = await result.fetchone()
            
            # Get connection count (PostgreSQL specific)
            try:
                result = await conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                connections = await result.fetchone()
                connection_count = connections[0] if connections else None
            except:
                connection_count = None
        
        return {
            "status": "connected",
            "database_version": version[0] if version else "unknown",
            "connection_count": connection_count,
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow
        }
        
    except Exception as e:
        logger.error("Failed to get database info", error=str(e))
        return {
            "status": "error",
            "error": str(e)
        }


# Initialize database on module import
try:
    create_database_engine()
except Exception as e:
    logger.warning("Database engine creation deferred", error=str(e))
