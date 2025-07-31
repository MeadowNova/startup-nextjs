"""
Configuration management for ResumeForge
Handles environment variables, API keys, and application settings
"""

import os
from typing import Optional, List
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import structlog

logger = structlog.get_logger()


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    Uses Pydantic for validation and type conversion
    """
    
    # Application Settings
    app_name: str = Field(default="ResumeForge", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model_text: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL_TEXT")
    openai_model_vision: str = Field(default="gpt-4o", env="OPENAI_MODEL_VISION")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.25, env="OPENAI_TEMPERATURE")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")
    
    # Blob Storage Configuration
    blob_read_write_token: Optional[str] = Field(default=None, env="BLOB_READ_WRITE_TOKEN")
    blob_storage_url: str = Field(default="https://blob.vercel-storage.com", env="BLOB_STORAGE_URL")
    blob_storage_fallback: bool = Field(default=True, env="BLOB_STORAGE_FALLBACK")
    
    # File Processing Settings
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(
        default=["application/pdf", "image/png", "image/jpeg", "application/zip"],
        env="ALLOWED_FILE_TYPES"
    )
    temp_dir: str = Field(default="/tmp/resumeforge", env="TEMP_DIR")
    cleanup_temp_files: bool = Field(default=True, env="CLEANUP_TEMP_FILES")
    
    # Processing Limits
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    job_timeout_minutes: int = Field(default=10, env="JOB_TIMEOUT_MINUTES")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Security Settings
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # Firebase Authentication (optional)
    firebase_project_id: Optional[str] = Field(default=None, env="FIREBASE_PROJECT_ID")
    firebase_private_key: Optional[str] = Field(default=None, env="FIREBASE_PRIVATE_KEY")
    firebase_client_email: Optional[str] = Field(default=None, env="FIREBASE_CLIENT_EMAIL")
    
    # Monitoring & Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # PDF Processing Settings
    pdf_dpi: int = Field(default=150, env="PDF_DPI")
    pdf_max_pages: int = Field(default=5, env="PDF_MAX_PAGES")
    pdf_quality: int = Field(default=85, env="PDF_QUALITY")
    
    # AI Processing Settings
    ai_retry_attempts: int = Field(default=3, env="AI_RETRY_ATTEMPTS")
    ai_retry_delay: float = Field(default=1.0, env="AI_RETRY_DELAY")
    ai_max_context_length: int = Field(default=8000, env="AI_MAX_CONTEXT_LENGTH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v.startswith(("postgresql://", "postgres://", "sqlite://", "sqlite+aiosqlite://")):
            raise ValueError("Database URL must be a PostgreSQL or SQLite connection string")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format"""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("allowed_file_types", mode="before")
    @classmethod
    def parse_allowed_file_types(cls, v):
        """Parse allowed file types from string or list"""
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(",")]
        return v

    @field_validator("temp_dir")
    @classmethod
    def create_temp_dir(cls, v):
        """Ensure temp directory exists"""
        temp_path = Path(v)
        temp_path.mkdir(parents=True, exist_ok=True)
        return str(temp_path)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() in ("production", "prod")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() in ("development", "dev")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def job_timeout_seconds(self) -> int:
        """Convert job timeout to seconds"""
        return self.job_timeout_minutes * 60
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration dictionary"""
        return {
            "api_key": self.openai_api_key,
            "model_text": self.openai_model_text,
            "model_vision": self.openai_model_vision,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
            "timeout": self.openai_timeout
        }
    
    def get_database_config(self) -> dict:
        """Get database configuration dictionary"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow
        }
    
    def get_blob_storage_config(self) -> dict:
        """Get blob storage configuration dictionary"""
        return {
            "token": self.blob_read_write_token,
            "url": self.blob_storage_url,
            "fallback": self.blob_storage_fallback
        }


class DevelopmentSettings(Settings):
    """Development-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000", "*"]  # Allow all origins in development


class ProductionSettings(Settings):
    """Production-specific settings"""
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 4
    
    @field_validator("secret_key")
    @classmethod
    def validate_production_secret_key(cls, v):
        """Ensure secret key is secure in production"""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters in production")
        return v


class TestSettings(Settings):
    """Test-specific settings"""
    debug: bool = True
    database_url: str = "postgresql://test:test@localhost/test_resumeforge"
    openai_api_key: str = "sk-test-key-for-testing"
    blob_read_write_token: Optional[str] = None
    cleanup_temp_files: bool = True


def get_settings() -> Settings:
    """
    Get application settings based on environment
    
    Returns:
        Settings: Configured settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment in ("production", "prod"):
        settings = ProductionSettings()
    elif environment in ("test", "testing"):
        settings = TestSettings()
    else:
        settings = DevelopmentSettings()
    
    logger.info("Settings loaded", 
               environment=settings.environment,
               debug=settings.debug,
               app_name=settings.app_name)
    
    return settings


# Global settings instance
settings = get_settings()


def validate_required_settings():
    """
    Validate that all required settings are present
    Raises ValueError if critical settings are missing
    """
    required_settings = [
        ("DATABASE_URL", settings.database_url),
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("SECRET_KEY", settings.secret_key)
    ]
    
    missing_settings = []
    for name, value in required_settings:
        if not value:
            missing_settings.append(name)
    
    if missing_settings:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_settings)}")
    
    logger.info("All required settings validated successfully")


def log_settings_summary():
    """Log a summary of current settings (without sensitive data)"""
    logger.info("Application configuration summary",
               app_name=settings.app_name,
               environment=settings.environment,
               debug=settings.debug,
               host=settings.host,
               port=settings.port,
               openai_model_text=settings.openai_model_text,
               openai_model_vision=settings.openai_model_vision,
               max_file_size_mb=settings.max_file_size_mb,
               max_concurrent_jobs=settings.max_concurrent_jobs,
               blob_storage_enabled=bool(settings.blob_read_write_token))


# Validate settings on import
try:
    validate_required_settings()
    log_settings_summary()
except ValueError as e:
    logger.error("Settings validation failed", error=str(e))
    # Don't raise in import to allow for testing scenarios
