"""
SQLAlchemy database models for ResumeForge MVP
Simplified no-auth model with anonymous processing jobs
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float, JSON, Index
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class ProcessingStatus(str, Enum):
    """Processing job status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"





class ProcessingJob(Base):
    """
    Anonymous processing job model for ResumeForge MVP
    Tracks resume optimization without user authentication
    Includes 24-hour auto-cleanup for temporary storage
    """
    __tablename__ = "processing_jobs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Unique job identifier for public access
    job_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)

    # Client IP for basic tracking (optional)
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    # Job status
    status: Mapped[str] = mapped_column(
        String(50),
        default=ProcessingStatus.PENDING,
        index=True
    )
    
    # Input files and data
    original_resume_blob_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    job_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    job_description_file_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    job_title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Processing results
    optimized_resume_blob_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    cover_letter_blob_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    package_zip_blob_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # AI processing metadata (JSON fields)
    resume_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ats_keyword_map: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    change_log: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    layout_coordinates: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Performance metrics
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    openai_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now()
    )

    # Auto-cleanup: expires after 24 hours
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.utcnow() + timedelta(hours=24),
        index=True
    )
    
    # Additional metadata
    processing_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    user_feedback: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, job_id='{self.job_id}', status='{self.status}')>"
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed successfully"""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if job has failed"""
        return self.status == ProcessingStatus.FAILED
    
    @property
    def is_processing(self) -> bool:
        """Check if job is currently processing"""
        return self.status == ProcessingStatus.PROCESSING
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_expired(self) -> bool:
        """Check if job has expired (past 24 hours)"""
        return datetime.utcnow() > self.expires_at

    @property
    def time_until_expiry(self) -> timedelta:
        """Get time remaining until expiry"""
        return self.expires_at - datetime.utcnow()

    @classmethod
    def generate_job_id(cls) -> str:
        """Generate a unique job ID"""
        return str(uuid.uuid4())
    
    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert processing job to dictionary
        
        Args:
            include_metadata: Whether to include AI metadata
            
        Returns:
            dict: Job data
        """
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "job_title": self.job_title,
            "company_name": self.company_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_seconds": self.processing_time_seconds,
            "ai_tokens_used": self.ai_tokens_used,
            "retry_count": self.retry_count,
            "is_completed": self.is_completed,
            "is_failed": self.is_failed,
            "is_processing": self.is_processing,
            "duration_seconds": self.duration_seconds,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired,
            "time_until_expiry_hours": self.time_until_expiry.total_seconds() / 3600 if not self.is_expired else 0
        }
        
        # Add file URLs if available
        if self.optimized_resume_blob_url:
            data["optimized_resume_url"] = self.optimized_resume_blob_url
        if self.cover_letter_blob_url:
            data["cover_letter_url"] = self.cover_letter_blob_url
        if self.package_zip_blob_url:
            data["package_url"] = self.package_zip_blob_url
        
        # Add error information if failed
        if self.is_failed:
            data.update({
                "error_message": self.error_message,
                "error_code": self.error_code
            })
        
        # Add metadata if requested
        if include_metadata:
            data.update({
                "resume_analysis": self.resume_analysis,
                "ats_keyword_map": self.ats_keyword_map,
                "change_log": self.change_log,
                "processing_metadata": self.processing_metadata
            })
        
        return data


# Indexes are automatically created by SQLAlchemy from the index=True parameters
