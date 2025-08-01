"""
Resume processing endpoints for ResumeForge MVP
Anonymous file upload and processing without authentication
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import time
import magic
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.database import get_db_session, get_async_session
from app.models.database import ProcessingJob, ProcessingStatus
from app.core.config import settings
from app.services.ai_processor import AIProcessor
from app.services.blob_storage import BlobStorageService
from app.services.pdf_reconstructor import PDFReconstructor
from app.services.package_creator import PackageCreator
from app.services.smoldocling_processor import SmolDoclingProcessor

logger = structlog.get_logger()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# Services will be initialized when needed to avoid startup issues
def get_ai_processor():
    return AIProcessor(settings.openai_api_key)

def get_blob_storage():
    return BlobStorageService(settings.blob_read_write_token)

def get_pdf_reconstructor():
    return PDFReconstructor()

def get_smoldocling_processor():
    return SmolDoclingProcessor()

def get_package_creator():
    return PackageCreator(get_blob_storage())


async def validate_pdf_file(file: UploadFile) -> int:
    """
    Comprehensive PDF file validation with security checks
    Returns file size in bytes
    """
    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read file content for validation
    content = await file.read()
    file_size = len(content)

    # Check file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")

    # Check minimum file size (1KB)
    if file_size < 1024:
        raise HTTPException(status_code=400, detail="File too small. Minimum size: 1KB")

    # Check PDF magic bytes
    if not content.startswith(b'%PDF'):
        raise HTTPException(status_code=400, detail="Invalid PDF file format")

    # Additional security: Check for suspicious content
    content_str = content[:1024].decode('latin-1', errors='ignore').lower()
    suspicious_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
    for pattern in suspicious_patterns:
        if pattern in content_str:
            raise HTTPException(status_code=400, detail="File contains suspicious content")

    # Reset file pointer for subsequent processing
    await file.seek(0)

    logger.info("PDF file validation passed",
                filename=file.filename,
                size_bytes=file_size,
                size_mb=round(file_size / (1024*1024), 2))

    return file_size


def sanitize_text_input(text: str, max_length: int = 50000) -> str:
    """Sanitize and validate text input"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    text = text.strip()

    if len(text) > max_length:
        raise HTTPException(status_code=400, detail=f"Text too long. Maximum length: {max_length} characters")

    # Basic XSS prevention
    suspicious_patterns = ['<script', '</script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if pattern in text_lower:
            raise HTTPException(status_code=400, detail="Text contains suspicious content")

    return text


async def save_uploaded_file_streaming(upload_file: UploadFile, destination_path: str) -> int:
    """
    Save uploaded file with streaming to avoid memory issues
    Returns total bytes written
    """
    total_size = 0
    chunk_size = 8192  # 8KB chunks

    try:
        with open(destination_path, 'wb') as f:
            while chunk := await upload_file.read(chunk_size):
                f.write(chunk)
                total_size += len(chunk)

                # Safety check to prevent extremely large files
                if total_size > 50 * 1024 * 1024:  # 50MB absolute limit
                    raise HTTPException(status_code=400, detail="File too large during streaming")

        logger.info("File saved with streaming",
                   destination=destination_path,
                   total_bytes=total_size,
                   total_mb=round(total_size / (1024*1024), 2))

        return total_size

    except Exception as e:
        # Clean up partial file on error
        try:
            import os
            if os.path.exists(destination_path):
                os.unlink(destination_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@router.post("/process")
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def process_resume(
    background_tasks: BackgroundTasks,
    request: Request,
    resume_file: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
    job_title: Optional[str] = Form(None, description="Job title (optional)"),
    company_name: Optional[str] = Form(None, description="Company name (optional)"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Process resume for ATS optimization with comprehensive validation

    Creates a processing job and starts background processing
    Returns job_id for status tracking
    """

    try:
        # Validate and sanitize inputs
        file_size = await validate_pdf_file(resume_file)
        job_description = sanitize_text_input(job_description, max_length=50000)

        if job_title:
            job_title = sanitize_text_input(job_title, max_length=200)
        if company_name:
            company_name = sanitize_text_input(company_name, max_length=200)

        logger.info("File validation passed",
                   filename=resume_file.filename,
                   file_size_bytes=file_size,
                   file_size_mb=round(file_size / (1024*1024), 2),
                   client_ip=request.client.host if request.client else "unknown",
                   user_agent=request.headers.get("user-agent", "unknown"),
                   timestamp=datetime.utcnow().isoformat())

        # Get client IP for basic tracking
        client_ip = request.client.host if request.client else None
        # Create processing job
        job = ProcessingJob(
            job_id=ProcessingJob.generate_job_id(),
            client_ip=client_ip,
            status=ProcessingStatus.PENDING,
            job_description=job_description,
            job_title=job_title,
            company_name=company_name
        )
        
        db.add(job)
        await db.commit()
        await db.refresh(job)
        
        logger.info(
            "Processing job created",
            job_id=job.job_id,
            client_ip=client_ip,
            filename=resume_file.filename,
            job_description_length=len(job_description),
            has_job_title=bool(job_title),
            has_company_name=bool(company_name),
            timestamp=datetime.utcnow().isoformat()
        )

        # Read file content before starting background task
        resume_file_content = await resume_file.read()
        logger.info("Resume file content read", job_id=job.job_id, file_size=len(resume_file_content))

        # Start background processing
        background_tasks.add_task(
            process_resume_background,
            job.job_id,
            resume_file_content,
            job_description,
            job_title,
            company_name
        )
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": "Resume processing started",
            "estimated_time_minutes": 2,
            "expires_at": job.expires_at.isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to create processing job", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start processing")


async def process_resume_background(
    job_id: str,
    resume_file_content: bytes,
    job_description: str,
    job_title: Optional[str],
    company_name: Optional[str]
):
    """
    Background task for resume processing
    Implements the complete AI optimization pipeline
    """

    logger.info("🚀 BACKGROUND TASK STARTED", job_id=job_id)

    try:
        logger.info("📊 Acquiring database session", job_id=job_id)
        async with get_async_session() as db:
            try:
                logger.info("🔍 Looking up job in database", job_id=job_id)
                # Get the job
                result = await db.execute(
                    select(ProcessingJob).where(ProcessingJob.job_id == job_id)
                )
                job = result.scalar_one_or_none()

                if not job:
                    logger.error("❌ Job not found for background processing", job_id=job_id)
                    return

                logger.info("✅ Job found, updating status to processing", job_id=job_id)

                # Update status to processing
                job.status = ProcessingStatus.PROCESSING
                job.started_at = datetime.utcnow()
                await db.commit()

                logger.info("Starting resume processing", job_id=job_id)

                # Step 1: Upload original resume to blob storage
                blob_storage = get_blob_storage()
                original_url = await blob_storage.upload_bytes(
                    resume_file_content,
                    f"resumes/original/{job_id}.pdf",
                    "application/pdf"
                )
                job.original_resume_blob_url = original_url

                # Save resume to temporary file for processing
                pdf_reconstructor = get_pdf_reconstructor()
                temp_pdf_path = pdf_reconstructor.temp_dir / f"{job_id}_original.pdf"
                with open(temp_pdf_path, 'wb') as f:
                    f.write(resume_file_content)

                # Step 2: Extract text from PDF using PyMuPDF
                resume_text = pdf_reconstructor.pdf_to_text(temp_pdf_path)
                logger.info("PDF text extraction completed", characters=len(resume_text))

                # Step 3: Convert first page to image for layout analysis
                image_path = pdf_reconstructor.first_page_to_png(temp_pdf_path)
                logger.info("PDF to image conversion completed", image_path=str(image_path))

                # Step 4: Extract layout coordinates using SmolDocling
                smoldocling = get_smoldocling_processor()
                layout_coords = {}

                if smoldocling.is_available():
                    try:
                        layout_coords = smoldocling.extract_layout_coordinates(image_path)
                        logger.info("SmolDocling layout extraction completed")
                    except Exception as e:
                        logger.warning("SmolDocling failed, using fallback layout", error=str(e))
                        # Get image dimensions for fallback
                        from PIL import Image
                        with Image.open(image_path) as img:
                            image_size = img.size
                        layout_coords = smoldocling._create_fallback_layout(image_size)
                else:
                    logger.warning("SmolDocling not available, using fallback layout")
                    layout_coords = {"text_blocks": [], "image_size": (612, 792), "extraction_method": "fallback"}

                # Step 5: AI optimization using simple prompts
                ai_processor = get_ai_processor()
                optimized_resume_markdown = await ai_processor.optimize_resume_text(
                    resume_text,
                    job_description
                )
                logger.info("Resume optimization completed")

                # Step 6: Generate cover letter
                cover_letter_markdown = await ai_processor.generate_cover_letter(
                    optimized_resume_markdown,
                    job_description,
                    job_title,
                    company_name
                )
                logger.info("Cover letter generation completed")

                # Step 7: Reconstruct PDF with layout preservation
                optimized_pdf_path = pdf_reconstructor.reconstruct_pdf_with_smoldocling(
                    optimized_resume_markdown,
                    layout_coords,
                    f"{job_id}_optimized.pdf"
                )

                # Step 8: Create cover letter PDF
                cover_letter_pdf_path = pdf_reconstructor._create_fallback_pdf(
                    cover_letter_markdown,
                    f"{job_id}_cover_letter.pdf"
                )

                # Read PDF files for upload
                with open(optimized_pdf_path, 'rb') as f:
                    optimized_pdf_content = f.read()
                with open(cover_letter_pdf_path, 'rb') as f:
                    cover_letter_pdf_content = f.read()

                # Step 6: Upload processed files
                blob_storage = get_blob_storage()
                optimized_url = await blob_storage.upload_bytes(
                    optimized_pdf_content,
                    f"resumes/optimized/{job_id}.pdf",
                    "application/pdf"
                )

                cover_letter_url = await blob_storage.upload_bytes(
                    cover_letter_pdf_content,
                    f"cover_letters/{job_id}.pdf",
                    "application/pdf"
                )

                # Step 7: Create package
                package_creator = get_package_creator()
                package_path = await package_creator.create_simple_package(
                    optimized_pdf_path,
                    cover_letter_pdf_path,
                    f"{job_id}_package.zip"
                )

                # Read package content for upload
                with open(package_path, 'rb') as f:
                    package_content = f.read()

                package_url = await blob_storage.upload_bytes(
                    package_content,
                    f"packages/{job_id}.zip",
                    "application/zip"
                )

                # Update job with results
                job.optimized_resume_blob_url = optimized_url
                job.cover_letter_blob_url = cover_letter_url
                job.package_zip_blob_url = package_url
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()

                # Store the optimized content for display
                job.optimized_resume_markdown = optimized_resume_markdown
                job.cover_letter_text = cover_letter_markdown

                # Store layout analysis results
                job.resume_analysis = {
                    "layout_method": layout_coords.get("extraction_method", "unknown"),
                    "blocks_found": len(layout_coords.get("text_blocks", [])),
                    "image_size": layout_coords.get("image_size", [0, 0]),
                    "smoldocling_available": smoldocling.is_available()
                }

                # Clean up temporary files
                try:
                    temp_pdf_path.unlink(missing_ok=True)
                    image_path.unlink(missing_ok=True)
                    optimized_pdf_path.unlink(missing_ok=True)
                    cover_letter_pdf_path.unlink(missing_ok=True)
                    package_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Failed to clean up temporary files", error=str(e))

                await db.commit()

                logger.info(
                    "Resume processing completed successfully",
                    job_id=job_id,
                    processing_time=job.processing_time_seconds
                )

            except Exception as e:
                logger.error(
                    "Resume processing failed",
                    job_id=job_id,
                    error=str(e),
                    exc_info=True
                )

                # Update job with error
                job.status = ProcessingStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await db.commit()

    except Exception as e:
        logger.error("🚨 BACKGROUND TASK FAILED AT TOP LEVEL", job_id=job_id, error=str(e), exc_info=True)
        # Try to update job status to failed
        try:
            async with get_async_session() as db:
                result = await db.execute(
                    select(ProcessingJob).where(ProcessingJob.job_id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = f"Background task failed: {str(e)}"
                    job.completed_at = datetime.utcnow()
                    await db.commit()
        except Exception as db_error:
            logger.error("Failed to update job status after background task failure", error=str(db_error))


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get processing job status
    """
    
    result = await db.execute(
        select(ProcessingJob).where(ProcessingJob.job_id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.is_expired:
        raise HTTPException(status_code=410, detail="Job has expired")
    
    return job.to_dict(include_metadata=False)


@router.get("/result/{job_id}")
async def get_job_result(
    job_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    Get complete job result with download URLs
    """
    
    result = await db.execute(
        select(ProcessingJob).where(ProcessingJob.job_id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.is_expired:
        raise HTTPException(status_code=410, detail="Job has expired")
    
    if not job.is_completed:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return job.to_dict(include_metadata=True)
