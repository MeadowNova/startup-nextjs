"""
Resume processing endpoints for ResumeForge MVP
Anonymous file upload and processing without authentication
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from app.database import get_db_session
from app.models.database import ProcessingJob, ProcessingStatus
from app.core.config import settings
from app.services.ai_processor import AIProcessor
from app.services.blob_storage import BlobStorageService
from app.services.pdf_reconstructor import PDFReconstructor
from app.services.package_creator import PackageCreator
from app.services.smoldocling_processor import SmolDoclingProcessor

logger = structlog.get_logger()
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


@router.post("/process")
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
    Process resume for ATS optimization
    
    Creates a processing job and starts background processing
    Returns job_id for status tracking
    """
    
    # Validate file type
    if not resume_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    if resume_file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )
    
    # Get client IP for basic tracking
    client_ip = request.client.host if request.client else None
    
    try:
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
            job_title=job_title,
            company_name=company_name
        )
        
        # Start background processing
        background_tasks.add_task(
            process_resume_background,
            job.job_id,
            resume_file,
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
    resume_file: UploadFile,
    job_description: str,
    job_title: Optional[str],
    company_name: Optional[str]
):
    """
    Background task for resume processing
    Implements the complete AI optimization pipeline
    """
    
    async with get_db_session() as db:
        try:
            # Get the job
            result = await db.execute(
                select(ProcessingJob).where(ProcessingJob.job_id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                logger.error("Job not found for background processing", job_id=job_id)
                return
            
            # Update status to processing
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await db.commit()
            
            logger.info("Starting resume processing", job_id=job_id)
            
            # Step 1: Upload original resume to blob storage
            blob_storage = get_blob_storage()
            resume_content = await resume_file.read()
            original_url = await blob_storage.upload_file(
                resume_content,
                f"resumes/original/{job_id}.pdf",
                "application/pdf"
            )
            job.original_resume_blob_url = original_url

            # Save resume to temporary file for processing
            pdf_reconstructor = get_pdf_reconstructor()
            temp_pdf_path = pdf_reconstructor.temp_dir / f"{job_id}_original.pdf"
            with open(temp_pdf_path, 'wb') as f:
                f.write(resume_content)

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
                    layout_coords = smoldocling._create_fallback_layout(image_path.stat().st_size)
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
            optimized_url = await blob_storage.upload_file(
                optimized_pdf_content,
                f"resumes/optimized/{job_id}.pdf",
                "application/pdf"
            )

            cover_letter_url = await blob_storage.upload_file(
                cover_letter_pdf_content,
                f"cover_letters/{job_id}.pdf",
                "application/pdf"
            )

            # Step 7: Create package
            package_creator = get_package_creator()
            package_content = await package_creator.create_package(
                optimized_pdf_content,
                cover_letter_pdf_content,
                job_title or "Position",
                company_name or "Company"
            )

            package_url = await blob_storage.upload_file(
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
