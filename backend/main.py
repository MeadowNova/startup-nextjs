#!/usr/bin/env python3
"""
Pixel-Perfect Resume Optimizer API
Uses GPT-4o Vision for exact layout recreation with optimized content
This is what you wanted - true layout preservation!
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Import our services
from app.services.ai_processor import AIProcessor
from app.services.pdf_reconstructor import PDFReconstructor
from app.services.vision_layout_analyzer import VisionLayoutAnalyzer, PrecisePDFReconstructor
from app.core.config import settings

logger = structlog.get_logger()

app = FastAPI(title="Pixel-Perfect Resume Optimizer", version="2.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global temp directory for files
TEMP_DIR = Path("/tmp/pixel_perfect_resumes")
TEMP_DIR.mkdir(exist_ok=True)

# In-memory storage for processed files
processed_files = {}


@app.get("/health")
async def health_check():
    """Health check with vision model status"""
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key.startswith("sk-")),
        "model_text": settings.openai_model_text,
        "model_vision": settings.openai_model_vision,
        "pixel_perfect_enabled": True
    }


@app.post("/optimize")
async def optimize_resume(
    resume_file: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
    job_title: Optional[str] = Form(None, description="Job title (optional)"),
    company_name: Optional[str] = Form(None, description="Company name (optional)")
):
    """
    Pixel-perfect resume optimization using GPT-4o Vision
    Recreates exact layout with optimized content
    """
    
    session_id = str(uuid.uuid4())
    logger.info("Starting pixel-perfect resume optimization", session_id=session_id)
    
    try:
        # Step 1: Save uploaded file
        resume_content = await resume_file.read()
        temp_pdf_path = TEMP_DIR / f"{session_id}_original.pdf"
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(resume_content)
        
        logger.info("Resume file saved", session_id=session_id, size=len(resume_content))
        
        # Step 2: Extract text from PDF
        pdf_reconstructor = PDFReconstructor()
        resume_text = pdf_reconstructor.pdf_to_text(temp_pdf_path)
        logger.info("Text extracted from PDF", session_id=session_id, text_length=len(resume_text))
        
        # Step 3: Convert to high-quality image for vision analysis
        image_path = pdf_reconstructor.first_page_to_png(temp_pdf_path, dpi=300)  # High DPI for precision
        logger.info("High-resolution image created", session_id=session_id, image_path=str(image_path))
        
        # Step 4: AI optimization
        ai_processor = AIProcessor(settings.openai_api_key)
        
        # Optimize resume content
        optimized_resume_markdown = await ai_processor.optimize_resume_text(
            resume_text, job_description
        )
        logger.info("Resume optimization completed", session_id=session_id)
        
        # Generate cover letter
        cover_letter_markdown = await ai_processor.generate_cover_letter(
            optimized_resume_markdown, job_description, job_title, company_name
        )
        logger.info("Cover letter generation completed", session_id=session_id)
        
        # Step 5: GPT-4o Vision layout analysis
        vision_analyzer = VisionLayoutAnalyzer(ai_processor.client)
        layout_data = vision_analyzer.analyze_resume_layout(image_path)
        logger.info("GPT-4o Vision layout analysis completed", session_id=session_id)
        
        # Step 6: Parse optimized content into sections
        optimized_sections = _parse_markdown_to_sections(optimized_resume_markdown)
        
        # Step 7: Pixel-perfect PDF recreation
        precise_reconstructor = PrecisePDFReconstructor(vision_analyzer)
        
        # Create pixel-perfect resume
        pixel_perfect_resume_path = TEMP_DIR / f"{session_id}_pixel_perfect_resume.pdf"
        precise_reconstructor.recreate_with_optimized_content(
            original_image_path=image_path,
            optimized_content=optimized_sections,
            output_path=pixel_perfect_resume_path
        )
        
        # Create cover letter (using standard layout since it's new)
        cover_letter_path = TEMP_DIR / f"{session_id}_cover_letter.pdf"
        pdf_reconstructor.markdown_to_pdf_with_layout(
            markdown_content=cover_letter_markdown,
            layout_coords={},  # Use clean layout for cover letter
            output_filename=str(cover_letter_path),
            document_type="cover_letter"
        )
        
        logger.info("Pixel-perfect PDF recreation completed", session_id=session_id)
        
        # Step 8: Store file paths for download
        processed_files[f"{session_id}_pixel_perfect_resume"] = pixel_perfect_resume_path
        processed_files[f"{session_id}_cover_letter"] = cover_letter_path
        processed_files[f"{session_id}_resume_md"] = optimized_resume_markdown
        processed_files[f"{session_id}_cover_letter_md"] = cover_letter_markdown
        
        # Get AI processing metadata
        ai_metadata = ai_processor.get_processing_metadata()
        
        # Clean up temp files
        try:
            temp_pdf_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to clean up temp files", error=str(e))
        
        return {
            "session_id": session_id,
            "status": "completed",
            "method": "pixel_perfect_vision",
            "downloads": {
                "pixel_perfect_resume_pdf": f"/download/{session_id}_pixel_perfect_resume",
                "cover_letter_pdf": f"/download/{session_id}_cover_letter",
                "optimized_resume_markdown": f"/download/{session_id}_resume_md",
                "cover_letter_markdown": f"/download/{session_id}_cover_letter_md"
            },
            "ai_processing": {
                "used_fallback_resume": ai_metadata.get("used_fallback_resume", False),
                "used_fallback_cover_letter": ai_metadata.get("used_fallback_cover_letter", False),
                "model_used": ai_metadata.get("openai_model_text", "unknown"),
                "vision_model": settings.openai_model_vision
            },
            "layout_analysis": {
                "method": "gpt4o_vision",
                "precision": "pixel_perfect",
                "elements_detected": len(layout_data.get('elements', [])),
                "fonts_detected": len(layout_data.get('fonts', []))
            },
            "message": "Pixel-perfect resume optimization completed! Exact layout preserved with optimized content."
        }
        
    except Exception as e:
        logger.error("Pixel-perfect optimization failed", session_id=session_id, error=str(e))
        # Clean up on error
        try:
            temp_pdf_path.unlink(missing_ok=True)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Pixel-perfect processing failed: {str(e)}")


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download processed files"""
    
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path_or_content = processed_files[file_id]
    
    # Handle markdown content (stored as string)
    if file_id.endswith("_md"):
        # Return markdown as text file
        temp_md_path = TEMP_DIR / f"{file_id}.md"
        with open(temp_md_path, 'w') as f:
            f.write(file_path_or_content)
        
        filename = f"{file_id.replace('_md', '')}.md"
        return FileResponse(
            temp_md_path,
            media_type="text/markdown",
            filename=filename
        )
    
    # Handle PDF files
    if not Path(file_path_or_content).exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    filename = f"{file_id.replace('_', '_')}.pdf"
    return FileResponse(
        file_path_or_content,
        media_type="application/pdf",
        filename=filename
    )


def _parse_markdown_to_sections(markdown_content: str) -> Dict[str, str]:
    """Parse markdown content into sections for layout mapping"""
    
    sections = {}
    current_section = "header"
    current_content = []
    
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('# '):
            # Main header
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = "header"
            current_content = [line[2:]]  # Remove '# '
            
        elif line.startswith('## '):
            # Section header
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            section_name = line[3:].lower()  # Remove '## '
            if 'summary' in section_name or 'profile' in section_name:
                current_section = "summary"
            elif 'experience' in section_name or 'work' in section_name:
                current_section = "experience"
            elif 'skill' in section_name or 'technical' in section_name:
                current_section = "skills"
            elif 'education' in section_name or 'academic' in section_name:
                current_section = "education"
            else:
                current_section = section_name.replace(' ', '_')
            
            current_content = []
            
        elif line:
            current_content.append(line)
    
    # Add final section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
