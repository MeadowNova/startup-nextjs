#!/usr/bin/env python3
"""
Simple Working Resume Optimizer - Based on Reference Document
This implements the EXACT algorithm from reference.md that actually works
"""

import os
import tempfile
import zipfile
import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Simple imports - no over-engineering
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from openai import OpenAI

from app.core.config import settings

logger = structlog.get_logger()

app = FastAPI(title="Simple Working Resume Optimizer", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global temp directory
TEMP_DIR = Path("/tmp/simple_resumes")
TEMP_DIR.mkdir(exist_ok=True)

# OpenAI client
client = OpenAI(api_key=settings.openai_api_key)

# In-memory storage for downloads
processed_files = {}


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key.startswith("sk-")),
        "approach": "simple_working_solution"
    }


@app.post("/optimize")
async def optimize_resume(
    resume_file: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text")
):
    """
    Simple working resume optimization - implements reference.md algorithm exactly
    """
    
    session_id = f"simple_{hash(job_description[:100])}"
    logger.info("Starting simple resume optimization", session_id=session_id)
    
    try:
        # Step 1: Save uploaded resume
        resume_content = await resume_file.read()
        resume_path = TEMP_DIR / f"{session_id}_original.pdf"
        
        with open(resume_path, 'wb') as f:
            f.write(resume_content)
        
        logger.info("Resume saved", session_id=session_id, size=len(resume_content))
        
        # Step 2: Extract text from resume PDF (PyMuPDF)
        old_text = pdf_to_text(resume_path)
        logger.info("Text extracted from resume", session_id=session_id, text_length=len(old_text))
        
        # Step 3: Optimize resume text with GPT-4o-mini
        new_md = optimize_resume_text(old_text, job_description)
        logger.info("Resume text optimized", session_id=session_id)
        
        # Step 4: Generate cover letter
        cover_md = matching_cover_letter(new_md, job_description)
        logger.info("Cover letter generated", session_id=session_id)
        
        # Step 5: Convert first page to PNG (for template reference)
        png_path = first_page_to_png(resume_path)
        logger.info("PNG template created", session_id=session_id)
        
        # Step 6: Create new PDFs using simple layout
        new_resume_pdf = make_pdf_from_layout(new_md, png_path, "resume")
        cover_pdf = make_pdf_from_layout(cover_md, png_path, "cover")
        
        logger.info("PDFs created successfully", session_id=session_id)
        
        # Step 7: Store files for download
        processed_files[f"{session_id}_resume"] = new_resume_pdf
        processed_files[f"{session_id}_cover"] = cover_pdf
        processed_files[f"{session_id}_resume_md"] = new_md
        processed_files[f"{session_id}_cover_md"] = cover_md
        
        # Clean up temp files
        try:
            resume_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to clean up temp files", error=str(e))
        
        return {
            "session_id": session_id,
            "status": "completed",
            "method": "simple_working_solution",
            "downloads": {
                "optimized_resume_pdf": f"/download/{session_id}_resume",
                "cover_letter_pdf": f"/download/{session_id}_cover",
                "optimized_resume_markdown": f"/download/{session_id}_resume_md",
                "cover_letter_markdown": f"/download/{session_id}_cover_md"
            },
            "message": "Resume optimized successfully using proven simple approach!"
        }
        
    except Exception as e:
        logger.error("Simple optimization failed", session_id=session_id, error=str(e))
        # Clean up on error
        try:
            resume_path.unlink(missing_ok=True)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


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
    
    filename = f"{file_id}.pdf"
    return FileResponse(
        file_path_or_content,
        media_type="application/pdf",
        filename=filename
    )


# ============================================================================
# CORE FUNCTIONS - Exact implementation from reference.md
# ============================================================================

def pdf_to_text(path: Path) -> str:
    """Extract text from PDF using PyMuPDF - from reference.md"""
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def first_page_to_png(pdf_path: Path) -> Path:
    """Convert first page to PNG - from reference.md"""
    imgs = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
    png_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    imgs[0].save(png_path, "PNG")
    return Path(png_path)


def make_pdf_from_layout(markdown: str, template_png: Path, kind: str) -> Path:
    """
    Simple PDF creation - EXACT implementation from reference.md
    This is the working version that just re-flows text
    """
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(out, pagesize=letter)
    width, height = letter
    
    # Start text positioning
    text_obj = c.beginText(50, height-100)
    
    # Try to use a better font, fallback to Helvetica
    try:
        # Try to register a font (this might fail, that's OK)
        text_obj.setFont("Helvetica", 11)
    except:
        text_obj.setFont("Helvetica", 11)
    
    # Add content line by line
    for line in markdown.splitlines():
        if line.strip():  # Skip empty lines
            text_obj.textLine(line.strip())
        else:
            text_obj.textLine(" ")  # Add space for empty lines
    
    c.drawText(text_obj)
    c.showPage()
    c.save()
    return Path(out)


def optimize_resume_text(resume: str, jd: str) -> str:
    """Optimize resume text with GPT-4o-mini - from reference.md"""
    
    SYS_RESUME = """You are an expert résumé writer who beats ATS.  
Rewrite the résumé below so it mirrors the job description keywords while remaining truthful.  
Return ONLY plain Markdown (no ```markdown fence).  
Keep sections: Summary, Skills, Experience, Education, etc.  
Make every bullet start with action verbs and quantify impact."""
    
    prompt = f"Job Description:\n{jd}\n\nCurrent Résumé:\n{resume}"
    
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYS_RESUME},
            {"role": "user", "content": prompt}
        ],
        temperature=0.25
    )
    return chat.choices[0].message.content.strip()


def matching_cover_letter(resume_md: str, jd: str) -> str:
    """Generate matching cover letter - from reference.md"""
    
    SYS_COVER = """Write a concise 250-word cover letter in Markdown that matches the résumé and job description.  
Return ONLY the letter body (no salutation block)."""
    
    prompt = f"Résumé:\n{resume_md}\n\nJob Description:\n{jd}"
    
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYS_COVER},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return chat.choices[0].message.content.strip()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
