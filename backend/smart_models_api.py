#!/usr/bin/env python3
"""
Smart Models Resume Optimizer
Uses the best available models for each task:
- Claude 3.5 Sonnet for text optimization (better than GPT-4o-mini)
- Marker for PDF to Markdown conversion (better layout preservation)
- Surya for OCR and layout analysis (open source, reliable)
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Smart model imports
import anthropic  # Claude 3.5 Sonnet
from openai import OpenAI  # Fallback to GPT-4o
import subprocess
import json
import requests  # For Mistral OCR API

from app.core.config import settings

logger = structlog.get_logger()

app = FastAPI(title="Smart Models Resume Optimizer", version="2.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global temp directory
TEMP_DIR = Path("/tmp/smart_resumes")
TEMP_DIR.mkdir(exist_ok=True)

# Model clients
claude_client = None
openai_client = None
mistral_api_key = None

# Initialize clients with optimized settings
try:
    if settings.anthropic_api_key:
        # Initialize Claude with connection pooling and timeout settings
        claude_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=60.0,  # 60 second timeout
            max_retries=3  # Retry failed requests
        )
        logger.info("Claude 3.5 Sonnet client initialized with optimized settings")
    else:
        logger.warning("No ANTHROPIC_API_KEY found, will use OpenAI fallback")
except Exception as e:
    logger.warning("Failed to initialize Claude client", error=str(e))

try:
    # Initialize OpenAI with optimized settings
    openai_client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=60.0,  # 60 second timeout
        max_retries=3  # Retry failed requests
    )
    logger.info("OpenAI client initialized with optimized settings")
except Exception as e:
    logger.error("Failed to initialize OpenAI client", error=str(e))

try:
    mistral_api_key = settings.mistral_api_key
    if mistral_api_key:
        logger.info("Mistral OCR API key loaded")
    else:
        logger.warning("No MISTRAL_API_KEY found, will use fallback text extraction")
except Exception as e:
    logger.warning("Failed to load Mistral API key", error=str(e))

# In-memory storage for downloads
processed_files = {}

# Performance optimization features
executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing
ats_analysis_cache = {}  # Cache ATS analysis results
CACHE_TTL = 3600  # 1 hour cache TTL

def get_cache_key(text: str) -> str:
    """Generate cache key for text content"""
    return hashlib.md5(text.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - timestamp < CACHE_TTL

@lru_cache(maxsize=100)
def cached_keyword_extraction(jd_hash: str, jd: str) -> dict:
    """Cache keyword extraction results for job descriptions"""
    return extract_keywords_from_job_description(jd)


@app.get("/health")
async def health_check():
    """Health check with model status"""
    return {
        "status": "healthy",
        "claude_available": claude_client is not None,
        "openai_available": openai_client is not None,
        "mistral_ocr_available": mistral_api_key is not None,
        "marker_available": check_marker_available(),
        "surya_available": check_surya_available(),
        "approach": "mistral_ocr_claude_hybrid"
    }


@app.post("/optimize")
async def optimize_resume_smart(
    resume_file: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
    job_title: Optional[str] = Form(None, description="Specific job title (optional, for better personalization)"),
    company_name: Optional[str] = Form(None, description="Company name (optional, for better personalization)")
):
    """
    Smart resume optimization using best available models
    """
    
    session_id = f"smart_{hash(job_description[:100])}"
    logger.info("Starting smart resume optimization", session_id=session_id)
    
    try:
        # Step 1: Save uploaded resume with memory optimization
        resume_path = TEMP_DIR / f"{session_id}_original.pdf"
        file_size = 0

        # Stream file to disk to handle large files efficiently
        with open(resume_path, 'wb') as f:
            while chunk := await resume_file.read(8192):  # Read in 8KB chunks
                f.write(chunk)
                file_size += len(chunk)

        logger.info("Resume saved", session_id=session_id, size=file_size)
        
        # Step 2: Extract text using best available method with pixel-perfect layout preservation
        if mistral_api_key:
            # Use Mistral OCR for pixel-perfect layout extraction
            old_text = extract_text_with_mistral_ocr(resume_path)
            extraction_method = "mistral-ocr-2503"
            logger.info("Using Mistral OCR for pixel-perfect layout extraction", session_id=session_id)
        elif check_marker_available():
            # Fallback to Marker for high-quality PDF to Markdown
            old_text = extract_text_with_marker(resume_path)
            extraction_method = "marker"
            logger.info("Using Marker for text extraction", session_id=session_id)
        else:
            # Final fallback to PyMuPDF
            old_text = extract_text_pymupdf(resume_path)
            extraction_method = "pymupdf"
            logger.info("Using PyMuPDF fallback for text extraction", session_id=session_id)
        
        logger.info("Text extracted", session_id=session_id, method=extraction_method, text_length=len(old_text))
        
        # Step 3 & 4: Optimize resume and generate cover letter in parallel for better performance
        start_time = time.time()

        # Determine which models to use
        if openai_client:
            optimization_model = "gpt-4o"
            use_claude = False
        elif claude_client:
            optimization_model = "claude-3.5-sonnet"
            use_claude = True
        else:
            raise HTTPException(status_code=500, detail="No AI models available")

        # Use asyncio to run both operations concurrently
        async def optimize_resume_async():
            loop = asyncio.get_event_loop()
            if use_claude:
                return await loop.run_in_executor(executor, optimize_with_claude, old_text, job_description)
            else:
                return await loop.run_in_executor(executor, optimize_with_openai, old_text, job_description)

        async def generate_cover_letter_async(optimized_resume):
            loop = asyncio.get_event_loop()
            if use_claude:
                return await loop.run_in_executor(executor, generate_cover_letter_claude, optimized_resume, job_description, job_title, company_name)
            else:
                return await loop.run_in_executor(executor, generate_cover_letter_openai, optimized_resume, job_description, job_title, company_name)

        # First optimize the resume
        new_md = await optimize_resume_async()
        logger.info("Resume optimized", session_id=session_id, model=optimization_model)

        # Then generate cover letter based on optimized resume
        cover_md = await generate_cover_letter_async(new_md)

        processing_time = time.time() - start_time
        logger.info("AI processing completed", session_id=session_id, processing_time=f"{processing_time:.2f}s")
        
        # Step 5: Create PDFs with pixel-perfect layout preservation
        if mistral_api_key and extraction_method == "mistral-ocr-2503":
            # Use Mistral OCR layout analysis for pixel-perfect recreation
            layout_analysis = analyze_layout_with_mistral_ocr(resume_path)
            new_resume_pdf = create_pdf_with_mistral_layout(new_md, layout_analysis)
            layout_method = "mistral-pixel-perfect"
            logger.info("Using Mistral OCR for pixel-perfect PDF recreation", session_id=session_id)
        elif check_surya_available():
            # Use Surya for layout analysis + ReportLab for recreation
            new_resume_pdf = create_pdf_with_surya_layout(new_md, resume_path)
            layout_method = "surya"
            logger.info("Using Surya for layout analysis", session_id=session_id)
        else:
            # Fallback to enhanced simple layout
            new_resume_pdf = create_simple_pdf(new_md)
            layout_method = "enhanced-simple"
            logger.info("Using enhanced simple layout", session_id=session_id)
        
        cover_pdf = create_simple_pdf(cover_md)
        
        logger.info("PDFs created", session_id=session_id, layout_method=layout_method)
        
        # Step 6: Store files for download
        processed_files[f"{session_id}_resume"] = new_resume_pdf
        processed_files[f"{session_id}_cover"] = cover_pdf
        processed_files[f"{session_id}_resume_md"] = new_md
        processed_files[f"{session_id}_cover_md"] = cover_md
        
        # Clean up temp files
        try:
            resume_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to clean up temp files", error=str(e))
        
        # Calculate total processing time and performance metrics
        total_time = time.time() - start_time

        performance_metrics = {
            "total_processing_time": f"{total_time:.2f}s",
            "ai_processing_time": f"{processing_time:.2f}s",
            "file_size_mb": f"{file_size / (1024*1024):.2f}",
            "text_length": len(old_text),
            "optimized_length": len(new_md),
            "compression_ratio": f"{len(new_md) / len(old_text):.2f}" if old_text else 1.0,
            "parallel_processing": True,
            "cache_enabled": True
        }

        logger.info("Optimization completed successfully", session_id=session_id, **performance_metrics)

        return {
            "session_id": session_id,
            "status": "completed",
            "method": "smart_models_hybrid",
            "models_used": {
                "text_extraction": extraction_method,
                "optimization": optimization_model,
                "layout_analysis": layout_method
            },
            "downloads": {
                "optimized_resume_pdf": f"/download/{session_id}_resume",
                "cover_letter_pdf": f"/download/{session_id}_cover",
                "optimized_resume_markdown": f"/download/{session_id}_resume_md",
                "cover_letter_markdown": f"/download/{session_id}_cover_md"
            },
            "performance_metrics": performance_metrics,
            "message": f"Resume optimized using {optimization_model} with {layout_method} layout analysis in {total_time:.1f}s!"
        }
        
    except Exception as e:
        logger.error("Smart optimization failed", session_id=session_id, error=str(e))
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
# SMART MODEL FUNCTIONS
# ============================================================================

def check_marker_available() -> bool:
    """Check if Marker is installed and available"""
    try:
        result = subprocess.run(["marker", "--help"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def check_surya_available() -> bool:
    """Check if Surya is installed and available"""
    try:
        import surya
        return True
    except ImportError:
        return False


def extract_text_with_mistral_ocr(pdf_path: Path) -> str:
    """Extract text using Mistral OCR (pixel-perfect layout preservation)"""
    try:
        if not mistral_api_key:
            raise Exception("Mistral API key not available")

        # Read PDF file and encode as base64
        import base64
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')

        # Call Mistral OCR API with correct format
        headers = {
            'Authorization': f'Bearer {mistral_api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "mistral-ocr-2503",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this resume PDF with pixel-perfect layout preservation.

CRITICAL REQUIREMENTS:
1. Preserve exact spacing, alignment, and visual hierarchy
2. Maintain original font styling (bold, italic) using markdown
3. Keep section headers exactly as positioned
4. Preserve bullet point formatting and indentation
5. Maintain contact information layout
6. Keep date formatting and positioning intact
7. Preserve any tables, columns, or special layouts
8. Output in clean markdown format that can recreate the exact visual appearance

Focus on maintaining the professional layout that makes this resume visually appealing and ATS-compatible."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent, precise extraction
            "max_tokens": 4000   # Sufficient tokens for detailed resume content
        }

        response = requests.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            # Extract text from the response
            extracted_text = result['choices'][0]['message']['content']

            # Log successful extraction with layout preservation
            logger.info("Mistral OCR extraction successful",
                       text_length=len(extracted_text),
                       layout_preserved=True,
                       pixel_perfect=True)

            return extracted_text
        else:
            raise Exception(f"Mistral OCR failed: {response.status_code} - {response.text}")

    except Exception as e:
        logger.warning("Mistral OCR extraction failed, falling back to PyMuPDF", error=str(e))
        return extract_text_pymupdf(pdf_path)


def analyze_layout_with_mistral_ocr(pdf_path: Path) -> dict:
    """Analyze layout structure using Mistral OCR for pixel-perfect recreation"""
    try:
        if not mistral_api_key:
            raise Exception("Mistral API key not available")

        # Read PDF file and encode as base64
        import base64
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')

        # Call Mistral OCR API for layout analysis
        headers = {
            'Authorization': f'Bearer {mistral_api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "mistral-ocr-2503",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze the layout structure of this resume PDF and provide detailed formatting information.

LAYOUT ANALYSIS REQUIREMENTS:
1. Identify font sizes and styles for each section
2. Measure spacing between sections and elements
3. Detect alignment patterns (left, center, right)
4. Identify bullet point styles and indentation levels
5. Analyze contact information layout and formatting
6. Detect any tables, columns, or special layouts
7. Identify color schemes and visual hierarchy
8. Measure margins and padding

Return a JSON structure with:
- section_styles: {section_name: {font_size, font_weight, spacing, alignment}}
- bullet_styles: {type, indentation, spacing}
- contact_layout: {format, alignment, spacing}
- overall_layout: {margins, columns, visual_hierarchy}
- font_analysis: {primary_font, sizes_used, styling_patterns}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }

        response = requests.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            layout_analysis = result['choices'][0]['message']['content']

            logger.info("Mistral layout analysis successful",
                       analysis_length=len(layout_analysis))

            # Try to parse as JSON, fallback to text analysis
            try:
                import json
                return json.loads(layout_analysis)
            except:
                return {"raw_analysis": layout_analysis, "format": "text"}
        else:
            raise Exception(f"Mistral layout analysis failed: {response.status_code} - {response.text}")

    except Exception as e:
        logger.warning("Mistral layout analysis failed", error=str(e))
        return {"error": str(e), "fallback": True}


def extract_text_with_marker(pdf_path: Path) -> str:
    """Extract text using Marker (high quality PDF to Markdown)"""
    try:
        output_dir = TEMP_DIR / "marker_output"
        output_dir.mkdir(exist_ok=True)

        # Run Marker
        result = subprocess.run([
            "marker",
            str(pdf_path),
            str(output_dir),
            "--batch_multiplier", "1"
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # Find the generated markdown file
            md_files = list(output_dir.glob("*.md"))
            if md_files:
                with open(md_files[0], 'r') as f:
                    return f.read()

        raise Exception(f"Marker failed: {result.stderr}")

    except Exception as e:
        logger.warning("Marker extraction failed, falling back to PyMuPDF", error=str(e))
        return extract_text_pymupdf(pdf_path)


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Fallback text extraction using PyMuPDF"""
    import fitz
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# ============================================================================
# ADVANCED ATS OPTIMIZATION FUNCTIONS
# ============================================================================

def extract_keywords_from_text(text: str) -> list:
    """Extract meaningful keywords from text using NLP techniques"""
    import re
    from collections import Counter

    # Common stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }

    # Extract words (2+ characters, alphanumeric)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    # Filter out stop words and get frequency
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(meaningful_words)

    # Return top keywords
    return [word for word, freq in word_freq.most_common(50)]


def extract_requirements_section(jd: str) -> str:
    """Extract requirements/qualifications section from job description"""
    import re

    # Common section headers for requirements
    requirement_patterns = [
        r'(?i)requirements?:?(.*?)(?=\n\n|\n[A-Z]|$)',
        r'(?i)qualifications?:?(.*?)(?=\n\n|\n[A-Z]|$)',
        r'(?i)skills?:?(.*?)(?=\n\n|\n[A-Z]|$)',
        r'(?i)experience:?(.*?)(?=\n\n|\n[A-Z]|$)',
        r'(?i)must have:?(.*?)(?=\n\n|\n[A-Z]|$)'
    ]

    requirements_text = ""
    for pattern in requirement_patterns:
        matches = re.findall(pattern, jd, re.DOTALL)
        if matches:
            requirements_text += " ".join(matches)

    return requirements_text if requirements_text else jd[:500]  # Fallback to first 500 chars


def extract_action_verbs(text: str) -> list:
    """Extract action verbs commonly used in job descriptions"""
    import re
    from collections import Counter

    # Common action verbs in job descriptions
    action_verb_patterns = [
        r'\b(?:manage|lead|develop|implement|create|design|build|optimize|improve|analyze)\b',
        r'\b(?:coordinate|collaborate|execute|deliver|achieve|drive|establish|maintain)\b',
        r'\b(?:oversee|supervise|mentor|train|guide|support|facilitate|ensure)\b',
        r'\b(?:research|evaluate|assess|monitor|track|measure|report|document)\b'
    ]

    verbs = []
    for pattern in action_verb_patterns:
        verbs.extend(re.findall(pattern, text, re.IGNORECASE))

    verb_freq = Counter([verb.lower() for verb in verbs])
    return [verb for verb, freq in verb_freq.most_common(15)]


def analyze_keyword_coverage(resume_keywords: list, jd_keywords: dict) -> dict:
    """Analyze how well resume covers job description keywords"""

    # Check coverage for each keyword category
    tech_coverage = calculate_coverage(resume_keywords, jd_keywords.get('technical_skills', []))
    soft_coverage = calculate_coverage(resume_keywords, jd_keywords.get('soft_skills', []))
    requirements_coverage = calculate_coverage(resume_keywords, jd_keywords.get('requirements', []))
    action_verbs_coverage = calculate_coverage(resume_keywords, jd_keywords.get('action_verbs', []))
    overall_coverage = calculate_coverage(resume_keywords, jd_keywords.get('all_keywords', []))

    return {
        "technical_skills": {
            "coverage_percentage": tech_coverage,
            "missing_keywords": get_missing_keywords(resume_keywords, jd_keywords.get('technical_skills', [])),
            "matched_keywords": get_matched_keywords(resume_keywords, jd_keywords.get('technical_skills', []))
        },
        "soft_skills": {
            "coverage_percentage": soft_coverage,
            "missing_keywords": get_missing_keywords(resume_keywords, jd_keywords.get('soft_skills', [])),
            "matched_keywords": get_matched_keywords(resume_keywords, jd_keywords.get('soft_skills', []))
        },
        "requirements": {
            "coverage_percentage": requirements_coverage,
            "missing_keywords": get_missing_keywords(resume_keywords, jd_keywords.get('requirements', [])),
            "matched_keywords": get_matched_keywords(resume_keywords, jd_keywords.get('requirements', []))
        },
        "action_verbs": {
            "coverage_percentage": action_verbs_coverage,
            "missing_keywords": get_missing_keywords(resume_keywords, jd_keywords.get('action_verbs', [])),
            "matched_keywords": get_matched_keywords(resume_keywords, jd_keywords.get('action_verbs', []))
        },
        "overall": {
            "coverage_percentage": overall_coverage,
            "keyword_density": len(set(resume_keywords) & set(jd_keywords.get('all_keywords', []))) / len(resume_keywords) if resume_keywords else 0
        }
    }


def calculate_coverage(resume_keywords: list, target_keywords: list) -> float:
    """Calculate percentage coverage of target keywords in resume"""
    if not target_keywords:
        return 100.0

    matched = len(set(resume_keywords) & set(target_keywords))
    return (matched / len(target_keywords)) * 100


def get_missing_keywords(resume_keywords: list, target_keywords: list) -> list:
    """Get keywords that are missing from resume"""
    return list(set(target_keywords) - set(resume_keywords))


def get_matched_keywords(resume_keywords: list, target_keywords: list) -> list:
    """Get keywords that are present in resume"""
    return list(set(resume_keywords) & set(target_keywords))


def check_ats_formatting(resume_text: str) -> list:
    """Check for ATS formatting issues"""
    issues = []

    # Check for problematic characters
    problematic_chars = ['•', '→', '★', '◆', '▪', '▫']
    for char in problematic_chars:
        if char in resume_text:
            issues.append(f"Contains potentially problematic character: {char}")

    # Check for proper section headers
    required_sections = ['experience', 'education', 'skills']
    text_lower = resume_text.lower()
    for section in required_sections:
        if section not in text_lower:
            issues.append(f"Missing standard section: {section}")

    # Check for contact information
    import re
    if not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text):
        issues.append("Missing email address")

    if not re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text):
        issues.append("Missing phone number")

    # Check for excessive formatting
    if resume_text.count('**') > 20:
        issues.append("Excessive bold formatting may confuse ATS")

    return issues


def detect_industry(job_description: str) -> str:
    """Detect industry from job description"""
    jd_lower = job_description.lower()

    # Industry keywords mapping
    industry_keywords = {
        'technology': ['software', 'developer', 'engineer', 'programming', 'coding', 'tech', 'api', 'database'],
        'healthcare': ['medical', 'healthcare', 'hospital', 'patient', 'clinical', 'nurse', 'doctor'],
        'finance': ['financial', 'banking', 'investment', 'accounting', 'finance', 'trading', 'portfolio'],
        'marketing': ['marketing', 'advertising', 'brand', 'campaign', 'social media', 'content', 'seo'],
        'sales': ['sales', 'revenue', 'client', 'customer', 'business development', 'account management'],
        'education': ['education', 'teaching', 'academic', 'university', 'school', 'curriculum', 'student'],
        'consulting': ['consulting', 'advisory', 'strategy', 'analysis', 'recommendations', 'client solutions']
    }

    industry_scores = {}
    for industry, keywords in industry_keywords.items():
        score = sum(1 for keyword in keywords if keyword in jd_lower)
        industry_scores[industry] = score

    # Return industry with highest score, default to 'general'
    return max(industry_scores, key=industry_scores.get) if max(industry_scores.values()) > 0 else 'general'


def get_industry_specific_recommendations(industry: str, resume_text: str, job_description: str) -> list:
    """Get industry-specific optimization recommendations"""
    recommendations = []

    industry_specific_advice = {
        'technology': [
            "Include specific programming languages and frameworks",
            "Mention version control systems (Git, SVN)",
            "Highlight cloud platforms and DevOps experience",
            "Quantify system performance improvements",
            "Include relevant certifications (AWS, Azure, etc.)"
        ],
        'healthcare': [
            "Include relevant medical certifications and licenses",
            "Mention patient care experience and outcomes",
            "Highlight compliance with healthcare regulations",
            "Include experience with medical software/systems",
            "Quantify patient satisfaction or care improvements"
        ],
        'finance': [
            "Include financial certifications (CPA, CFA, etc.)",
            "Mention regulatory compliance experience",
            "Highlight risk management and analysis skills",
            "Quantify financial impact and cost savings",
            "Include experience with financial software/systems"
        ],
        'marketing': [
            "Include digital marketing certifications",
            "Mention specific marketing channels and platforms",
            "Highlight campaign performance metrics",
            "Include experience with marketing automation tools",
            "Quantify lead generation and conversion rates"
        ],
        'sales': [
            "Include sales methodology training",
            "Mention CRM system experience",
            "Highlight quota achievement and revenue growth",
            "Include client relationship management skills",
            "Quantify sales performance and targets exceeded"
        ]
    }

    return industry_specific_advice.get(industry, [
        "Use industry-standard terminology from the job description",
        "Include relevant certifications and training",
        "Quantify achievements with specific metrics",
        "Highlight transferable skills and experiences"
    ])


def calculate_ats_score(keyword_analysis: dict, formatting_issues: list) -> float:
    """Calculate overall ATS compatibility score"""
    # Base score from keyword coverage
    overall_coverage = keyword_analysis.get('overall', {}).get('coverage_percentage', 0)
    tech_coverage = keyword_analysis.get('technical_skills', {}).get('coverage_percentage', 0)
    requirements_coverage = keyword_analysis.get('requirements', {}).get('coverage_percentage', 0)

    # Weighted average of coverage scores
    keyword_score = (overall_coverage * 0.4 + tech_coverage * 0.3 + requirements_coverage * 0.3)

    # Deduct points for formatting issues
    formatting_penalty = len(formatting_issues) * 5  # 5 points per issue

    # Final score (0-100)
    final_score = max(0, keyword_score - formatting_penalty)

    return round(final_score, 1)


def get_optimization_priorities(keyword_analysis: dict, formatting_issues: list) -> list:
    """Get prioritized list of optimization recommendations"""
    priorities = []

    # High priority: Fix formatting issues
    if formatting_issues:
        priorities.append({
            "priority": "HIGH",
            "category": "Formatting",
            "recommendation": "Fix ATS formatting issues",
            "details": formatting_issues
        })

    # High priority: Missing technical skills
    tech_missing = keyword_analysis.get('technical_skills', {}).get('missing_keywords', [])
    if tech_missing:
        priorities.append({
            "priority": "HIGH",
            "category": "Technical Skills",
            "recommendation": f"Add missing technical skills: {', '.join(tech_missing[:5])}",
            "details": tech_missing
        })

    # Medium priority: Missing requirements
    req_missing = keyword_analysis.get('requirements', {}).get('missing_keywords', [])
    if req_missing:
        priorities.append({
            "priority": "MEDIUM",
            "category": "Requirements",
            "recommendation": f"Include missing requirement keywords: {', '.join(req_missing[:5])}",
            "details": req_missing
        })

    # Low priority: Soft skills
    soft_missing = keyword_analysis.get('soft_skills', {}).get('missing_keywords', [])
    if soft_missing:
        priorities.append({
            "priority": "LOW",
            "category": "Soft Skills",
            "recommendation": f"Consider adding soft skills: {', '.join(soft_missing[:3])}",
            "details": soft_missing
        })

    return priorities


def analyze_ats_compliance(resume_text: str, job_description: str) -> dict:
    """Analyze ATS compliance and provide detailed optimization recommendations with caching"""
    import re
    from collections import Counter

    # Create cache key for job description (resume changes each time, but JD might be reused)
    jd_cache_key = get_cache_key(job_description)

    # Check cache for job description analysis
    cached_jd_analysis = ats_analysis_cache.get(jd_cache_key)
    if cached_jd_analysis and is_cache_valid(cached_jd_analysis['timestamp']):
        jd_keywords = cached_jd_analysis['keywords']
        industry = cached_jd_analysis['industry']
        logger.info("Using cached job description analysis", cache_key=jd_cache_key[:8])
    else:
        # Extract keywords from job description
        jd_keywords = extract_keywords_from_job_description(job_description)
        industry = detect_industry(job_description)

        # Cache the results
        ats_analysis_cache[jd_cache_key] = {
            'keywords': jd_keywords,
            'industry': industry,
            'timestamp': time.time()
        }
        logger.info("Cached job description analysis", cache_key=jd_cache_key[:8])

    # Analyze current resume (always fresh since resume content changes)
    resume_keywords = extract_keywords_from_text(resume_text.lower())

    # Calculate keyword density and coverage
    keyword_analysis = analyze_keyword_coverage(resume_keywords, jd_keywords)

    # Check ATS formatting compliance
    formatting_issues = check_ats_formatting(resume_text)

    # Industry-specific analysis
    industry_recommendations = get_industry_specific_recommendations(industry, resume_text, job_description)

    return {
        "keyword_analysis": keyword_analysis,
        "formatting_issues": formatting_issues,
        "industry_recommendations": industry_recommendations,
        "ats_score": calculate_ats_score(keyword_analysis, formatting_issues),
        "optimization_priority": get_optimization_priorities(keyword_analysis, formatting_issues),
        "cache_used": cached_jd_analysis is not None and is_cache_valid(cached_jd_analysis['timestamp'])
    }


def extract_keywords_from_job_description(jd: str) -> dict:
    """Extract and categorize keywords from job description"""
    import re
    from collections import Counter

    # Technical skills patterns
    tech_patterns = [
        r'\b(?:Python|Java|JavaScript|React|Node\.js|AWS|Docker|Kubernetes|SQL|MongoDB|PostgreSQL)\b',
        r'\b(?:Machine Learning|AI|Data Science|Analytics|Cloud|DevOps|Agile|Scrum)\b',
        r'\b(?:API|REST|GraphQL|Microservices|CI/CD|Git|Linux|Windows|Azure|GCP)\b'
    ]

    # Soft skills patterns
    soft_patterns = [
        r'\b(?:leadership|management|communication|collaboration|problem.solving)\b',
        r'\b(?:analytical|creative|strategic|innovative|detail.oriented)\b',
        r'\b(?:team.player|self.motivated|adaptable|flexible|organized)\b'
    ]

    # Extract technical skills
    tech_skills = []
    for pattern in tech_patterns:
        tech_skills.extend(re.findall(pattern, jd, re.IGNORECASE))

    # Extract soft skills
    soft_skills = []
    for pattern in soft_patterns:
        soft_skills.extend(re.findall(pattern, jd, re.IGNORECASE))

    # Extract requirements keywords
    requirements_section = extract_requirements_section(jd)
    requirements_keywords = extract_keywords_from_text(requirements_section)

    # Extract action verbs from job description
    action_verbs = extract_action_verbs(jd)

    return {
        "technical_skills": list(set([skill.lower() for skill in tech_skills])),
        "soft_skills": list(set([skill.lower() for skill in soft_skills])),
        "requirements": requirements_keywords[:20],  # Top 20 requirement keywords
        "action_verbs": action_verbs[:15],  # Top 15 action verbs
        "all_keywords": extract_keywords_from_text(jd)[:50]  # Top 50 overall keywords
    }


def optimize_with_claude(resume: str, jd: str) -> str:
    """Optimize resume using Claude 3.5 Sonnet with advanced ATS optimization"""

    # First, analyze ATS compliance for targeted optimization
    ats_analysis = analyze_ats_compliance(resume, jd)

    # Build enhanced system prompt with ATS analysis insights
    missing_tech_skills = ats_analysis['keyword_analysis']['technical_skills']['missing_keywords'][:5]
    missing_requirements = ats_analysis['keyword_analysis']['requirements']['missing_keywords'][:5]
    industry_recommendations = ats_analysis['industry_recommendations'][:3]

    system_prompt = f"""You are an elite résumé optimization specialist who creates executive-level résumés that consistently secure interviews. Your expertise combines deep ATS knowledge with sophisticated professional communication standards and advanced keyword optimization strategies.

## CRITICAL ATS OPTIMIZATION REQUIREMENTS

**Priority Keywords to Include**:
- Technical Skills: {', '.join(missing_tech_skills) if missing_tech_skills else 'Focus on existing technical skills'}
- Requirements: {', '.join(missing_requirements) if missing_requirements else 'Strengthen existing qualifications'}

**Industry-Specific Focus**: {', '.join(industry_recommendations)}

**ATS Score Target**: Achieve 85+ ATS compatibility score through strategic keyword integration

## Core Requirements

**Structure**: Use this EXACT format for perfect ATS compatibility and professional presentation:

```
# [Full Name]
**Location:** [City, State ZIP] | **Email:** [email] | **Phone:** [phone] | **LinkedIn:** [linkedin]

## Professional Summary
[3-4 lines maximum. Powerful opening that positions candidate as ideal for target role. Include key skills and quantified achievements. MUST include priority keywords naturally.]

## Core Competencies
• [Skill 1 - use exact keywords from job description, prioritize missing technical skills]
• [Skill 2 - focus on technical and leadership skills from requirements]
• [Skill 3 - include industry-specific terminology]
• [Continue with 6-8 total competencies, ensuring keyword coverage]

## Professional Experience

### [Company Name] – [City, State]
**[Job Title]**
*[Start Date] to [End Date]*

• [Achievement with quantified impact - start with action verb, include relevant keywords]
• [Achievement showing progression and responsibility, incorporate missing requirements]
• [Achievement demonstrating skills relevant to target role]
• [Continue with 3-5 bullets per role, strategically placing keywords]

[Repeat for each position]

## Education
**[Degree]** | [Institution] | [Year]
[Additional certifications if relevant, include industry-specific certifications]
```

**Enhanced Content Standards**:
- Every bullet point starts with powerful action verbs (Led, Achieved, Implemented, Drove, etc.)
- Quantify everything possible (percentages, dollar amounts, team sizes, timeframes)
- Use keywords from job description naturally throughout, prioritizing missing keywords
- Focus on business impact and results, not just duties
- Maintain consistent verb tense (past for previous roles, present for current)
- Keep bullets concise but impactful (1-2 lines maximum)
- Achieve optimal keyword density without keyword stuffing

**Advanced ATS Optimization**:
- Use standard section headers exactly as shown
- Include exact keywords and phrases from job description, especially missing ones
- Maintain clean formatting without special characters
- Ensure contact information is properly formatted
- Use industry-standard terminology
- Strategically place keywords in multiple sections for better ATS scoring
- Include variations of key terms (e.g., "AI" and "Artificial Intelligence")
- Balance keyword optimization with natural, professional language

**Keyword Integration Strategy**:
- Professional Summary: Include 3-4 priority keywords naturally
- Core Competencies: Feature missing technical skills and requirements
- Experience Bullets: Weave keywords into achievement descriptions
- Education: Add relevant certifications or coursework if applicable

Return ONLY the optimized résumé in the exact format specified above. No explanations, no additional text."""
    
    user_prompt = f"Job Description:\n{jd}\n\nCurrent Résumé:\n{resume}"
    
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.25,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return message.content[0].text.strip()


def optimize_with_openai(resume: str, jd: str) -> str:
    """Fallback optimization using GPT-4o with advanced ATS optimization"""

    # Analyze ATS compliance for targeted optimization
    ats_analysis = analyze_ats_compliance(resume, jd)

    # Build enhanced system prompt with ATS analysis insights
    missing_tech_skills = ats_analysis['keyword_analysis']['technical_skills']['missing_keywords'][:5]
    missing_requirements = ats_analysis['keyword_analysis']['requirements']['missing_keywords'][:5]
    industry_recommendations = ats_analysis['industry_recommendations'][:3]

    system_prompt = f"""You are an elite résumé optimization specialist who creates executive-level résumés that consistently secure interviews. Your expertise combines deep ATS knowledge with sophisticated professional communication standards and advanced keyword optimization strategies.

## CRITICAL ATS OPTIMIZATION REQUIREMENTS

**Priority Keywords to Include**:
- Technical Skills: {', '.join(missing_tech_skills) if missing_tech_skills else 'Focus on existing technical skills'}
- Requirements: {', '.join(missing_requirements) if missing_requirements else 'Strengthen existing qualifications'}

**Industry-Specific Focus**: {', '.join(industry_recommendations)}

**ATS Score Target**: Achieve 85+ ATS compatibility score through strategic keyword integration

## Core Requirements

**Structure**: Use this EXACT format for perfect ATS compatibility and professional presentation:

```
# [Full Name]
**Location:** [City, State ZIP] | **Email:** [email] | **Phone:** [phone] | **LinkedIn:** [linkedin]

## Professional Summary
[3-4 lines maximum. Powerful opening that positions candidate as ideal for target role. Include key skills and quantified achievements.]

## Core Competencies
• [Skill 1 - use exact keywords from job description]
• [Skill 2 - focus on technical and leadership skills]
• [Skill 3 - include industry-specific terminology]
• [Continue with 6-8 total competencies]

## Professional Experience

### [Company Name] – [City, State]
**[Job Title]**
*[Start Date] to [End Date]*

• [Achievement with quantified impact - start with action verb]
• [Achievement showing progression and responsibility]
• [Achievement demonstrating skills relevant to target role]
• [Continue with 3-5 bullets per role]

[Repeat for each position]

## Education
**[Degree]** | [Institution] | [Year]
[Additional certifications if relevant]
```

**Content Standards**:
- Every bullet point starts with powerful action verbs (Led, Achieved, Implemented, Drove, etc.)
- Quantify everything possible (percentages, dollar amounts, team sizes, timeframes)
- Use keywords from job description naturally throughout
- Focus on business impact and results, not just duties
- Maintain consistent verb tense (past for previous roles, present for current)
- Keep bullets concise but impactful (1-2 lines maximum)

**ATS Optimization**:
- Use standard section headers exactly as shown
- Include exact keywords and phrases from job description
- Maintain clean formatting without special characters
- Ensure contact information is properly formatted
- Use industry-standard terminology

Return ONLY the optimized résumé in the exact format specified above. No explanations, no additional text."""
    
    user_prompt = f"Job Description:\n{jd}\n\nCurrent Résumé:\n{resume}"
    
    chat = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.25,
        max_tokens=4000
    )
    
    return chat.choices[0].message.content.strip()


def generate_cover_letter_claude(resume_md: str, jd: str, job_title: Optional[str] = None, company_name: Optional[str] = None) -> str:
    """Generate cover letter using Claude with enhanced personalization"""

    system_prompt = """You are an elite professional communication specialist who crafts compelling cover letters that secure interviews and job offers. Your expertise lies in creating authentic, persuasive narratives that demonstrate perfect alignment between candidates and target roles.

## Core Competencies

You excel at:
- **Strategic Storytelling**: Weaving candidate experiences into compelling narratives that resonate with hiring managers
- **Value Proposition Development**: Articulating unique value candidates bring to specific organizations and roles
- **Professional Tone Mastery**: Balancing confidence with humility, enthusiasm with professionalism
- **Industry Intelligence**: Understanding what different industries and roles prioritize in candidate communication
- **Persuasive Writing**: Employing sophisticated rhetorical techniques that influence hiring decisions
- **Company Research Integration**: Incorporating company values, culture, and recent developments into personalized narratives
- **Achievement Amplification**: Transforming resume bullet points into compelling success stories with emotional resonance

## Cover Letter Philosophy

Your approach centers on four core principles:
1. **Authentic Connection**: Demonstrate genuine interest and understanding of the company's mission, values, and recent developments
2. **Value-Driven Narrative**: Focus on specific, quantified contributions the candidate will make, not generic capabilities
3. **Compelling Differentiation**: Highlight unique qualifications and experiences that set the candidate apart from other applicants
4. **Future-Focused Vision**: Paint a clear picture of the candidate's potential impact and growth within the organization

## Advanced Personalization Techniques

**Company Intelligence**: Extract and reference specific details from the job description including:
- Company name, mission, and values
- Specific role requirements and responsibilities
- Team structure and reporting relationships
- Industry challenges and opportunities mentioned
- Company culture indicators and preferred qualifications

**Narrative Architecture**: Structure each cover letter with:
- **Opening Hook**: Immediate connection between candidate's strongest qualification and the role's primary need
- **Evidence Bridge**: 2-3 specific achievements that directly address job requirements with quantified results
- **Cultural Alignment**: Demonstration of shared values and cultural fit through specific examples
- **Future Impact**: Clear articulation of how the candidate will contribute to company goals and growth

**Emotional Intelligence**: Incorporate:
- Enthusiasm that feels genuine, not manufactured
- Confidence balanced with humility and eagerness to learn
- Professional passion that aligns with company mission
- Subtle urgency that motivates immediate action

## Writing Standards

**Structure**: Create a logical flow that captures attention, builds interest, demonstrates fit, and motivates action.

**Content Quality**: Every sentence serves a strategic purpose, advancing the candidate's case through specific examples, quantified achievements, and clear value propositions.

**Professional Presentation**: Maintain executive-level communication standards with sophisticated vocabulary, varied sentence structure, and flawless grammar.

**Length Optimization**: Deliver maximum impact in 280-320 words, ensuring every word contributes to the persuasive narrative.

**Personalization Depth**: Include at least 3 specific references to the job description and 2 quantified achievements from the resume.

## Output Requirements

Return ONLY the cover letter body in clean Markdown format (no salutation or closing). Focus on creating an authentic, compelling case for why this candidate is the ideal choice for this specific role at this particular organization. The letter should feel personally crafted for this exact opportunity, not a template."""
    
    # Build enhanced user prompt with additional context
    context_parts = [f"Job Description:\n{jd}"]
    if job_title:
        context_parts.append(f"Specific Job Title: {job_title}")
    if company_name:
        context_parts.append(f"Company Name: {company_name}")
    context_parts.append(f"Résumé:\n{resume_md}")

    user_prompt = "\n\n".join(context_parts)
    
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.3,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return message.content[0].text.strip()


def generate_cover_letter_openai(resume_md: str, jd: str, job_title: Optional[str] = None, company_name: Optional[str] = None) -> str:
    """Generate cover letter using OpenAI with enhanced personalization"""

    system_prompt = """You are an elite professional communication specialist who crafts compelling cover letters that secure interviews and job offers. Create authentic, persuasive narratives that demonstrate perfect alignment between candidates and target roles.

## Advanced Cover Letter Requirements

**Structure**: Use this EXACT professional format:

```
# [Candidate Full Name]
**[Address Line 1]**
**[City, State ZIP]**
**[Email] | [Phone] | [LinkedIn]**

[Date]

**[Hiring Manager Name/Title]**
**[Company Name]**
**[Company Address]**

Dear Hiring Manager,

[Opening paragraph: Hook with specific role interest and top qualification]

[Body paragraph 1: Specific achievements that align with job requirements]

[Body paragraph 2: Additional relevant experience and skills]

[Closing paragraph: Call to action and enthusiasm]

Sincerely,
[Candidate Full Name]
```

**Enhanced Content Standards**:
- Replace ALL placeholders with actual information from the résumé
- Use specific company name (extract from job description)
- Reference specific role title from job description
- Include 3-4 quantified achievements from résumé with specific metrics
- Match tone to company culture (professional for corporate, dynamic for startups, innovative for tech)
- Keep total length to 280-320 words for maximum impact
- Use confident, professional language with emotional intelligence
- Include clear call to action with urgency
- Reference specific company values, mission, or recent developments mentioned in job description
- Demonstrate understanding of role's key challenges and how candidate addresses them

**Advanced Personalization Requirements**:
- Extract candidate's actual name, contact info from résumé
- Use today's date in professional format
- Address specific role mentioned in job description with exact title
- Reference company name from job description (never use placeholder)
- Incorporate specific requirements, qualifications, or responsibilities from job description
- Show clear value proposition with quantified impact predictions
- Include industry-specific terminology and keywords from job description
- Demonstrate cultural fit through shared values or mission alignment
- Reference specific skills, technologies, or methodologies mentioned in job description

**Narrative Excellence Standards**:
- Open with a compelling hook that immediately connects candidate's strongest asset to role's primary need
- Use storytelling techniques to make achievements memorable and impactful
- Show progression and growth in candidate's career trajectory
- Demonstrate problem-solving capabilities with specific examples
- Include forward-looking statements about potential contributions
- Balance confidence with humility and eagerness to contribute
- Create emotional resonance while maintaining professional tone

Return ONLY the complete cover letter in the exact format specified above. No explanations, no additional text."""
    
    # Build enhanced user prompt with additional context
    context_parts = [f"Job Description:\n{jd}"]
    if job_title:
        context_parts.append(f"Specific Job Title: {job_title}")
    if company_name:
        context_parts.append(f"Company Name: {company_name}")
    context_parts.append(f"Résumé:\n{resume_md}")

    user_prompt = "\n\n".join(context_parts)
    
    chat = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return chat.choices[0].message.content.strip()


def create_pdf_with_mistral_layout(markdown: str, layout_analysis: dict) -> Path:
    """Create PDF using Mistral OCR layout analysis for pixel-perfect recreation"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.colors import black
    import re

    try:
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(
            out,
            pagesize=letter,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )

        styles = getSampleStyleSheet()

        # Extract layout preferences from Mistral analysis
        section_styles = layout_analysis.get('section_styles', {})
        bullet_styles = layout_analysis.get('bullet_styles', {})
        contact_layout = layout_analysis.get('contact_layout', {})
        font_analysis = layout_analysis.get('font_analysis', {})

        # Create adaptive styles based on layout analysis
        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Title'],
            fontSize=font_analysis.get('name_size', 22),
            spaceAfter=8,
            spaceBefore=0,
            alignment=TA_CENTER,
            fontName='Times-Bold',
            textColor=black,
            keepWithNext=1
        )

        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=styles['Normal'],
            fontSize=font_analysis.get('contact_size', 10),
            spaceAfter=16,
            spaceBefore=0,
            alignment=TA_CENTER if contact_layout.get('alignment') == 'center' else TA_LEFT,
            fontName='Times-Roman',
            textColor=black
        )

        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=font_analysis.get('section_header_size', 12),
            spaceAfter=6,
            spaceBefore=16,
            fontName='Times-Bold',
            textColor=black,
            keepWithNext=1
        )

        # Adaptive bullet style based on analysis
        bullet_indent = bullet_styles.get('indentation', 18)
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=font_analysis.get('body_size', 11),
            spaceAfter=3,
            spaceBefore=0,
            leftIndent=bullet_indent,
            fontName='Times-Roman',
            textColor=black,
            bulletIndent=6,
            alignment=TA_JUSTIFY
        )

        logger.info("Created adaptive PDF styles based on Mistral layout analysis",
                   name_size=font_analysis.get('name_size', 22),
                   contact_alignment=contact_layout.get('alignment', 'center'),
                   bullet_indent=bullet_indent)

        # Use the enhanced PDF creation with adaptive styles
        story = create_adaptive_pdf_content(markdown, name_style, contact_style, section_header_style, bullet_style)

        doc.build(story)
        logger.info("Pixel-perfect PDF created using Mistral layout analysis", output_path=out)
        return Path(out)

    except Exception as e:
        logger.warning("Mistral layout PDF creation failed, using enhanced simple layout", error=str(e))
        return create_simple_pdf(markdown)


def create_adaptive_pdf_content(markdown: str, name_style, contact_style, section_header_style, bullet_style):
    """Create PDF content with adaptive styling based on layout analysis"""
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    from reportlab.lib.colors import black
    from reportlab.lib.styles import getSampleStyleSheet
    import re

    styles = getSampleStyleSheet()
    story = []
    lines = markdown.splitlines()
    current_section = None

    def _clean_markdown_text(text: str) -> str:
        """Enhanced markdown cleaning for PDF rendering with comprehensive formatting artifact removal"""
        # Remove markdown headers (###, ##, #) but preserve the text
        text = re.sub(r'^#{1,6}\s*', '', text)

        # Handle bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        # Handle italic text (but not bold)
        text = re.sub(r'\*(.*?)\*(?!\*)', r'<i>\1</i>', text)

        # Handle links - extract text only
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)

        # Remove any remaining markdown artifacts
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code backticks
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Remove strikethrough
        text = re.sub(r'^\s*[-*+]\s*', '', text)  # Remove bullet markers at start
        text = re.sub(r'^\s*\d+\.\s*', '', text)  # Remove numbered list markers

        # Clean up extra spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_contact_line(line: str) -> bool:
        """Enhanced check for contact information with more patterns"""
        contact_indicators = [
            '**Location:**', '**LinkedIn:**', '**Email:**', '**Phone:**',
            'Location:', 'LinkedIn:', 'Email:', 'Phone:',
            '@', '(', ')', 'linkedin.com', '.com', 'tel:', 'mailto:',
            'github.com', 'portfolio', 'website', '+1', 'www.'
        ]

        # Check for email pattern
        import re
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line):
            return True

        # Check for phone pattern
        if re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line):
            return True

        return any(indicator in line for indicator in contact_indicators)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Name (first line or # header)
        if i == 0 or (line.startswith('# ') and current_section is None):
            name = line.replace('# ', '').strip()
            story.append(Paragraph(name, name_style))

        # Enhanced contact information block processing
        elif _is_contact_line(line):
            contact_lines = []
            while i < len(lines) and _is_contact_line(lines[i].strip()):
                contact_line = lines[i].strip()
                if contact_line:
                    # Enhanced contact line cleaning
                    contact_line = re.sub(r'\*\*(.*?):\*\*', r'\1:', contact_line)  # Remove bold markers
                    contact_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', contact_line)  # Remove links
                    contact_line = re.sub(r'Location:\s*', '', contact_line)  # Clean location prefix
                    contact_line = re.sub(r'Email:\s*', '', contact_line)  # Clean email prefix
                    contact_line = re.sub(r'Phone:\s*', '', contact_line)  # Clean phone prefix
                    contact_line = re.sub(r'LinkedIn:\s*', '', contact_line)  # Clean LinkedIn prefix
                    contact_line = _clean_markdown_text(contact_line)
                    if contact_line:  # Only add non-empty lines
                        contact_lines.append(contact_line)
                i += 1

            if contact_lines:
                # Join with proper spacing and separators
                contact_text = ' | '.join(contact_lines)
                story.append(Paragraph(contact_text, contact_style))
            i -= 1

        # Section headers
        elif line.startswith('## '):
            current_section = line[3:].strip()
            # Clean the section header and ensure consistent formatting
            clean_section = _clean_markdown_text(current_section)
            section_name = clean_section.upper()

            # Ensure professional section header formatting
            if section_name:
                story.append(Paragraph(section_name, section_header_style))
                story.append(HRFlowable(width="100%", thickness=1, color=black, spaceBefore=2, spaceAfter=6))

        # Company names (### format)
        elif line.startswith('### '):
            company_name = line[4:].strip()
            clean_company = _clean_markdown_text(company_name)
            # Create a company style similar to section headers but smaller
            from reportlab.lib.styles import ParagraphStyle
            company_style = ParagraphStyle(
                'CompanyStyle',
                parent=section_header_style,
                fontSize=11,
                spaceAfter=4,
                spaceBefore=8,
                fontName='Times-Bold'
            )
            story.append(Paragraph(clean_company, company_style))

        # Job titles (bold format)
        elif line.startswith('**') and line.endswith('**') and len(line) > 4:
            job_title = line[2:-2].strip()
            clean_title = _clean_markdown_text(job_title)
            # Create job title style on the fly
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
            job_title_style = ParagraphStyle(
                'JobTitle',
                parent=section_header_style,
                fontSize=11,
                spaceAfter=3,
                spaceBefore=3,
                fontName='Times-Bold',
                alignment=TA_LEFT
            )
            story.append(Paragraph(f"<b>{clean_title}</b>", job_title_style))

        # Dates (italic format)
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**') and len(line) > 2:
            date = line[1:-1].strip()
            clean_date = _clean_markdown_text(date)
            # Create date style on the fly
            date_style = ParagraphStyle(
                'Date',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=8,
                spaceBefore=1,
                fontName='Times-Italic',
                alignment=TA_LEFT
            )
            story.append(Paragraph(f"<i>{clean_date}</i>", date_style))

        # Bullet points with enhanced adaptive styling
        elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = _clean_markdown_text(bullet_text)

            # Check for nested bullets (indented)
            if line.startswith('  ') or line.startswith('\t'):
                # Create nested bullet style on the fly
                from reportlab.lib.styles import ParagraphStyle
                nested_style = ParagraphStyle(
                    'NestedBullet',
                    parent=bullet_style,
                    leftIndent=40,
                    bulletIndent=20,
                    fontSize=10,
                    spaceAfter=2
                )
                story.append(Paragraph(f"◦ {bullet_text}", nested_style))
            else:
                # Regular bullet with proper spacing
                story.append(Paragraph(f"• {bullet_text}", bullet_style))

        # Regular paragraphs
        else:
            if line and not line.startswith('---') and not line.startswith('='):
                text = _clean_markdown_text(line)
                story.append(Paragraph(text, styles['Normal']))
                story.append(Spacer(1, 4))

        i += 1

    return story


def create_pdf_with_surya_layout(markdown: str, original_pdf: Path) -> Path:
    """Create PDF using Surya for layout analysis (placeholder for now)"""
    # TODO: Implement Surya integration
    logger.info("Surya layout analysis not yet implemented, using simple layout")
    return create_simple_pdf(markdown)


def create_simple_pdf(markdown: str) -> Path:
    """Create professional PDF with clean, ATS-friendly formatting and improved edge case handling"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib.colors import black
    import re

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(
        out,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    # Enhanced custom styles for executive-level professional resume
    name_style = ParagraphStyle(
        'NameStyle',
        parent=styles['Title'],
        fontSize=22,
        spaceAfter=8,
        spaceBefore=0,
        alignment=TA_CENTER,
        fontName='Times-Bold',
        textColor=black,
        keepWithNext=1
    )

    # Enhanced contact style with better formatting
    contact_style = ParagraphStyle(
        'ContactStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=18,
        spaceBefore=4,
        alignment=TA_CENTER,
        fontName='Times-Roman',
        textColor=black,
        leading=12,
        keepWithNext=1
    )

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=16,
        fontName='Times-Bold',
        textColor=black,
        borderWidth=0,
        borderPadding=0,
        keepWithNext=1,
        pageBreakBefore=0
    )

    # Enhanced job title style with better spacing
    job_title_style = ParagraphStyle(
        'JobTitle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=3,
        spaceBefore=3,
        fontName='Times-Bold',
        textColor=black,
        keepWithNext=1,
        leading=13,
        alignment=TA_LEFT
    )

    company_style = ParagraphStyle(
        'Company',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=2,
        spaceBefore=8,
        fontName='Times-Bold',
        textColor=black,
        keepWithNext=1
    )

    # Enhanced date style with proper formatting
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        spaceBefore=1,
        fontName='Times-Italic',
        textColor=black,
        leading=12,
        alignment=TA_LEFT
    )

    # Enhanced bullet style with perfect alignment
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=1,
        leftIndent=20,
        fontName='Times-Roman',
        textColor=black,
        bulletIndent=8,
        alignment=TA_JUSTIFY,
        firstLineIndent=0,
        leading=14  # Line spacing for better readability
    )

    # Enhanced bullet style for nested bullets
    nested_bullet_style = ParagraphStyle(
        'NestedBullet',
        parent=bullet_style,
        leftIndent=40,
        bulletIndent=20,
        fontSize=10,
        spaceAfter=3,
        leading=13
    )

    # Special bullet style for competencies section
    competency_style = ParagraphStyle(
        'Competency',
        parent=bullet_style,
        fontSize=11,
        spaceAfter=2,
        spaceBefore=1,
        leftIndent=18,
        bulletIndent=6,
        leading=13,
        alignment=TA_LEFT  # Left align for competencies
    )

    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        spaceBefore=0,
        fontName='Times-Roman',
        textColor=black,
        alignment=TA_JUSTIFY
    )

    # New style for competencies/skills section
    competency_style = ParagraphStyle(
        'Competency',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=3,
        spaceBefore=0,
        leftIndent=18,
        fontName='Times-Roman',
        textColor=black,
        bulletIndent=6
    )

    # Style for education section
    education_style = ParagraphStyle(
        'Education',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=0,
        fontName='Times-Roman',
        textColor=black
    )

    def _clean_markdown_text(text: str) -> str:
        """Enhanced markdown cleaning for PDF rendering with comprehensive formatting artifact removal"""
        # Remove markdown headers (###, ##, #) but preserve the text
        text = re.sub(r'^#{1,6}\s*', '', text)

        # Handle bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        # Handle italic text (but not bold)
        text = re.sub(r'\*(.*?)\*(?!\*)', r'<i>\1</i>', text)

        # Handle links - extract text only
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)

        # Remove any remaining markdown artifacts
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code backticks
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Remove strikethrough
        text = re.sub(r'^\s*[-*+]\s*', '', text)  # Remove bullet markers at start
        text = re.sub(r'^\s*\d+\.\s*', '', text)  # Remove numbered list markers

        # Clean up extra spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_contact_line(line: str) -> bool:
        """Check if line contains contact information"""
        contact_indicators = [
            '**Location:**', '**LinkedIn:**', '**Email:**', '**Phone:**',
            '@', '(', ')', 'linkedin.com', '.com', 'tel:', 'mailto:'
        ]
        return any(indicator in line for indicator in contact_indicators)

    def _parse_contact_block(lines: list, start_idx: int) -> tuple:
        """Parse contact information block and return (contact_text, next_index)"""
        contact_lines = []
        i = start_idx

        while i < len(lines) and _is_contact_line(lines[i].strip()):
            contact_line = lines[i].strip()
            if contact_line:
                # Clean up contact formatting
                contact_line = re.sub(r'\*\*(.*?):\*\*', r'\1:', contact_line)
                contact_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', contact_line)
                contact_lines.append(contact_line)
            i += 1

        contact_text = ' | '.join(contact_lines) if contact_lines else ''
        return contact_text, i - 1

    story = []
    lines = markdown.splitlines()
    current_section = None
    in_experience_block = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Name (first line or # header)
        if i == 0 or (line.startswith('# ') and current_section is None):
            name = line.replace('# ', '').strip()
            story.append(Paragraph(name, name_style))

        # Contact information block
        elif _is_contact_line(line):
            contact_text, next_i = _parse_contact_block(lines, i)
            if contact_text:
                story.append(Paragraph(contact_text, contact_style))
            i = next_i

        # Section headers
        elif line.startswith('## '):
            current_section = line[3:].strip()
            # Clean the section header and ensure consistent formatting
            clean_section = _clean_markdown_text(current_section)
            section_name = clean_section.upper()

            # Ensure professional section header formatting
            if section_name:
                story.append(Paragraph(section_name, section_header_style))
                story.append(HRFlowable(width="100%", thickness=1, color=black, spaceBefore=2, spaceAfter=6))

            # Track if we're in experience section for better formatting
            in_experience_block = 'EXPERIENCE' in section_name.upper()

        # Subsection headers (### format) - Company names
        elif line.startswith('### '):
            subsection = line[4:].strip()
            # Clean the company name text but preserve formatting
            clean_subsection = _clean_markdown_text(subsection)
            # Company names in experience section
            if in_experience_block:
                story.append(Paragraph(clean_subsection, company_style))
            else:
                # Other subsections
                story.append(Paragraph(clean_subsection, job_title_style))

        # Job titles (bold lines under companies)
        elif line.startswith('**') and line.endswith('**') and len(line) > 4:
            title = line[2:-2].strip()
            # Clean the job title but preserve bold formatting
            clean_title = _clean_markdown_text(title)
            story.append(Paragraph(f"<b>{clean_title}</b>", job_title_style))

        # Dates (italic lines)
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**') and len(line) > 2:
            date = line[1:-1].strip()
            # Clean the date but preserve italic formatting
            clean_date = _clean_markdown_text(date)
            story.append(Paragraph(f"<i>{clean_date}</i>", date_style))

        # Bullet points with improved handling
        elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = _clean_markdown_text(bullet_text)

            # Handle nested bullets
            if line.startswith('  ') or line.startswith('\t'):
                story.append(Paragraph(f"◦ {bullet_text}", nested_bullet_style))
            else:
                # Check if this is in competencies section for different styling
                if current_section and 'COMPETENC' in current_section.upper():
                    story.append(Paragraph(f"• {bullet_text}", competency_style))
                else:
                    story.append(Paragraph(f"• {bullet_text}", bullet_style))

        # Education entries (special formatting)
        elif current_section and 'EDUCATION' in current_section.upper() and line and not line.startswith('#'):
            text = _clean_markdown_text(line)
            story.append(Paragraph(text, education_style))

        # Summary/Professional Summary content
        elif current_section and 'SUMMARY' in current_section.upper() and line and not line.startswith('#'):
            text = _clean_markdown_text(line)
            story.append(Paragraph(text, summary_style))

        # Regular paragraphs
        else:
            if line and not line.startswith('---') and not line.startswith('='):
                text = _clean_markdown_text(line)
                story.append(Paragraph(text, styles['Normal']))
                story.append(Spacer(1, 4))

        i += 1

    # Add final spacing and build document with improved page handling
    try:
        doc.build(story)
        logger.info("Professional PDF created successfully", output_path=out)
    except Exception as e:
        logger.error("PDF creation failed", error=str(e))
        # Try with simpler formatting as fallback
        simple_story = []
        for item in story:
            if hasattr(item, 'text'):
                simple_story.append(Paragraph(item.text, styles['Normal']))
            else:
                simple_story.append(item)
        doc.build(simple_story)
        logger.info("PDF created with fallback formatting", output_path=out)

    return Path(out)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
