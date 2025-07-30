import asyncio
import json
import base64
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from pathlib import Path
import structlog

logger = structlog.get_logger()

class AIProcessor:
    """
    Enhanced AI processing service based on reference.md
    Handles resume optimization and cover letter generation
    """
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model_text = "gpt-4o-mini"
        self.model_vision = "gpt-4o"
    
    async def optimize_resume_text(self, resume_text: str, job_description: str) -> Dict:
        """
        Enhanced resume optimization from reference.md
        Returns structured analysis and optimized content
        """
        system_prompt = """You are an expert ATS resume optimizer. Your task is to:

1. Analyze the resume against the job description
2. Identify missing keywords and skills
3. Rewrite content to include relevant keywords naturally
4. Maintain truthfulness while optimizing for ATS
5. Return structured JSON with analysis and optimized content

Response must be valid JSON with this exact structure:
{
  "resume_analysis": {
    "extracted_keywords": ["keyword1", "keyword2"],
    "missing_skills": ["skill1", "skill2"],
    "section_map": {"Summary": true, "Experience": true, "Skills": true, "Education": true},
    "ats_score": 85,
    "recommendations": ["recommendation1", "recommendation2"]
  },
  "ats_keyword_map": [
    {"keyword": "Python", "present": true, "action": "emphasized in skills section"},
    {"keyword": "Machine Learning", "present": false, "action": "added to relevant experience"}
  ],
  "optimized_resume_markdown": "# Full optimized resume in markdown format...",
  "change_log": [
    {"section": "Summary", "old": "original text", "new": "optimized text", "reason": "added key terms"}
  ]
}"""

        user_prompt = f"""
JOB DESCRIPTION:
{job_description}

CURRENT RESUME:
{resume_text}

Optimize this resume for the job description. Focus on:
- Natural keyword integration
- Quantified achievements
- ATS-friendly formatting
- Truthful enhancements only
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_text,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                logger.info("Resume optimization completed", tokens_used=response.usage.total_tokens)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("AI response was not valid JSON, using fallback")
                return {
                    "resume_analysis": {
                        "extracted_keywords": [],
                        "missing_skills": [],
                        "section_map": {},
                        "ats_score": 0,
                        "recommendations": ["AI processing error - manual review needed"]
                    },
                    "ats_keyword_map": [],
                    "optimized_resume_markdown": content,
                    "change_log": []
                }
                
        except Exception as e:
            logger.error("Resume optimization failed", error=str(e))
            raise ValueError(f"AI processing failed: {str(e)}")
    
    async def generate_cover_letter(self, optimized_resume: str, job_description: str) -> Dict:
        """
        Generate matching cover letter (from reference.md)
        Returns structured result with cover letter text and metadata
        """
        system_prompt = """Write a professional cover letter that:
- Is 250-300 words maximum
- Matches the resume content and job requirements
- Uses specific examples from the resume
- Shows enthusiasm for the role
- Has a professional but engaging tone
- Return ONLY the letter body (no header/footer)"""

        user_prompt = f"""
OPTIMIZED RESUME:
{optimized_resume}

JOB DESCRIPTION:
{job_description}

Write a compelling cover letter for this position."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_text,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )

            cover_letter = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            logger.info("Cover letter generated",
                       length=len(cover_letter),
                       tokens_used=tokens_used)

            return {
                "cover_letter_text": cover_letter,
                "tokens_used": tokens_used,
                "word_count": len(cover_letter.split()),
                "character_count": len(cover_letter)
            }

        except Exception as e:
            logger.error("Cover letter generation failed", error=str(e))
            raise ValueError(f"Cover letter generation failed: {str(e)}")
    
    async def extract_layout_coordinates(self, png_path: Path) -> Dict:
        """
        Use GPT-4o Vision to extract layout coordinates for PDF reconstruction
        This is the key innovation from reference.md
        """
        try:
            # Read and encode image
            with open(png_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            system_prompt = """You are a layout analysis expert. Analyze this resume image and return JSON with precise bounding box coordinates for each text section.

Return this exact JSON structure:
{
  "page_dimensions": {"width": 612, "height": 792},
  "sections": [
    {
      "type": "header",
      "bbox": [x1, y1, x2, y2],
      "content_type": "name_and_contact"
    },
    {
      "type": "summary", 
      "bbox": [x1, y1, x2, y2],
      "content_type": "professional_summary"
    },
    {
      "type": "experience",
      "bbox": [x1, y1, x2, y2], 
      "content_type": "work_experience"
    },
    {
      "type": "skills",
      "bbox": [x1, y1, x2, y2],
      "content_type": "technical_skills"
    },
    {
      "type": "education",
      "bbox": [x1, y1, x2, y2],
      "content_type": "education_section"
    }
  ],
  "fonts": [
    {"name": "Arial", "size": 12, "weight": "normal"},
    {"name": "Arial", "size": 16, "weight": "bold"}
  ]
}

Coordinates should be in points (72 DPI). Be precise with bounding boxes."""

            response = await self.client.chat.completions.create(
                model=self.model_vision,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            layout_data = json.loads(content)
            logger.info("Layout coordinates extracted", sections_count=len(layout_data.get("sections", [])))
            return layout_data
            
        except Exception as e:
            logger.error("Layout extraction failed", error=str(e), png_path=str(png_path))
            # Return fallback layout for standard resume
            return {
                "page_dimensions": {"width": 612, "height": 792},
                "sections": [
                    {"type": "header", "bbox": [50, 720, 562, 780], "content_type": "name_and_contact"},
                    {"type": "summary", "bbox": [50, 650, 562, 710], "content_type": "professional_summary"},
                    {"type": "experience", "bbox": [50, 400, 562, 640], "content_type": "work_experience"},
                    {"type": "skills", "bbox": [50, 300, 562, 390], "content_type": "technical_skills"},
                    {"type": "education", "bbox": [50, 200, 562, 290], "content_type": "education_section"}
                ],
                "fonts": [{"name": "Arial", "size": 11, "weight": "normal"}]
            }