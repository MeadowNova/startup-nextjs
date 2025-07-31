"""
Simple AI Processor using proven prompts from reference.md
Handles resume optimization and cover letter generation with OpenAI
"""

from typing import Optional
from openai import AsyncOpenAI
import structlog

logger = structlog.get_logger()


class AIProcessor:
    """
    Simple AI processing service using proven prompts from reference.md
    Handles resume optimization and cover letter generation
    """

    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model_text = "gpt-4o-mini"

        # Simple, proven prompts from reference.md
        self.SYS_RESUME = """You are an expert résumé writer who beats ATS.
Rewrite the résumé below so it mirrors the job description keywords while remaining truthful.
Return ONLY plain Markdown (no ```markdown fence).
Keep sections: Summary, Skills, Experience, Education, etc.
Make every bullet start with action verbs and quantify impact."""

        self.SYS_COVER = """Write a concise 250-word cover letter in Markdown that matches the résumé and job description.
Return ONLY the letter body (no salutation block)."""
    
    async def optimize_resume_text(self, resume_text: str, job_description: str) -> str:
        """
        Simple resume optimization using proven prompts from reference.md
        Returns optimized resume in markdown format
        """
        try:
            prompt = f"Job Description:\n{job_description}\n\nCurrent Résumé:\n{resume_text}"

            response = await self.client.chat.completions.create(
                model=self.model_text,
                messages=[
                    {"role": "system", "content": self.SYS_RESUME},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25
            )

            optimized_resume = response.choices[0].message.content.strip()
            logger.info("Resume optimization completed successfully")
            return optimized_resume

        except Exception as e:
            logger.error(f"Resume optimization failed: {str(e)}")
            raise

    async def generate_cover_letter(
        self,
        resume_markdown: str,
        job_description: str,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> str:
        """
        Generate cover letter using proven prompts from reference.md
        Returns cover letter in markdown format
        """
        try:
            prompt = f"Résumé:\n{resume_markdown}\n\nJob Description:\n{job_description}"

            response = await self.client.chat.completions.create(
                model=self.model_text,
                messages=[
                    {"role": "system", "content": self.SYS_COVER},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            cover_letter = response.choices[0].message.content.strip()
            logger.info("Cover letter generation completed successfully")
            return cover_letter

        except Exception as e:
            logger.error(f"Cover letter generation failed: {str(e)}")
            raise