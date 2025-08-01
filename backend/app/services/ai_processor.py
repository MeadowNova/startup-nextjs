"""
Simple AI Processor using proven prompts from reference.md
Handles resume optimization and cover letter generation with OpenAI
"""

from typing import Optional
import structlog
import asyncio
import time

logger = structlog.get_logger()


class AIProcessor:
    """
    Simple AI processing service using proven prompts from reference.md
    Handles resume optimization and cover letter generation
    """

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.client = None
        self.model_text = "gpt-4o-mini"
        self.fallback_mode = False

        # Simple, proven prompts from reference.md
        self.SYS_RESUME = """You are an expert résumé writer who beats ATS.
Rewrite the résumé below so it mirrors the job description keywords while remaining truthful.
Return ONLY plain Markdown (no ```markdown fence).
Keep sections: Summary, Skills, Experience, Education, etc.
Make every bullet start with action verbs and quantify impact."""

        self.SYS_COVER = """Write a concise 250-word cover letter in Markdown that matches the résumé and job description.
Return ONLY the letter body (no salutation block)."""

        self._initialize_client_with_strategies()

    def _initialize_client_with_strategies(self):
        """Initialize OpenAI client with multiple fallback strategies"""
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            self.fallback_mode = True
            return

        strategies = [
            self._strategy_basic,
            self._strategy_with_timeout,
            self._strategy_minimal
        ]

        for i, strategy in enumerate(strategies, 1):
            try:
                self.client = strategy()
                logger.info(f"OpenAI client initialized successfully (strategy {i})")
                return
            except Exception as e:
                logger.warning(f"OpenAI initialization strategy {i} failed: {e}")

        logger.error("All OpenAI initialization strategies failed, using fallback mode")
        self.fallback_mode = True

    def _strategy_basic(self):
        """Basic OpenAI client initialization"""
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        except ImportError:
            raise Exception("OpenAI library not available")

    def _strategy_with_timeout(self):
        """OpenAI client with extended timeout"""
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, timeout=60.0)
        except ImportError:
            raise Exception("OpenAI library not available")

    def _strategy_minimal(self):
        """Minimal OpenAI client configuration"""
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, max_retries=1)
        except ImportError:
            raise Exception("OpenAI library not available")
    
    async def optimize_resume_text(self, resume_text: str, job_description: str) -> str:
        """
        Simple resume optimization using proven prompts from reference.md
        Returns optimized resume in markdown format
        """
        if self.fallback_mode or self.client is None:
            logger.warning("Using fallback optimization (OpenAI not available)")
            return self._create_fallback_optimized_resume(resume_text, job_description)

        try:
            return await self._openai_optimize_resume(resume_text, job_description)
        except Exception as e:
            logger.warning(f"OpenAI optimization failed: {e}, using fallback")
            return self._create_fallback_optimized_resume(resume_text, job_description)

    async def _openai_optimize_resume(self, resume_text: str, job_description: str) -> str:
        """Perform OpenAI-based resume optimization"""
        prompt = f"Job Description:\n{job_description}\n\nCurrent Résumé:\n{resume_text}"

        start_time = time.time()
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_text,
            messages=[
                {"role": "system", "content": self.SYS_RESUME},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=2000
        )

        processing_time = time.time() - start_time
        logger.info(f"OpenAI resume optimization completed in {processing_time:.2f}s")

        optimized_resume = response.choices[0].message.content.strip()
        logger.info("Resume optimization completed successfully")
        return optimized_resume

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
        if self.fallback_mode or self.client is None:
            logger.warning("Using fallback cover letter (OpenAI not available)")
            return self._create_fallback_cover_letter(resume_markdown, job_description)

        try:
            return await self._openai_generate_cover_letter(resume_markdown, job_description, job_title, company_name)
        except Exception as e:
            logger.warning(f"OpenAI cover letter generation failed: {e}, using fallback")
            return self._create_fallback_cover_letter(resume_markdown, job_description)

    async def _openai_generate_cover_letter(self, resume_markdown: str, job_description: str, job_title: Optional[str], company_name: Optional[str]) -> str:
        """Generate cover letter using OpenAI"""
        # Build context
        context_parts = [f"Job Description:\n{job_description}"]
        if job_title:
            context_parts.append(f"Job Title: {job_title}")
        if company_name:
            context_parts.append(f"Company: {company_name}")
        context_parts.append(f"Optimized Résumé:\n{resume_markdown}")

        prompt = "\n\n".join(context_parts)

        start_time = time.time()
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_text,
            messages=[
                {"role": "system", "content": self.SYS_COVER},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        processing_time = time.time() - start_time
        logger.info(f"OpenAI cover letter generation completed in {processing_time:.2f}s")

        cover_letter = response.choices[0].message.content.strip()
        logger.info("Cover letter generation completed successfully")
        return cover_letter

    def _create_fallback_optimized_resume(self, resume_text: str, job_description: str) -> str:
        """Fallback resume optimization when OpenAI is not available"""
        logger.info("Creating fallback optimized resume")

        # Simple keyword matching and formatting
        optimized = f"""# Optimized Resume

## Summary
Experienced professional with relevant skills for the target position.

## Skills
- Technical expertise
- Problem-solving abilities
- Team collaboration
- Project management

## Experience
{resume_text}

---
*Note: This is a fallback optimization. For full AI optimization, please ensure OpenAI API is properly configured.*
"""
        return optimized

    def _create_fallback_cover_letter(self, resume_markdown: str, job_description: str) -> str:
        """Fallback cover letter when OpenAI is not available"""
        logger.info("Creating fallback cover letter")

        cover_letter = f"""# Cover Letter

Dear Hiring Manager,

I am writing to express my interest in the position described in your job posting. Based on my background and experience, I believe I would be a strong candidate for this role.

My qualifications include relevant experience and skills that align with your requirements. I am excited about the opportunity to contribute to your team and would welcome the chance to discuss how my background can benefit your organization.

Thank you for your consideration. I look forward to hearing from you.

Sincerely,
[Your Name]

---
*Note: This is a fallback cover letter. For full AI-generated cover letters, please ensure OpenAI API is properly configured.*
"""
        return cover_letter