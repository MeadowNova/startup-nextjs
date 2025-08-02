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
        # Default values will be overridden if config is available
        self.model_text = "gpt-4o-mini"
        self.timeout = 60.0
        self.fallback_mode = False
        # Track fallback usage for metadata
        self.used_fallback_resume = False
        self.used_fallback_cover_letter = False
        # Non-sensitive diagnostic
        logger.info(
            "AIProcessor init",
            openai_key_present=bool(openai_api_key),
            openai_key_prefix=(openai_api_key[:5] + "***") if openai_api_key else "none",
            model_text=self.model_text,
            timeout=self.timeout,
        )

        # Check for placeholder/invalid API key
        if (
            not openai_api_key
            or openai_api_key.startswith('sk-test-placeholder')
            or openai_api_key == 'your-openai-api-key-here'
        ):
            logger.warning("OpenAI API key is not configured properly. Using fallback mode.")
            self.fallback_mode = True

        # Expert-level system prompts for professional resume optimization
        self.SYS_RESUME = """You are an elite resume optimization expert with 15+ years of experience helping candidates land roles at Fortune 500 companies. Your expertise includes ATS optimization, executive recruiting, and industry-specific resume strategies.

**CORE MISSION**: Transform the provided resume to maximize ATS compatibility and human appeal while maintaining 100% truthfulness.

**ATS OPTIMIZATION REQUIREMENTS**:
- Mirror job description keywords naturally throughout all sections
- Use exact terminology from the job posting when applicable
- Include industry-standard skill names and certifications
- Optimize for keyword density without stuffing
- Use ATS-friendly formatting (clear headers, bullet points, standard fonts)

**CONTENT ENHANCEMENT STRATEGY**:
- Start every bullet point with powerful action verbs (Led, Developed, Implemented, Optimized, etc.)
- Quantify ALL achievements with specific metrics (%, $, #, timeframes)
- Use the STAR method (Situation, Task, Action, Result) for experience bullets
- Highlight transferable skills that match job requirements
- Include relevant technical skills, tools, and methodologies mentioned in job description

**SECTION OPTIMIZATION**:
- **Professional Summary**: 3-4 lines highlighting most relevant qualifications and achievements
- **Core Competencies/Skills**: Prioritize skills mentioned in job description
- **Professional Experience**: Focus on accomplishments over responsibilities
- **Education**: Include relevant coursework, projects, or certifications if applicable
- **Additional Sections**: Add relevant sections (Certifications, Projects, Publications) if they strengthen candidacy

**CRITICAL FORMATTING REQUIREMENTS**:
- Return ONLY clean Markdown format - NO HTML tags whatsoever
- Use **bold** for emphasis - NEVER use <b> or </b> tags
- Use *italic* for emphasis - NEVER use <i> or </i> tags
- Use consistent bullet point formatting with hyphens (-)
- Maintain professional section hierarchy (# ## ###)
- Ensure proper spacing and readability
- Keep total length appropriate for experience level (1-2 pages equivalent)
- ABSOLUTELY NO HTML TAGS - only pure Markdown syntax
- If you use any HTML tags like <b>, <i>, <strong>, <em>, the output will fail

**QUALITY STANDARDS**:
- Every statement must be truthful and verifiable
- Language should be professional and confident
- Eliminate redundancy and weak language
- Ensure grammatical perfection and consistent tense usage
- Tailor content specifically to the target role and industry

Transform the resume to be irresistible to both ATS systems and hiring managers."""

        self.SYS_COVER = """You are an expert cover letter writer specializing in executive-level business correspondence. Your letters consistently help candidates secure interviews at top-tier companies.

**OBJECTIVE**: Create a compelling, professional cover letter that demonstrates perfect alignment between the candidate's background and the target role.

**STRUCTURE REQUIREMENTS**:
- **Opening Paragraph**: Strong hook that immediately establishes relevance and enthusiasm
- **Body Paragraph 1**: Highlight 2-3 most relevant achievements with specific metrics
- **Body Paragraph 2**: Demonstrate knowledge of company/role and explain mutual fit
- **Closing Paragraph**: Professional call-to-action with confidence and gratitude

**CONTENT STRATEGY**:
- Mirror key terminology from job description naturally
- Showcase quantified achievements that directly relate to role requirements
- Demonstrate research about the company and position
- Convey genuine enthusiasm and cultural fit
- Address any potential concerns proactively
- Use confident, professional tone throughout

**WRITING STANDARDS**:
- Length: 250-300 words maximum
- Tone: Professional, confident, and engaging
- Language: Clear, concise, and error-free
- Format: Clean business letter structure in Markdown
- Flow: Logical progression with smooth transitions

**FORMATTING REQUIREMENTS**:
- Return ONLY the letter body in Markdown format
- Use proper paragraph breaks for readability
- No salutation or signature blocks (body content only)
- Professional formatting with appropriate emphasis

Create a cover letter that compels the hiring manager to immediately schedule an interview."""

        self._initialize_client_with_strategies()

    def _initialize_client_with_strategies(self):
        """Initialize OpenAI client with multiple fallback strategies"""
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            self.fallback_mode = True
            return
        # Ensure no ambient env var overrides the provided key during client use
        try:
            import os
            if os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"] != self.api_key:
                logger.warning("Ambient OPENAI_API_KEY differs from provided key; using provided key explicitly")
        except Exception:
            pass
        # Try to import config for dynamic model/timeout settings
        try:
            from app.core.config import settings  # deferred import to avoid cycles at module import time
            if settings.openai_model_text:
                self.model_text = settings.openai_model_text
            if settings.openai_timeout:
                self.timeout = float(settings.openai_timeout)
            logger.info(
                "AIProcessor config applied",
                model_text=self.model_text,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning("Could not import settings for AIProcessor, using defaults", error=str(e))

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

    def get_processing_metadata(self) -> dict:
        """Get metadata about AI processing including fallback usage"""
        return {
            "fallback_mode_enabled": self.fallback_mode,
            "used_fallback_resume": self.used_fallback_resume,
            "used_fallback_cover_letter": self.used_fallback_cover_letter,
            "openai_model_text": self.model_text,
            "openai_timeout": self.timeout,
            "client_initialized": self.client is not None
        }

    def _strategy_basic(self):
        """Basic OpenAI client initialization"""
        try:
            from openai import OpenAI
            # Explicitly pass api_key param to avoid SDK reading any ambient OPENAI_API_KEY
            return OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise Exception("OpenAI library not available")

    def _strategy_with_timeout(self):
        """OpenAI client with extended timeout"""
        try:
            from openai import OpenAI
            # Add slight buffer to configured timeout and force explicit key usage
            return OpenAI(api_key=self.api_key, timeout=max(self.timeout, 60.0))
        except ImportError:
            raise Exception("OpenAI library not available")

    def _strategy_minimal(self):
        """Minimal OpenAI client configuration"""
        try:
            from openai import OpenAI
            # Force explicit key usage; minimal retries
            return OpenAI(api_key=self.api_key, max_retries=1, timeout=self.timeout)
        except ImportError:
            raise Exception("OpenAI library not available")
    
    async def optimize_resume_text(self, resume_text: str, job_description: str) -> str:
        """
        Simple resume optimization using proven prompts from reference.md
        Returns optimized resume in markdown format
        """
        # Diagnostic logging for fallback detection
        logger.info(
            "Resume optimization starting",
            fallback_mode=self.fallback_mode,
            client_initialized=self.client is not None,
            model_text=self.model_text
        )

        if self.fallback_mode or self.client is None:
            logger.warning("Using fallback optimization (OpenAI not available)")
            self.used_fallback_resume = True
            return self._create_fallback_optimized_resume(resume_text, job_description)

        try:
            return await self._openai_optimize_resume(resume_text, job_description)
        except Exception as e:
            # Classify 401 to make debugging clearer
            msg = str(e)
            auth_error = "401" in msg or "invalid_api_key" in msg or "Incorrect API key provided" in msg
            logger.warning(
                "OpenAI optimization failed, using fallback",
                error=msg,
                is_auth_error=auth_error
            )
            self.used_fallback_resume = True
            return self._create_fallback_optimized_resume(resume_text, job_description)

    async def _openai_optimize_resume(self, resume_text: str, job_description: str) -> str:
        """Perform OpenAI-based resume optimization"""
        prompt = f"Job Description:\n{job_description}\n\nCurrent Résumé:\n{resume_text}"

        start_time = time.time()
        # Use explicit API key; guard against SDK reading ambient env by passing api_key in client construction (already done)
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_text,
            messages=[
                {"role": "system", "content": self.SYS_RESUME},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=3000
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
        # Diagnostic logging for fallback detection
        logger.info(
            "Cover letter generation starting",
            fallback_mode=self.fallback_mode,
            client_initialized=self.client is not None,
            model_text=self.model_text
        )

        if self.fallback_mode or self.client is None:
            logger.warning("Using fallback cover letter (OpenAI not available)")
            self.used_fallback_cover_letter = True
            return self._create_fallback_cover_letter(resume_markdown, job_description)

        try:
            return await self._openai_generate_cover_letter(resume_markdown, job_description, job_title, company_name)
        except Exception as e:
            msg = str(e)
            auth_error = "401" in msg or "invalid_api_key" in msg or "Incorrect API key provided" in msg
            logger.warning(
                "OpenAI cover letter generation failed, using fallback",
                error=msg,
                is_auth_error=auth_error
            )
            self.used_fallback_cover_letter = True
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