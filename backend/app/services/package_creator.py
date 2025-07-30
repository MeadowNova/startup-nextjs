"""
Package Creator Service for ResumeForge
Creates ZIP packages with optimized resume and cover letter
Handles file naming, organization, and blob storage upload
"""

import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import structlog

from .blob_storage import BlobStorageService

logger = structlog.get_logger()


class PackageCreator:
    """
    Service for creating downloadable packages with resume and cover letter
    Handles ZIP creation, file organization, and upload to blob storage
    """
    
    def __init__(self, blob_storage: BlobStorageService):
        """
        Initialize package creator
        
        Args:
            blob_storage: Blob storage service for uploads
        """
        self.blob_storage = blob_storage
        self.temp_dir = Path(tempfile.gettempdir()) / "resumeforge_packages"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def create_job_package(
        self,
        resume_pdf_path: Path,
        cover_letter_pdf_path: Path,
        user_id: int,
        job_id: int,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Create a complete job application package
        
        Args:
            resume_pdf_path: Path to optimized resume PDF
            cover_letter_pdf_path: Path to cover letter PDF
            user_id: User ID for file organization
            job_id: Job ID for tracking
            job_title: Optional job title for naming
            company_name: Optional company name for naming
            
        Returns:
            Tuple of (package_blob_url, package_metadata)
            
        Raises:
            ValueError: If package creation fails
        """
        try:
            # Generate package filename
            package_filename = self._generate_package_filename(
                job_title, company_name, job_id
            )
            
            # Create ZIP package
            package_path = await self._create_zip_package(
                resume_pdf_path=resume_pdf_path,
                cover_letter_pdf_path=cover_letter_pdf_path,
                package_filename=package_filename,
                job_title=job_title,
                company_name=company_name
            )
            
            # Upload to blob storage
            blob_key = f"packages/user_{user_id}/job_{job_id}/{package_filename}"
            package_blob_url = await self.blob_storage.upload_file(
                file_path=package_path,
                blob_key=blob_key,
                content_type="application/zip"
            )
            
            # Generate package metadata
            package_metadata = self._generate_package_metadata(
                package_path=package_path,
                package_blob_url=package_blob_url,
                job_title=job_title,
                company_name=company_name,
                job_id=job_id
            )
            
            # Cleanup temporary file
            package_path.unlink(missing_ok=True)
            
            logger.info("Job package created successfully",
                       user_id=user_id,
                       job_id=job_id,
                       package_url=package_blob_url,
                       package_size=package_metadata.get('size_bytes'))
            
            return package_blob_url, package_metadata
            
        except Exception as e:
            logger.error("Package creation failed",
                        error=str(e),
                        user_id=user_id,
                        job_id=job_id)
            raise ValueError(f"Package creation failed: {str(e)}")
    
    async def create_simple_package(
        self,
        resume_pdf_path: Path,
        cover_letter_pdf_path: Path,
        package_name: Optional[str] = None
    ) -> Path:
        """
        Create a simple ZIP package without blob storage upload
        
        Args:
            resume_pdf_path: Path to resume PDF
            cover_letter_pdf_path: Path to cover letter PDF
            package_name: Optional custom package name
            
        Returns:
            Path: Local path to created ZIP package
        """
        try:
            if not package_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                package_name = f"resume_package_{timestamp}.zip"
            
            package_path = self.temp_dir / package_name
            
            # Create ZIP file
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add resume
                if resume_pdf_path.exists():
                    zipf.write(resume_pdf_path, "optimized_resume.pdf")
                
                # Add cover letter
                if cover_letter_pdf_path.exists():
                    zipf.write(cover_letter_pdf_path, "cover_letter.pdf")
                
                # Add README
                readme_content = self._generate_readme_content()
                zipf.writestr("README.txt", readme_content)
            
            logger.info("Simple package created", package_path=str(package_path))
            return package_path
            
        except Exception as e:
            logger.error("Simple package creation failed", error=str(e))
            raise ValueError(f"Package creation failed: {str(e)}")
    
    def _generate_package_filename(
        self,
        job_title: Optional[str],
        company_name: Optional[str],
        job_id: int
    ) -> str:
        """Generate a descriptive filename for the package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean and format job title and company name
        clean_job_title = self._clean_filename_part(job_title) if job_title else None
        clean_company = self._clean_filename_part(company_name) if company_name else None
        
        # Build filename parts
        parts = []
        
        if clean_company:
            parts.append(clean_company)
        
        if clean_job_title:
            parts.append(clean_job_title)
        
        if not parts:
            parts.append(f"job_{job_id}")
        
        parts.append(timestamp)
        
        filename = "_".join(parts) + ".zip"
        
        # Ensure filename isn't too long
        if len(filename) > 100:
            filename = f"job_{job_id}_{timestamp}.zip"
        
        return filename
    
    def _clean_filename_part(self, text: str) -> str:
        """Clean text for use in filename"""
        if not text:
            return ""
        
        # Remove special characters and limit length
        import re
        cleaned = re.sub(r'[^\w\s-]', '', text)
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        cleaned = cleaned.strip('_')
        
        # Limit length
        if len(cleaned) > 30:
            cleaned = cleaned[:30]
        
        return cleaned.lower()
    
    async def _create_zip_package(
        self,
        resume_pdf_path: Path,
        cover_letter_pdf_path: Path,
        package_filename: str,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> Path:
        """Create ZIP package with organized file structure"""
        package_path = self.temp_dir / package_filename
        
        # Determine file names within ZIP
        resume_filename = self._generate_resume_filename(job_title, company_name)
        cover_filename = self._generate_cover_filename(job_title, company_name)
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add resume PDF
            if resume_pdf_path.exists():
                zipf.write(resume_pdf_path, resume_filename)
            else:
                logger.warning("Resume PDF not found", path=str(resume_pdf_path))
            
            # Add cover letter PDF
            if cover_letter_pdf_path.exists():
                zipf.write(cover_letter_pdf_path, cover_filename)
            else:
                logger.warning("Cover letter PDF not found", path=str(cover_letter_pdf_path))
            
            # Add application instructions
            instructions = self._generate_application_instructions(job_title, company_name)
            zipf.writestr("APPLICATION_INSTRUCTIONS.txt", instructions)
            
            # Add metadata file
            metadata = self._generate_zip_metadata(job_title, company_name)
            zipf.writestr("package_info.json", metadata)
        
        return package_path
    
    def _generate_resume_filename(self, job_title: Optional[str], company_name: Optional[str]) -> str:
        """Generate descriptive filename for resume within ZIP"""
        parts = ["resume"]
        
        if company_name:
            clean_company = self._clean_filename_part(company_name)
            if clean_company:
                parts.append(clean_company)
        
        if job_title:
            clean_title = self._clean_filename_part(job_title)
            if clean_title:
                parts.append(clean_title)
        
        return "_".join(parts) + "_optimized.pdf"
    
    def _generate_cover_filename(self, job_title: Optional[str], company_name: Optional[str]) -> str:
        """Generate descriptive filename for cover letter within ZIP"""
        parts = ["cover_letter"]
        
        if company_name:
            clean_company = self._clean_filename_part(company_name)
            if clean_company:
                parts.append(clean_company)
        
        return "_".join(parts) + ".pdf"
    
    def _generate_readme_content(self) -> str:
        """Generate README content for the package"""
        return """ResumeForge - ATS-Optimized Application Package
=============================================

This package contains your ATS-optimized resume and matching cover letter.

Files included:
- optimized_resume.pdf: Your resume optimized for Applicant Tracking Systems
- cover_letter.pdf: Matching cover letter tailored to the job description

Tips for application:
1. Use the optimized resume for online applications
2. Customize the cover letter with specific details if needed
3. Follow up within 1-2 weeks of application

Generated by ResumeForge - AI-Powered Resume Optimization
"""
    
    def _generate_application_instructions(self, job_title: Optional[str], company_name: Optional[str]) -> str:
        """Generate application-specific instructions"""
        instructions = "Application Package Instructions\n"
        instructions += "=" * 35 + "\n\n"
        
        if company_name and job_title:
            instructions += f"Position: {job_title}\n"
            instructions += f"Company: {company_name}\n\n"
        
        instructions += """Files in this package:
1. Resume (ATS-optimized for this specific job)
2. Cover letter (tailored to match the job requirements)

Application checklist:
□ Review both documents for accuracy
□ Upload resume to company's application system
□ Include cover letter in email or upload separately
□ Follow any specific application instructions from the job posting
□ Keep a copy of this application for your records

Best practices:
- Apply within 24-48 hours of job posting when possible
- Follow up with a professional email after 1-2 weeks
- Customize the cover letter further if you have specific insights about the company

Generated by ResumeForge
"""
        return instructions
    
    def _generate_zip_metadata(self, job_title: Optional[str], company_name: Optional[str]) -> str:
        """Generate JSON metadata for the package"""
        import json
        
        metadata = {
            "package_type": "job_application",
            "created_at": datetime.utcnow().isoformat(),
            "job_title": job_title,
            "company_name": company_name,
            "files": [
                {
                    "name": self._generate_resume_filename(job_title, company_name),
                    "type": "resume",
                    "description": "ATS-optimized resume"
                },
                {
                    "name": self._generate_cover_filename(job_title, company_name),
                    "type": "cover_letter", 
                    "description": "Tailored cover letter"
                }
            ],
            "generator": "ResumeForge",
            "version": "1.0"
        }
        
        return json.dumps(metadata, indent=2)
    
    def _generate_package_metadata(
        self,
        package_path: Path,
        package_blob_url: str,
        job_title: Optional[str],
        company_name: Optional[str],
        job_id: int
    ) -> Dict:
        """Generate metadata about the created package"""
        file_size = package_path.stat().st_size if package_path.exists() else 0
        
        return {
            "package_url": package_blob_url,
            "filename": package_path.name,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "job_id": job_id,
            "job_title": job_title,
            "company_name": company_name,
            "created_at": datetime.utcnow().isoformat(),
            "file_count": 2,  # Resume + Cover letter
            "package_type": "job_application"
        }


# Factory function for dependency injection
async def get_package_creator(blob_storage: BlobStorageService) -> PackageCreator:
    """Create PackageCreator instance with blob storage dependency"""
    return PackageCreator(blob_storage)
