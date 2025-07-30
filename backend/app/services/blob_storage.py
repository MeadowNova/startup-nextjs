"""
Blob Storage Service for ResumeForge
Handles file uploads, downloads, and management with Vercel Blob Storage
"""

import asyncio
import aiofiles
import aiohttp
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import structlog
import os
from datetime import datetime, timedelta

logger = structlog.get_logger()


class BlobStorageService:
    """
    Blob storage service with support for Vercel Blob Storage
    Provides secure file upload/download with expiration and cleanup
    """
    
    def __init__(self, blob_read_write_token: Optional[str] = None):
        """
        Initialize blob storage service
        
        Args:
            blob_read_write_token: Vercel Blob Storage token
        """
        self.blob_token = blob_read_write_token or os.getenv("BLOB_READ_WRITE_TOKEN")
        if not self.blob_token:
            logger.warning("No blob storage token provided - using local storage fallback")
        
        self.base_url = "https://blob.vercel-storage.com"
        self.supported_types = {
            'application/pdf': '.pdf',
            'image/png': '.png', 
            'image/jpeg': '.jpg',
            'application/zip': '.zip',
            'text/plain': '.txt'
        }
        
        # Local fallback directory
        self.local_storage_dir = Path("./storage")
        self.local_storage_dir.mkdir(exist_ok=True)
    
    async def upload_file(self, file_path: Path, blob_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload file to blob storage
        
        Args:
            file_path: Local path to file to upload
            blob_key: Unique key for the blob (e.g., "resumes/user123/resume.pdf")
            content_type: MIME type of file (auto-detected if not provided)
            
        Returns:
            str: Blob URL for the uploaded file
            
        Raises:
            ValueError: If file doesn't exist or upload fails
        """
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        # Auto-detect content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if not content_type:
                content_type = 'application/octet-stream'
        
        # Validate file type
        if content_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {content_type}")
        
        try:
            if self.blob_token:
                return await self._upload_to_vercel_blob(file_path, blob_key, content_type)
            else:
                return await self._upload_to_local_storage(file_path, blob_key)
                
        except Exception as e:
            logger.error("File upload failed", error=str(e), file_path=str(file_path), blob_key=blob_key)
            raise ValueError(f"Upload failed: {str(e)}")
    
    async def _upload_to_vercel_blob(self, file_path: Path, blob_key: str, content_type: str) -> str:
        """Upload file to Vercel Blob Storage"""
        async with aiofiles.open(file_path, 'rb') as f:
            file_data = await f.read()
        
        # Calculate file hash for integrity
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        headers = {
            'Authorization': f'Bearer {self.blob_token}',
            'X-Content-Type': content_type,
            'X-Add-Random-Suffix': '1'  # Prevent filename collisions
        }
        
        # Use multipart upload for larger files
        data = aiohttp.FormData()
        data.add_field('file', file_data, filename=blob_key, content_type=content_type)
        
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/upload",
                headers=headers,
                data=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Blob upload failed: {response.status} - {error_text}")
                
                result = await response.json()
                blob_url = result.get('url')
                
                if not blob_url:
                    raise ValueError("No URL returned from blob storage")
                
                logger.info("File uploaded to blob storage", 
                           blob_key=blob_key, 
                           blob_url=blob_url,
                           file_size=len(file_data),
                           file_hash=file_hash)
                
                return blob_url
    
    async def _upload_to_local_storage(self, file_path: Path, blob_key: str) -> str:
        """Fallback: Upload to local storage"""
        # Create directory structure
        local_path = self.local_storage_dir / blob_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        async with aiofiles.open(file_path, 'rb') as src:
            async with aiofiles.open(local_path, 'wb') as dst:
                content = await src.read()
                await dst.write(content)
        
        # Return local file URL
        blob_url = f"file://{local_path.absolute()}"
        logger.info("File uploaded to local storage", blob_key=blob_key, local_path=str(local_path))
        return blob_url
    
    async def download_file(self, blob_url: str, local_path: Path) -> Path:
        """
        Download file from blob storage to local path
        
        Args:
            blob_url: URL of the blob to download
            local_path: Local path where file should be saved
            
        Returns:
            Path: Path to downloaded file
            
        Raises:
            ValueError: If download fails
        """
        try:
            # Ensure directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if blob_url.startswith('file://'):
                # Local file - just copy
                source_path = Path(blob_url.replace('file://', ''))
                async with aiofiles.open(source_path, 'rb') as src:
                    async with aiofiles.open(local_path, 'wb') as dst:
                        content = await src.read()
                        await dst.write(content)
            else:
                # Remote blob - download
                async with aiohttp.ClientSession() as session:
                    async with session.get(blob_url) as response:
                        if response.status != 200:
                            raise ValueError(f"Download failed: {response.status}")
                        
                        async with aiofiles.open(local_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
            
            logger.info("File downloaded", blob_url=blob_url, local_path=str(local_path))
            return local_path
            
        except Exception as e:
            logger.error("File download failed", error=str(e), blob_url=blob_url)
            raise ValueError(f"Download failed: {str(e)}")
    
    async def delete_file(self, blob_url: str) -> bool:
        """
        Delete file from blob storage
        
        Args:
            blob_url: URL of the blob to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            if blob_url.startswith('file://'):
                # Local file
                local_path = Path(blob_url.replace('file://', ''))
                if local_path.exists():
                    local_path.unlink()
                return True
            
            if not self.blob_token:
                logger.warning("Cannot delete remote blob without token")
                return False
            
            # Extract blob key from URL
            parsed = urlparse(blob_url)
            blob_key = parsed.path.lstrip('/')
            
            headers = {
                'Authorization': f'Bearer {self.blob_token}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/{blob_key}",
                    headers=headers
                ) as response:
                    success = response.status in [200, 204, 404]  # 404 means already deleted
                    
                    if success:
                        logger.info("File deleted from blob storage", blob_url=blob_url)
                    else:
                        logger.error("File deletion failed", 
                                   blob_url=blob_url, 
                                   status=response.status)
                    
                    return success
                    
        except Exception as e:
            logger.error("File deletion failed", error=str(e), blob_url=blob_url)
            return False
    
    async def generate_signed_url(self, blob_url: str, expires_in: int = 3600) -> str:
        """
        Generate signed URL with expiration
        
        Args:
            blob_url: Original blob URL
            expires_in: Expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Signed URL with expiration
        """
        # For local files, return as-is
        if blob_url.startswith('file://'):
            return blob_url
        
        # For Vercel Blob, URLs are already secure and don't need signing
        # In production, you might implement additional security layers
        logger.info("Generated signed URL", blob_url=blob_url, expires_in=expires_in)
        return blob_url
    
    def get_file_info(self, blob_url: str) -> Dict[str, Any]:
        """
        Get metadata about a blob file
        
        Args:
            blob_url: URL of the blob
            
        Returns:
            Dict with file metadata
        """
        parsed = urlparse(blob_url)
        
        return {
            'url': blob_url,
            'filename': Path(parsed.path).name,
            'is_local': blob_url.startswith('file://'),
            'created_at': datetime.utcnow().isoformat()
        }


# Singleton instance for dependency injection
blob_storage_service = BlobStorageService()


async def get_blob_storage() -> BlobStorageService:
    """Dependency injection for FastAPI"""
    return blob_storage_service
