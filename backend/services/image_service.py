"""
Image Service for SatQuery AI Backend

This service handles all image-related operations:
1. Image upload and storage
2. Image validation and format checking
3. Temporary file management
4. Image metadata extraction
5. Image preprocessing (resizing, normalization)
6. Image cleanup and deletion

Features:
- Supports multiple formats (GeoTIFF, TIFF, PNG, JPEG)
- Automatic metadata extraction
- Temporary file management with cleanup
- Image validation and compatibility checking
- Asynchronous operations for better performance
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime
import asyncio
import aiofiles
from fastapi import UploadFile

# Import data processing modules
try:
    from data_processing import ImageLoader, FormatChecker, GeospatialUtils
except ImportError:
    # Fallback for testing
    class ImageLoader:
        @staticmethod
        def load_image(file_path):
            return None, {}
        @staticmethod
        def get_image_info(file_path):
            return {'width': 1024, 'height': 1024, 'format': 'GeoTIFF'}
    
    class FormatChecker:
        @staticmethod
        def check_file_format(file_path):
            return {'is_valid': True, 'format_type': 'GeoTIFF', 'metadata': {}}
    
    class GeospatialUtils:
        @staticmethod
        def get_bounds_from_image(file_path):
            return {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}

logger = logging.getLogger(__name__)


# ============================================
# Image Service Class
# ============================================

class ImageService:
    """
    Image Service for handling all image operations
    
    Features:
    1. Upload images with validation
    2. Extract metadata from images
    3. Manage temporary files
    4. Cleanup resources
    5. Support multiple formats
    """
    
    def __init__(
        self,
        temp_dir: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100 MB
        allowed_formats: Optional[List[str]] = None
    ):
        """
        Initialize Image Service
        
        Args:
            temp_dir: Temporary directory for uploaded images
            max_file_size: Maximum file size in bytes
            allowed_formats: List of allowed formats (None = all supported)
        """
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), 'satquery_images')
        self.max_file_size = max_file_size
        self.allowed_formats = allowed_formats or [
            '.tif', '.tiff', '.png', '.jpg', '.jpeg'
        ]
        
        # Create temp directory if it doesn't exist
        self._ensure_temp_dir()
        
        # Track uploaded files for cleanup
        self._uploaded_files: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"✅ ImageService initialized (temp_dir: {self.temp_dir})")
    
    def _ensure_temp_dir(self):
        """Ensure temporary directory exists"""
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # File Upload Methods
    # ============================================
    
    async def upload_image(
        self,
        file: UploadFile,
        save_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Upload and save an image file
        
        Args:
            file: UploadFile object
            save_metadata: Whether to save metadata
        
        Returns:
            Dictionary with file info and metadata
        """
        # Validate file
        validation_result = await self.validate_file(file)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'valid': False
            }
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{file_id}{ext}"
        file_path = os.path.join(self.temp_dir, filename)
        
        try:
            # Save file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_size = len(content)
            
            # Extract metadata
            metadata = {}
            if save_metadata:
                metadata = self.extract_metadata(file_path)
            
            # Store file info
            file_info = {
                'file_id': file_id,
                'filename': file.filename,
                'saved_path': file_path,
                'size': file_size,
                'upload_time': datetime.now().isoformat(),
                'metadata': metadata,
                'format': ext
            }
            
            self._uploaded_files[file_id] = file_info
            
            logger.info(f"✅ File uploaded: {file.filename} ({file_id})")
            
            return {
                'success': True,
                'file_id': file_id,
                'file_path': file_path,
                'filename': file.filename,
                'size': file_size,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def upload_images(
        self,
        files: List[UploadFile],
        save_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Upload multiple images
        
        Args:
            files: List of UploadFile objects
            save_metadata: Whether to save metadata
        
        Returns:
            List of upload results
        """
        results = []
        for file in files:
            result = await self.upload_image(file, save_metadata)
            results.append(result)
        
        return results
    
    async def save_uploaded_images(
        self,
        files: List[UploadFile],
        save_metadata: bool = True
    ) -> List[str]:
        """
        Save uploaded images and return file paths
        
        Args:
            files: List of UploadFile objects
            save_metadata: Whether to save metadata
        
        Returns:
            List of saved file paths
        """
        paths = []
        for file in files:
            result = await self.upload_image(file, save_metadata)
            if result.get('success'):
                paths.append(result['file_path'])
            else:
                logger.warning(f"Failed to upload: {file.filename} - {result.get('error')}")
        
        return paths
    
    # ============================================
    # File Validation Methods
    # ============================================
    
    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validate an uploaded file
        
        Args:
            file: UploadFile object
        
        Returns:
            Validation result dictionary
        """
        # Check file size
        if file.size and file.size > self.max_file_size:
            return {
                'valid': False,
                'error': f"File too large: {file.size} bytes (max: {self.max_file_size} bytes)"
            }
        
        # Check file format
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in self.allowed_formats:
            return {
                'valid': False,
                'error': f"Unsupported format: {ext}. Allowed: {', '.join(self.allowed_formats)}"
            }
        
        # Check filename
        if not file.filename or not file.filename.strip():
            return {
                'valid': False,
                'error': "Empty filename"
            }
        
        return {
            'valid': True,
            'format': ext,
            'filename': file.filename
        }
    
    def validate_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a file path
        
        Args:
            file_path: Path to file
        
        Returns:
            Validation result dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            return {'valid': False, 'error': f"File not found: {file_path}"}
        
        ext = path.suffix.lower()
        if ext not in self.allowed_formats:
            return {
                'valid': False,
                'error': f"Unsupported format: {ext}. Allowed: {', '.join(self.allowed_formats)}"
            }
        
        return {'valid': True, 'file_path': str(path)}
    
    # ============================================
    # Metadata Extraction Methods
    # ============================================
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from image file
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'format': os.path.splitext(file_path)[1].lower(),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get format info
            format_result = FormatChecker.check_file_format(file_path)
            if format_result.get('is_valid'):
                metadata['format_type'] = format_result.get('format_type', 'unknown')
                metadata['metadata'] = format_result.get('metadata', {})
                metadata['geospatial'] = format_result.get('geospatial', {})
                metadata['band_info'] = format_result.get('band_info', {})
            
            # Get image info
            info = ImageLoader.get_image_info(file_path)
            metadata.update(info)
            
            # Get geospatial bounds
            try:
                bounds = GeospatialUtils.get_bounds_from_image(file_path)
                metadata['bounds'] = bounds
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            metadata['error'] = str(e)
        
        return metadata
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an uploaded file
        
        Args:
            file_id: File ID
        
        Returns:
            File information dictionary
        """
        return self._uploaded_files.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get file path by ID
        
        Args:
            file_id: File ID
        
        Returns:
            File path or None
        """
        file_info = self._uploaded_files.get(file_id)
        return file_info.get('saved_path') if file_info else None
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """
        Get all uploaded files
        
        Returns:
            List of file information
        """
        return list(self._uploaded_files.values())
    
    # ============================================
    # File Cleanup Methods
    # ============================================
    
    def cleanup_file(self, file_id: str) -> bool:
        """
        Delete an uploaded file
        
        Args:
            file_id: File ID
        
        Returns:
            True if deleted successfully
        """
        file_info = self._uploaded_files.get(file_id)
        if not file_info:
            return False
        
        file_path = file_info.get('saved_path')
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                del self._uploaded_files[file_id]
                logger.info(f"File deleted: {file_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete file: {e}")
                return False
        
        return False
    
    def cleanup_temp_files(self, file_paths: List[str]) -> int:
        """
        Cleanup temporary files
        
        Args:
            file_paths: List of file paths to delete
        
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned += 1
                    logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
        
        return cleaned
    
    def cleanup_all(self) -> int:
        """
        Cleanup all uploaded files
        
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        file_ids = list(self._uploaded_files.keys())
        for file_id in file_ids:
            if self.cleanup_file(file_id):
                cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} files")
        return cleaned
    
    def cleanup_old_files(self, older_than_days: int = 1) -> int:
        """
        Cleanup files older than specified days
        
        Args:
            older_than_days: Age threshold in days
        
        Returns:
            Number of files cleaned up
        """
        cutoff = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        cleaned = 0
        
        for file_id, file_info in list(self._uploaded_files.items()):
            upload_time = datetime.fromisoformat(file_info['upload_time']).timestamp()
            if upload_time < cutoff:
                if self.cleanup_file(file_id):
                    cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} old files")
        return cleaned
    
    def clear_temp_dir(self) -> int:
        """
        Clear entire temporary directory
        
        Returns:
            Number of files deleted
        """
        count = 0
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        
        self._uploaded_files.clear()
        logger.info(f"Cleared temp directory: {count} files")
        return count
    
    # ============================================
    # Image Processing Methods
    # ============================================
    
    async def load_image(self, file_path: str, normalize: bool = True) -> Dict[str, Any]:
        """
        Load image from file path
        
        Args:
            file_path: Path to image file
            normalize: Whether to normalize image
        
        Returns:
            Dictionary with image data and metadata
        """
        try:
            image, metadata = ImageLoader.load_image(file_path, normalize=normalize)
            return {
                'success': True,
                'image': image,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_image_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get image statistics
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with image statistics
        """
        try:
            from data_processing.image_utils import ImageUtils
            image, _ = ImageLoader.load_image(file_path)
            stats = ImageUtils.get_image_stats(image)
            return {
                'success': True,
                'stats': stats
            }
        except Exception as e:
            logger.error(f"Failed to get image stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_compatible_pairs(
        self,
        file_paths: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Get compatible image pairs (for cross-modal or bi-temporal analysis)
        
        Args:
            file_paths: List of file paths
        
        Returns:
            List of compatible pairs
        """
        compatible_pairs = []
        
        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                # Check if images are compatible
                result = FormatChecker.validate_pair(file_paths[i], file_paths[j])
                if result.get('is_compatible'):
                    compatible_pairs.append((file_paths[i], file_paths[j]))
        
        return compatible_pairs
    
    # ============================================
    # Storage Information
    # ============================================
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information
        
        Returns:
            Dictionary with storage info
        """
        total_files = len(self._uploaded_files)
        total_size = sum(f.get('size', 0) for f in self._uploaded_files.values())
        
        return {
            'temp_dir': self.temp_dir,
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024) if total_size else 0,
            'max_file_size_mb': self.max_file_size / (1024 * 1024)
        }


# ============================================
# Test Functions
# ============================================

async def test_image_service():
    """Test the image service"""
    print("🧪 Testing ImageService...")
    print("=" * 60)
    
    # Create service
    service = ImageService()
    print(f"✅ Service initialized (temp_dir: {service.temp_dir})")
    
    # Test storage info
    info = service.get_storage_info()
    print(f"📊 Storage info: {info}")
    
    # Test cleanup
    cleaned = service.cleanup_all()
    print(f"🧹 Cleaned up {cleaned} files")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_image_service())