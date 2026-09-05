"""
Image Loader Module for SatQuery AI

This module handles loading of various image formats:
1. GeoTIFF (.tif, .tiff) - with geospatial metadata
2. TIFF (.tif, .tiff)
3. PNG (.png)
4. JPEG (.jpg, .jpeg)

Features:
- Load images with metadata
- Extract geospatial information
- Band selection
- Normalization
- Batch loading from directories
- Format detection
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try importing rasterio for GeoTIFF support
try:
    import rasterio
    from rasterio.io import DatasetReader
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

class ImageFormat:
    """Image format constants"""
    GEOTIFF = "GeoTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    UNKNOWN = "unknown"


class ImageMode:
    """Image mode constants"""
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    MULTIBAND = "multiband"


# ============================================
# Image Loader Class
# ============================================

class ImageLoader:
    """
    Loads and validates satellite imagery in various formats
    
    Supported formats:
    - GeoTIFF (.tif, .tiff) - with geospatial metadata
    - TIFF (.tif, .tiff)
    - PNG (.png)
    - JPEG (.jpg, .jpeg)
    """
    
    SUPPORTED_FORMATS = {
        '.tif': ImageFormat.GEOTIFF,
        '.tiff': ImageFormat.GEOTIFF,
        '.png': ImageFormat.PNG,
        '.jpg': ImageFormat.JPEG,
        '.jpeg': ImageFormat.JPEG
    }
    
    @staticmethod
    def load_image(
        file_path: Union[str, Path],
        bands: Optional[List[int]] = None,
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load an image from file
        
        Args:
            file_path: Path to image file
            bands: List of band indices to load (0-indexed, None = all)
            normalize: Whether to normalize to 0-1 range
            dtype: Output data type (default: float32)
        
        Returns:
            Tuple of (image_array, metadata)
        
        Raises:
            FileNotFoundError: If file not found
            ValueError: If unsupported format
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in ImageLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {list(ImageLoader.SUPPORTED_FORMATS.keys())}")
        
        format_type = ImageLoader.SUPPORTED_FORMATS[ext]
        
        # Load based on format
        if format_type in [ImageFormat.GEOTIFF, ImageFormat.TIFF]:
            return ImageLoader._load_geotiff(file_path, bands, normalize, dtype)
        else:
            return ImageLoader._load_rgb_image(file_path, normalize, dtype)
    
    @staticmethod
    def _load_geotiff(
        file_path: Path,
        bands: Optional[List[int]] = None,
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load GeoTIFF image with rasterio"""
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available. Falling back to PIL for TIFF.")
            return ImageLoader._load_tiff_with_pil(file_path, normalize, dtype)
        
        try:
            with rasterio.open(file_path) as src:
                # Get metadata
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs) if src.crs else None,
                    'crs_epsg': src.crs.to_epsg() if src.crs and hasattr(src.crs, 'to_epsg') else None,
                    'transform': src.transform,
                    'bounds': {
                        'left': src.bounds.left,
                        'bottom': src.bounds.bottom,
                        'right': src.bounds.right,
                        'top': src.bounds.top
                    },
                    'dtype': str(src.dtypes[0]),
                    'format': 'GeoTIFF',
                    'file_path': str(file_path),
                    'nodata': src.nodata,
                    'driver': src.driver,
                    'compression': src.compression if hasattr(src, 'compression') else None
                }
                
                # Load bands
                if bands is None:
                    bands = list(range(1, src.count + 1))
                else:
                    # Convert to 1-indexed for rasterio
                    bands = [b + 1 for b in bands if 0 <= b < src.count]
                    if not bands:
                        bands = list(range(1, src.count + 1))
                
                # Read bands
                data = src.read(bands)
                
                # Handle data shape
                if data.shape[0] == 1:
                    image = data[0]
                else:
                    image = np.moveaxis(data, 0, -1)
                
                # Normalize if needed
                if normalize:
                    image = ImageLoader._normalize_array(image)
                
                # Convert dtype
                if dtype is not None:
                    image = image.astype(dtype)
                
                return image, metadata
                
        except Exception as e:
            logger.error(f"Error loading GeoTIFF {file_path}: {e}")
            # Fallback to PIL
            return ImageLoader._load_tiff_with_pil(file_path, normalize, dtype)
    
    @staticmethod
    def _load_tiff_with_pil(
        file_path: Path,
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load TIFF using PIL (fallback)"""
        try:
            img = Image.open(file_path)
            
            # Get metadata
            metadata = {
                'width': img.width,
                'height': img.height,
                'bands': len(img.getbands()),
                'mode': img.mode,
                'format': img.format,
                'file_path': str(file_path),
                'is_georeferenced': False
            }
            
            # Convert to numpy array
            image = np.array(img)
            
            # Handle different modes
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
            
            # Normalize if needed
            if normalize:
                image = ImageLoader._normalize_array(image)
            
            # Convert dtype
            if dtype is not None:
                image = image.astype(dtype)
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error loading TIFF {file_path}: {e}")
            raise
    
    @staticmethod
    def _load_rgb_image(
        file_path: Path,
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load RGB image with PIL"""
        try:
            img = Image.open(file_path)
            
            # Get metadata
            metadata = {
                'width': img.width,
                'height': img.height,
                'bands': len(img.getbands()),
                'mode': img.mode,
                'format': img.format,
                'file_path': str(file_path)
            }
            
            # Convert to numpy array
            image = np.array(img)
            
            # Handle different modes
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
            
            # Normalize if needed
            if normalize:
                image = ImageLoader._normalize_array(image)
            
            # Convert dtype
            if dtype is not None:
                image = image.astype(dtype)
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Error loading RGB image {file_path}: {e}")
            raise
    
    @staticmethod
    def _normalize_array(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range
        
        Args:
            image: Input image array
        
        Returns:
            Normalized image
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image, dtype=np.float32)
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def load_image_from_bytes(
        data: bytes,
        filename: str,
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image from bytes data
        
        Args:
            data: Image data as bytes
            filename: Original filename (for format detection)
            normalize: Whether to normalize
            dtype: Output data type
        
        Returns:
            Tuple of (image_array, metadata)
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        try:
            return ImageLoader.load_image(tmp_path, normalize=normalize, dtype=dtype)
        finally:
            os.unlink(tmp_path)
    
    @staticmethod
    def load_images_from_directory(
        directory: Union[str, Path],
        pattern: str = "*.[tT][iI][fF]",
        normalize: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all images from a directory
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            normalize: Whether to normalize
            dtype: Output data type
        
        Returns:
            List of dictionaries with image, metadata, and file_path
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = []
        for file_path in directory.glob(pattern):
            try:
                image, metadata = ImageLoader.load_image(file_path, normalize=normalize, dtype=dtype)
                results.append({
                    'image': image,
                    'metadata': metadata,
                    'file_path': str(file_path),
                    'file_name': file_path.name
                })
                logger.debug(f"Loaded: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        return results
    
    @staticmethod
    def get_image_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get image metadata without loading full image
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with image metadata
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if not file_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        if ext in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
            return ImageLoader._get_geotiff_info(file_path)
        else:
            return ImageLoader._get_pil_info(file_path)
    
    @staticmethod
    def _get_geotiff_info(file_path: Path) -> Dict[str, Any]:
        """Get GeoTIFF metadata"""
        try:
            with rasterio.open(file_path) as src:
                return {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs) if src.crs else None,
                    'crs_epsg': src.crs.to_epsg() if src.crs and hasattr(src.crs, 'to_epsg') else None,
                    'bounds': {
                        'left': src.bounds.left,
                        'bottom': src.bounds.bottom,
                        'right': src.bounds.right,
                        'top': src.bounds.top
                    },
                    'dtype': str(src.dtypes[0]),
                    'format': 'GeoTIFF',
                    'file_size': file_path.stat().st_size,
                    'file_name': file_path.name,
                    'nodata': src.nodata,
                    'is_georeferenced': src.crs is not None
                }
        except Exception as e:
            return {'error': str(e), 'file_path': str(file_path)}
    
    @staticmethod
    def _get_pil_info(file_path: Path) -> Dict[str, Any]:
        """Get PIL image metadata"""
        try:
            with Image.open(file_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'bands': len(img.getbands()),
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': file_path.stat().st_size,
                    'file_name': file_path.name,
                    'is_georeferenced': False
                }
        except Exception as e:
            return {'error': str(e), 'file_path': str(file_path)}
    
    @staticmethod
    def validate_image(file_path: Union[str, Path]) -> bool:
        """
        Validate if file is a valid image
        
        Args:
            file_path: Path to image file
        
        Returns:
            True if valid, False otherwise
        """
        try:
            info = ImageLoader.get_image_info(file_path)
            return 'error' not in info
        except:
            return False
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported formats"""
        return list(ImageLoader.SUPPORTED_FORMATS.keys())
    
    @staticmethod
    def is_geotiff(file_path: Union[str, Path]) -> bool:
        """Check if file is a GeoTIFF"""
        ext = Path(file_path).suffix.lower()
        if ext not in ['.tif', '.tiff']:
            return False
        
        if not RASTERIO_AVAILABLE:
            return False
        
        try:
            info = ImageLoader.get_image_info(file_path)
            return info.get('is_georeferenced', False)
        except:
            return False


# ============================================
# Test Function
# ============================================

def test_image_loader():
    """Test the image loader module"""
    print("🧪 Testing ImageLoader...")
    print("=" * 60)
    
    # Create a dummy image for testing
    import tempfile
    from PIL import Image
    
    # Create a sample image
    sample_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(sample_img)
    
    # Save as temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Test loading
        print("\n📋 Testing load_image:")
        image, metadata = ImageLoader.load_image(tmp_path)
        print(f"   Image shape: {image.shape}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Metadata: {metadata}")
        
        # Test get_image_info
        print("\n📋 Testing get_image_info:")
        info = ImageLoader.get_image_info(tmp_path)
        print(f"   Info: {info}")
        
        # Test validate_image
        print("\n📋 Testing validate_image:")
        is_valid = ImageLoader.validate_image(tmp_path)
        print(f"   Is valid: {is_valid}")
        
        # Test supported formats
        print("\n📋 Testing get_supported_formats:")
        formats = ImageLoader.get_supported_formats()
        print(f"   Supported formats: {formats}")
        
    finally:
        os.unlink(tmp_path)
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_image_loader()