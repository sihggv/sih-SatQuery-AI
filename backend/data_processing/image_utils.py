"""
Image Utilities Module for SatQuery AI

This module provides comprehensive image processing utilities:
1. Image Resizing - Multiple interpolation methods
2. Image Normalization - Min-max, Z-score, Percentile
3. Color Space Conversions - RGB, BGR, HSV, LAB, YUV, Grayscale
4. Image Enhancement - Contrast, Brightness, Sharpening, Blur
5. CRS Alignment - Reprojection and geospatial alignment
6. Padding and Cropping
7. Image Statistics
8. Batch Processing

Features:
- Multiple interpolation methods
- Various normalization techniques
- Color space conversions
- Image enhancement
- Geospatial alignment
- Batch processing support
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import warnings
warnings.filterwarnings('ignore')

# Try importing rasterio for geospatial operations
try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

class InterpolationMethod:
    """Interpolation methods for resizing"""
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'
    LANCZOS = 'lanczos'
    AREA = 'area'


class NormalizationMethod:
    """Normalization methods"""
    MIN_MAX = 'min_max'
    Z_SCORE = 'z_score'
    PERCENTILE = 'percentile'
    STANDARD = 'standard'


class ColorSpace:
    """Color spaces for conversion"""
    RGB = 'rgb'
    BGR = 'bgr'
    HSV = 'hsv'
    LAB = 'lab'
    YUV = 'yuv'
    GRAY = 'gray'


# ============================================
# Image Utils Class
# ============================================

class ImageUtils:
    """
    Comprehensive image processing utilities
    """
    
    # ============================================
    # Resizing Methods
    # ============================================
    
    @staticmethod
    def resize(
        image: np.ndarray,
        target_size: Tuple[int, int],
        method: str = InterpolationMethod.BILINEAR,
        preserve_aspect: bool = False
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image (H, W) or (H, W, C)
            target_size: (height, width)
            method: Interpolation method
            preserve_aspect: Whether to preserve aspect ratio
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        if preserve_aspect:
            # Calculate new size preserving aspect ratio
            scale = min(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h, new_w = target_h, target_w
        
        # Map to OpenCV/PIL interpolation method
        interp_map = {
            InterpolationMethod.NEAREST: cv2.INTER_NEAREST,
            InterpolationMethod.BILINEAR: cv2.INTER_LINEAR,
            InterpolationMethod.BICUBIC: cv2.INTER_CUBIC,
            InterpolationMethod.LANCZOS: cv2.INTER_LANCZOS4,
            InterpolationMethod.AREA: cv2.INTER_AREA
        }
        
        interp = interp_map.get(method, cv2.INTER_LINEAR)
        
        # Resize
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        
        # Pad if aspect ratio was preserved and size doesn't match
        if preserve_aspect and (new_h != target_h or new_w != target_w):
            # Calculate padding
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            if len(image.shape) == 3:
                resized = np.pad(resized, 
                                ((pad_h, target_h - new_h - pad_h),
                                 (pad_w, target_w - new_w - pad_w),
                                 (0, 0)), 
                                mode='constant')
            else:
                resized = np.pad(resized,
                                ((pad_h, target_h - new_h - pad_h),
                                 (pad_w, target_w - new_w - pad_w)),
                                mode='constant')
        
        return resized
    
    @staticmethod
    def resize_by_scale(
        image: np.ndarray,
        scale: float,
        method: str = InterpolationMethod.BILINEAR
    ) -> np.ndarray:
        """
        Resize image by a scale factor
        
        Args:
            image: Input image
            scale: Scale factor (>1 for upscale, <1 for downscale)
            method: Interpolation method
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        return ImageUtils.resize(image, (new_h, new_w), method)
    
    @staticmethod
    def resize_to_fit(
        image: np.ndarray,
        max_size: int = 1024,
        method: str = InterpolationMethod.BILINEAR
    ) -> np.ndarray:
        """
        Resize image to fit within max_size while preserving aspect ratio
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            method: Interpolation method
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        return ImageUtils.resize_by_scale(image, scale, method)
    
    # ============================================
    # Normalization Methods
    # ============================================
    
    @staticmethod
    def normalize(
        image: np.ndarray,
        method: str = NormalizationMethod.MIN_MAX,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize image using specified method
        
        Args:
            image: Input image
            method: Normalization method
            min_val: Min value for min-max normalization
            max_val: Max value for min-max normalization
        
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == NormalizationMethod.MIN_MAX:
            return ImageUtils._normalize_min_max(image, min_val, max_val)
        elif method == NormalizationMethod.Z_SCORE:
            return ImageUtils._normalize_z_score(image)
        elif method == NormalizationMethod.PERCENTILE:
            return ImageUtils._normalize_percentile(image)
        elif method == NormalizationMethod.STANDARD:
            return ImageUtils._normalize_standard(image)
        else:
            logger.warning(f"Unknown normalization method: {method}. Using min-max.")
            return ImageUtils._normalize_min_max(image, min_val, max_val)
    
    @staticmethod
    def _normalize_min_max(
        image: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        if min_val is None:
            min_val = np.min(image)
        if max_val is None:
            max_val = np.max(image)
        
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        
        return np.clip(normalized, 0, 1)
    
    @staticmethod
    def _normalize_z_score(image: np.ndarray) -> np.ndarray:
        """Z-score normalization (standardization)"""
        mean = np.mean(image)
        std = np.std(image)
        
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    @staticmethod
    def _normalize_percentile(
        image: np.ndarray,
        lower_percentile: float = 2,
        upper_percentile: float = 98
    ) -> np.ndarray:
        """Percentile-based normalization"""
        lower = np.percentile(image, lower_percentile)
        upper = np.percentile(image, upper_percentile)
        
        if upper - lower > 0:
            normalized = (image - lower) / (upper - lower)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    @staticmethod
    def _normalize_standard(image: np.ndarray) -> np.ndarray:
        """Standard normalization to [0, 1]"""
        return ImageUtils._normalize_min_max(image)
    
    @staticmethod
    def denormalize(
        image: np.ndarray,
        original_min: float,
        original_max: float
    ) -> np.ndarray:
        """
        Denormalize image back to original range
        
        Args:
            image: Normalized image [0, 1]
            original_min: Original minimum value
            original_max: Original maximum value
        
        Returns:
            Denormalized image
        """
        return image * (original_max - original_min) + original_min
    
    # ============================================
    # Color Space Conversions
    # ============================================
    
    @staticmethod
    def convert_color_space(
        image: np.ndarray,
        from_space: str,
        to_space: str
    ) -> np.ndarray:
        """
        Convert image between color spaces
        
        Args:
            image: Input image
            from_space: Source color space
            to_space: Target color space
        
        Returns:
            Converted image
        """
        # Normalize to uint8 for OpenCV conversion
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Conversion maps
        conversions = {
            ('rgb', 'bgr'): cv2.COLOR_RGB2BGR,
            ('bgr', 'rgb'): cv2.COLOR_BGR2RGB,
            ('rgb', 'hsv'): cv2.COLOR_RGB2HSV,
            ('hsv', 'rgb'): cv2.COLOR_HSV2RGB,
            ('rgb', 'lab'): cv2.COLOR_RGB2LAB,
            ('lab', 'rgb'): cv2.COLOR_LAB2RGB,
            ('rgb', 'yuv'): cv2.COLOR_RGB2YUV,
            ('yuv', 'rgb'): cv2.COLOR_YUV2RGB,
            ('rgb', 'gray'): cv2.COLOR_RGB2GRAY,
            ('bgr', 'gray'): cv2.COLOR_BGR2GRAY
        }
        
        key = (from_space.lower(), to_space.lower())
        if key in conversions:
            converted = cv2.cvtColor(image_uint8, conversions[key])
            return converted.astype(np.float32) / 255.0 if image.dtype != np.uint8 else converted
        else:
            raise ValueError(f"Unsupported color space conversion: {from_space} -> {to_space}")
    
    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB color space
        
        Args:
            image: Input image
        
        Returns:
            RGB image
        """
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            return image[:, :, :3]
        if image.shape[-1] >= 3:
            return image[:, :, :3]
        return image
    
    # ============================================
    # Image Enhancement
    # ============================================
    
    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        factor: float = 1.5
    ) -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input image
            factor: Contrast enhancement factor
        
        Returns:
            Contrast-enhanced image
        """
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        img_pil = Image.fromarray(image_uint8)
        enhancer = ImageEnhance.Contrast(img_pil)
        enhanced = enhancer.enhance(factor)
        
        enhanced_array = np.array(enhanced)
        return enhanced_array.astype(np.float32) / 255.0 if image.dtype != np.uint8 else enhanced_array
    
    @staticmethod
    def enhance_brightness(
        image: np.ndarray,
        factor: float = 1.2
    ) -> np.ndarray:
        """
        Enhance image brightness
        
        Args:
            image: Input image
            factor: Brightness enhancement factor
        
        Returns:
            Brightness-enhanced image
        """
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        img_pil = Image.fromarray(image_uint8)
        enhancer = ImageEnhance.Brightness(img_pil)
        enhanced = enhancer.enhance(factor)
        
        enhanced_array = np.array(enhanced)
        return enhanced_array.astype(np.float32) / 255.0 if image.dtype != np.uint8 else enhanced_array
    
    @staticmethod
    def apply_sharpening(
        image: np.ndarray,
        factor: float = 2.0
    ) -> np.ndarray:
        """
        Apply sharpening to image
        
        Args:
            image: Input image
            factor: Sharpening factor
        
        Returns:
            Sharpened image
        """
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ]) * factor / 8
        
        sharpened = cv2.filter2D(image_uint8, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255)
        
        return sharpened.astype(np.float32) / 255.0 if image.dtype != np.uint8 else sharpened
    
    @staticmethod
    def apply_gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply Gaussian blur to image
        
        Args:
            image: Input image
            kernel_size: Kernel size (odd number)
        
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        blurred = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
        
        return blurred.astype(np.float32) / 255.0 if image.dtype != np.uint8 else blurred
    
    # ============================================
    # Padding and Cropping
    # ============================================
    
    @staticmethod
    def pad_image(
        image: np.ndarray,
        padding: Union[int, Tuple[int, int, int, int]],
        mode: str = 'constant',
        constant_values: float = 0
    ) -> np.ndarray:
        """
        Pad image
        
        Args:
            image: Input image
            padding: Padding amount (top, bottom, left, right) or single int
            mode: Padding mode ('constant', 'reflect', 'edge')
            constant_values: Value for constant padding
        
        Returns:
            Padded image
        """
        if isinstance(padding, int):
            pad_width = ((padding, padding), (padding, padding))
            if len(image.shape) == 3:
                pad_width = pad_width + ((0, 0),)
        else:
            top, bottom, left, right = padding
            pad_width = ((top, bottom), (left, right))
            if len(image.shape) == 3:
                pad_width = pad_width + ((0, 0),)
        
        return np.pad(image, pad_width, mode=mode, constant_values=constant_values)
    
    @staticmethod
    def crop_to_bbox(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop image to bounding box
        
        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) pixel coordinates
        
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if len(image.shape) == 3:
            return image[y1:y2, x1:x2, :]
        else:
            return image[y1:y2, x1:x2]
    
    # ============================================
    # Image Statistics
    # ============================================
    
    @staticmethod
    def get_image_stats(
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get comprehensive image statistics
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'median': float(np.median(image)),
            'percentile_25': float(np.percentile(image, 25)),
            'percentile_75': float(np.percentile(image, 75))
        }
        
        if len(image.shape) == 3:
            stats['channels'] = image.shape[-1]
            stats['channel_stats'] = []
            for c in range(image.shape[-1]):
                channel = image[:, :, c]
                stats['channel_stats'].append({
                    'channel': c,
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel)),
                    'min': float(np.min(channel)),
                    'max': float(np.max(channel))
                })
        
        return stats
    
    # ============================================
    # Batch Processing
    # ============================================
    
    @staticmethod
    def process_batch(
        images: List[np.ndarray],
        operation: str,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Apply an operation to a batch of images
        
        Args:
            images: List of images
            operation: Operation name
            **kwargs: Operation arguments
        
        Returns:
            List of processed images
        """
        operations = {
            'resize': ImageUtils.resize,
            'normalize': ImageUtils.normalize,
            'to_rgb': ImageUtils.to_rgb,
            'enhance_contrast': ImageUtils.enhance_contrast,
            'enhance_brightness': ImageUtils.enhance_brightness,
            'apply_sharpening': ImageUtils.apply_sharpening,
            'apply_gaussian_blur': ImageUtils.apply_gaussian_blur,
            'convert_color_space': ImageUtils.convert_color_space
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        func = operations[operation]
        return [func(img, **kwargs) for img in images]
    
    # ============================================
    # Save and Load Utilities
    # ============================================
    
    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Save image to file
        
        Args:
            image: Image array
            output_path: Output path
            **kwargs: Additional arguments for rasterio/PIL
        
        Returns:
            True if saved successfully
        """
        output_path = Path(output_path)
        ext = output_path.suffix.lower()
        
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        try:
            if ext in ['.tif', '.tiff'] and RASTERIO_AVAILABLE:
                # Save as GeoTIFF
                if 'crs' in kwargs and 'transform' in kwargs:
                    with rasterio.open(
                        output_path, 'w',
                        driver='GTiff',
                        height=image_uint8.shape[0],
                        width=image_uint8.shape[1],
                        count=image_uint8.shape[-1] if len(image_uint8.shape) == 3 else 1,
                        dtype=image_uint8.dtype,
                        crs=kwargs['crs'],
                        transform=kwargs['transform']
                    ) as dst:
                        if len(image_uint8.shape) == 3:
                            dst.write(np.moveaxis(image_uint8, -1, 0))
                        else:
                            dst.write(image_uint8, 1)
                else:
                    Image.fromarray(image_uint8).save(output_path)
            else:
                Image.fromarray(image_uint8).save(output_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False


# ============================================
# Test Function
# ============================================

def test_image_utils():
    """Test the image utilities"""
    print("🧪 Testing ImageUtils...")
    print("=" * 60)
    
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test resizing
    print("\n📋 Testing resize:")
    resized = ImageUtils.resize(test_image, (128, 128))
    print(f"   Original shape: {test_image.shape}")
    print(f"   Resized shape: {resized.shape}")
    
    # Test normalization
    print("\n📋 Testing normalization:")
    normalized = ImageUtils.normalize(test_image, NormalizationMethod.MIN_MAX)
    print(f"   Normalized min: {np.min(normalized):.4f}")
    print(f"   Normalized max: {np.max(normalized):.4f}")
    
    # Test color conversion
    print("\n📋 Testing color conversion:")
    hsv = ImageUtils.convert_color_space(test_image, ColorSpace.RGB, ColorSpace.HSV)
    print(f"   HSV shape: {hsv.shape}")
    
    # Test enhancement
    print("\n📋 Testing enhancement:")
    enhanced = ImageUtils.enhance_contrast(test_image, factor=1.5)
    print(f"   Enhanced shape: {enhanced.shape}")
    print(f"   Enhanced mean: {np.mean(enhanced):.2f}")
    
    # Test stats
    print("\n📋 Testing statistics:")
    stats = ImageUtils.get_image_stats(test_image)
    print(f"   Shape: {stats['shape']}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std: {stats['std']:.2f}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_image_utils()