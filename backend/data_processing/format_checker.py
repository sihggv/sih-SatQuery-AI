"""
Format Checker Module for SatQuery AI

This module validates image formats and extracts metadata:
1. GeoTIFF/TIFF validation
2. Format detection
3. Metadata extraction
4. Image compatibility checking
5. CRS validation
6. Band information extraction

Features:
- Comprehensive format validation
- Geospatial metadata extraction
- Image compatibility checking
- Error reporting with details
- Support for multiple formats
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing rasterio
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.errors import RasterioError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Try importing PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

class ValidationStatus:
    """Validation status constants"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


class FormatType:
    """Format type constants"""
    GEOTIFF = "GeoTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    UNKNOWN = "unknown"


# ============================================
# Format Checker Class
# ============================================

class FormatChecker:
    """
    Validates image formats and extracts metadata
    
    Supported formats:
    - GeoTIFF (.tif, .tiff)
    - TIFF (.tif, .tiff)
    - PNG (.png)
    - JPEG (.jpg, .jpeg)
    """
    
    SUPPORTED_FORMATS = {
        '.tif': FormatType.GEOTIFF,
        '.tiff': FormatType.GEOTIFF,
        '.png': FormatType.PNG,
        '.jpg': FormatType.JPEG,
        '.jpeg': FormatType.JPEG
    }
    
    MINIMUM_IMAGE_SIZE = (10, 10)  # Minimum width, height
    MAXIMUM_IMAGE_SIZE = (10000, 10000)  # Maximum width, height
    
    @staticmethod
    def check_file_format(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Check file format and return validation results
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        # Initialize result
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'format_type': FormatType.UNKNOWN,
            'is_valid': False,
            'status': ValidationStatus.ERROR,
            'errors': [],
            'warnings': [],
            'metadata': {},
            'geospatial': {},
            'band_info': {}
        }
        
        # Check if file exists
        if not file_path.exists():
            result['errors'].append(f"File not found: {file_path}")
            return result
        
        # Check file extension
        ext = file_path.suffix.lower()
        if ext not in FormatChecker.SUPPORTED_FORMATS:
            result['errors'].append(f"Unsupported format: {ext}")
            result['format_type'] = FormatType.UNKNOWN
            return result
        
        result['format_type'] = FormatChecker.SUPPORTED_FORMATS[ext]
        
        # Validate based on format type
        if ext in ['.tif', '.tiff']:
            if RASTERIO_AVAILABLE:
                return FormatChecker._validate_geotiff(file_path, result)
            else:
                result['warnings'].append("rasterio not installed. Limited validation.")
                return FormatChecker._validate_tiff_with_pil(file_path, result)
        else:
            return FormatChecker._validate_rgb_image(file_path, result)
    
    @staticmethod
    def _validate_geotiff(
        file_path: Path,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate GeoTIFF using rasterio"""
        try:
            with rasterio.open(file_path) as src:
                # Basic validation
                if src.width < FormatChecker.MINIMUM_IMAGE_SIZE[0]:
                    result['warnings'].append(f"Image width too small: {src.width}")
                if src.height < FormatChecker.MINIMUM_IMAGE_SIZE[1]:
                    result['warnings'].append(f"Image height too small: {src.height}")
                if src.width > FormatChecker.MAXIMUM_IMAGE_SIZE[0]:
                    result['warnings'].append(f"Image width too large: {src.width}")
                if src.height > FormatChecker.MAXIMUM_IMAGE_SIZE[1]:
                    result['warnings'].append(f"Image height too large: {src.height}")
                
                # Extract metadata
                result['metadata'] = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': str(src.dtypes[0]),
                    'nodata': src.nodata,
                    'driver': src.driver,
                    'compression': src.compression if hasattr(src, 'compression') else None,
                    'block_size': (src.block_shapes[0][0] if src.block_shapes else 0, 
                                  src.block_shapes[0][1] if src.block_shapes else 0)
                }
                
                # Extract geospatial info
                result['geospatial'] = {
                    'crs': str(src.crs) if src.crs else None,
                    'crs_epsg': src.crs.to_epsg() if src.crs and hasattr(src.crs, 'to_epsg') else None,
                    'bounds': {
                        'left': src.bounds.left,
                        'bottom': src.bounds.bottom,
                        'right': src.bounds.right,
                        'top': src.bounds.top
                    },
                    'transform': str(src.transform) if src.transform else None,
                    'is_georeferenced': src.crs is not None
                }
                
                # Extract band info
                result['band_info'] = {
                    'num_bands': src.count,
                    'band_names': [f"Band_{i+1}" for i in range(src.count)],
                    'band_dtypes': [str(src.dtypes[i]) for i in range(src.count)]
                }
                
                # Check for common issues
                if src.crs is None:
                    result['warnings'].append("No CRS information found. Image may not be georeferenced.")
                
                # Check if image is readable
                try:
                    test_read = src.read(1, window=((0, 10), (0, 10)))
                    result['is_readable'] = True
                except:
                    result['warnings'].append("Image reading test failed. May be corrupted.")
                    result['is_readable'] = False
                
                # Mark as valid
                result['is_valid'] = True
                result['status'] = ValidationStatus.VALID if not result['warnings'] else ValidationStatus.WARNING
                
                return result
                
        except RasterioError as e:
            result['errors'].append(f"Rasterio error: {str(e)}")
            result['status'] = ValidationStatus.ERROR
            return result
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
            result['status'] = ValidationStatus.ERROR
            return result
    
    @staticmethod
    def _validate_tiff_with_pil(
        file_path: Path,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate TIFF using PIL (fallback)"""
        try:
            with Image.open(file_path) as img:
                # Get basic info
                width, height = img.size
                result['metadata'] = {
                    'width': width,
                    'height': height,
                    'mode': img.mode,
                    'format': img.format,
                    'info': img.info
                }
                
                # Band info
                bands = len(img.getbands())
                result['band_info'] = {
                    'num_bands': bands,
                    'band_names': list(img.getbands())
                }
                
                result['geospatial'] = {
                    'is_georeferenced': False,
                    'crs': None
                }
                
                # Validate
                if width < FormatChecker.MINIMUM_IMAGE_SIZE[0]:
                    result['warnings'].append(f"Image width too small: {width}")
                if height < FormatChecker.MINIMUM_IMAGE_SIZE[1]:
                    result['warnings'].append(f"Image height too small: {height}")
                
                result['is_valid'] = True
                result['status'] = ValidationStatus.WARNING  # Since no CRS info
                result['warnings'].append("No geospatial metadata available (PIL fallback)")
                
                return result
                
        except Exception as e:
            result['errors'].append(f"PIL error: {str(e)}")
            result['status'] = ValidationStatus.ERROR
            return result
    
    @staticmethod
    def _validate_rgb_image(
        file_path: Path,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate RGB image using PIL"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
                result['metadata'] = {
                    'width': width,
                    'height': height,
                    'mode': img.mode,
                    'format': img.format,
                    'info': img.info
                }
                
                bands = len(img.getbands())
                result['band_info'] = {
                    'num_bands': bands,
                    'band_names': list(img.getbands())
                }
                
                result['geospatial'] = {
                    'is_georeferenced': False,
                    'crs': None
                }
                
                # Validate
                if width < FormatChecker.MINIMUM_IMAGE_SIZE[0]:
                    result['warnings'].append(f"Image width too small: {width}")
                if height < FormatChecker.MINIMUM_IMAGE_SIZE[1]:
                    result['warnings'].append(f"Image height too small: {height}")
                
                result['is_valid'] = True
                result['status'] = ValidationStatus.VALID
                
                return result
                
        except Exception as e:
            result['errors'].append(f"PIL error: {str(e)}")
            result['status'] = ValidationStatus.ERROR
            return result
    
    # ============================================
    # Comparison and Validation Methods
    # ============================================
    
    @staticmethod
    def validate_pair(
        file1_path: Union[str, Path],
        file2_path: Union[str, Path],
        check_coregistration: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a pair of images (for cross-modal or bi-temporal analysis)
        
        Args:
            file1_path: First image path
            file2_path: Second image path
            check_coregistration: Whether to check co-registration
        
        Returns:
            Dictionary with validation results for both images
        """
        result1 = FormatChecker.check_file_format(file1_path)
        result2 = FormatChecker.check_file_format(file2_path)
        
        comparison = {
            'file1': result1,
            'file2': result2,
            'is_compatible': False,
            'issues': [],
            'differences': {},
            'recommendations': []
        }
        
        # Check if both are valid
        if not result1['is_valid'] or not result2['is_valid']:
            comparison['issues'].append("One or both files are invalid")
            return comparison
        
        # Compare dimensions
        dim1 = (result1['metadata'].get('width', 0), result1['metadata'].get('height', 0))
        dim2 = (result2['metadata'].get('width', 0), result2['metadata'].get('height', 0))
        
        if dim1 != dim2:
            comparison['differences']['dimensions'] = {
                'file1': dim1,
                'file2': dim2
            }
            comparison['issues'].append(f"Dimension mismatch: {dim1} vs {dim2}")
            comparison['recommendations'].append("Resize images to same dimensions")
        
        # Compare bands
        bands1 = result1['band_info'].get('num_bands', 0)
        bands2 = result2['band_info'].get('num_bands', 0)
        
        if bands1 != bands2:
            comparison['differences']['num_bands'] = {
                'file1': bands1,
                'file2': bands2
            }
            comparison['issues'].append(f"Band count mismatch: {bands1} vs {bands2}")
        
        # Compare CRS (if available)
        crs1 = result1['geospatial'].get('crs')
        crs2 = result2['geospatial'].get('crs')
        
        if crs1 and crs2:
            if crs1 != crs2:
                comparison['differences']['crs'] = {
                    'file1': crs1,
                    'file2': crs2
                }
                comparison['issues'].append("CRS mismatch")
                comparison['recommendations'].append("Reproject images to same CRS")
        
        # Check co-registration
        if check_coregistration and crs1 and crs2:
            bounds1 = result1['geospatial'].get('bounds', {})
            bounds2 = result2['geospatial'].get('bounds', {})
            
            if bounds1 and bounds2:
                overlap = FormatChecker._calculate_overlap(bounds1, bounds2)
                if overlap < 0.9:
                    comparison['issues'].append(f"Low overlap: {overlap:.2%}")
                    comparison['recommendations'].append("Ensure images are co-registered")
        
        # Determine compatibility
        comparison['is_compatible'] = len(comparison['issues']) == 0
        
        return comparison
    
    @staticmethod
    def _calculate_overlap(bounds1: Dict, bounds2: Dict) -> float:
        """Calculate overlap between two bounds"""
        try:
            # Get overlap area
            minx = max(bounds1.get('left', 0), bounds2.get('left', 0))
            miny = max(bounds1.get('bottom', 0), bounds2.get('bottom', 0))
            maxx = min(bounds1.get('right', 0), bounds2.get('right', 0))
            maxy = min(bounds1.get('top', 0), bounds2.get('top', 0))
            
            if minx >= maxx or miny >= maxy:
                return 0.0
            
            overlap_area = (maxx - minx) * (maxy - miny)
            
            area1 = (bounds1.get('right', 0) - bounds1.get('left', 0)) * \
                    (bounds1.get('top', 0) - bounds1.get('bottom', 0))
            
            return overlap_area / area1 if area1 > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def check_compatibility(
        image_paths: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        """
        Check compatibility of multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Dictionary with compatibility results
        """
        if len(image_paths) < 2:
            return {
                'is_compatible': True,
                'message': "Only one image provided"
            }
        
        results = [FormatChecker.check_file_format(p) for p in image_paths]
        
        # Check if all are valid
        all_valid = all(r['is_valid'] for r in results)
        
        if not all_valid:
            return {
                'is_compatible': False,
                'message': "One or more images are invalid",
                'invalid_indices': [i for i, r in enumerate(results) if not r['is_valid']]
            }
        
        # Check dimensions
        dims = [(r['metadata'].get('width', 0), r['metadata'].get('height', 0)) for r in results]
        all_same_dim = all(d == dims[0] for d in dims)
        
        if not all_same_dim:
            return {
                'is_compatible': False,
                'message': "Images have different dimensions",
                'dimensions': dims
            }
        
        # Check CRS
        crss = [r['geospatial'].get('crs') for r in results]
        all_same_crs = all(c == crss[0] for c in crss if c is not None)
        
        if not all_same_crs:
            return {
                'is_compatible': False,
                'message': "Images have different CRS",
                'crss': crss
            }
        
        return {
            'is_compatible': True,
            'message': "All images are compatible",
            'dimensions': dims[0],
            'num_images': len(image_paths)
        }
    
    @staticmethod
    def extract_geotiff_metadata(
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from GeoTIFF
        
        Args:
            file_path: Path to GeoTIFF
        
        Returns:
            Dictionary with comprehensive metadata
        """
        file_path = Path(file_path)
        
        if not RASTERIO_AVAILABLE:
            return {
                'error': 'rasterio not installed',
                'file_path': str(file_path)
            }
        
        try:
            with rasterio.open(file_path) as src:
                metadata = {
                    'file': {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size': file_path.stat().st_size
                    },
                    'image': {
                        'width': src.width,
                        'height': src.height,
                        'count': src.count,
                        'dtype': str(src.dtypes[0]),
                        'nodata': src.nodata,
                        'driver': src.driver,
                        'compression': src.compression if hasattr(src, 'compression') else None
                    },
                    'geospatial': {
                        'crs': str(src.crs) if src.crs else None,
                        'epsg': src.crs.to_epsg() if src.crs and hasattr(src.crs, 'to_epsg') else None,
                        'bounds': {
                            'left': src.bounds.left,
                            'bottom': src.bounds.bottom,
                            'right': src.bounds.right,
                            'top': src.bounds.top
                        },
                        'transform': list(src.transform) if src.transform else None,
                        'is_georeferenced': src.crs is not None
                    },
                    'bands': {
                        'count': src.count,
                        'descriptions': [src.descriptions[i] if src.descriptions else None for i in range(src.count)],
                        'dtypes': [str(src.dtypes[i]) for i in range(src.count)],
                        'block_shapes': [(src.block_shapes[i][0], src.block_shapes[i][1]) for i in range(src.count)]
                    },
                    'statistics': {
                        'min': [src.statistics(i).min if src.statistics(i) else None for i in range(1, src.count+1)],
                        'max': [src.statistics(i).max if src.statistics(i) else None for i in range(1, src.count+1)],
                        'mean': [src.statistics(i).mean if src.statistics(i) else None for i in range(1, src.count+1)],
                        'std': [src.statistics(i).std if src.statistics(i) else None for i in range(1, src.count+1)]
                    },
                    'tags': {
                        'all': src.tags(),
                        'profile': src.profile
                    }
                }
                
                return metadata
                
        except Exception as e:
            return {
                'error': str(e),
                'file_path': str(file_path)
            }
    
    @staticmethod
    def is_geotiff(file_path: Union[str, Path]) -> bool:
        """
        Check if file is a valid GeoTIFF
        
        Args:
            file_path: Path to file
        
        Returns:
            True if valid GeoTIFF
        """
        if not RASTERIO_AVAILABLE:
            return False
        
        try:
            with rasterio.open(file_path) as src:
                return src.crs is not None
        except:
            return False
    
    @staticmethod
    def get_required_bands(
        file_path: Union[str, Path],
        required_bands: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check if image has required bands
        
        Args:
            file_path: Path to image
            required_bands: List of required band names
        
        Returns:
            Dictionary with band availability
        """
        if required_bands is None:
            required_bands = ['Red', 'Green', 'Blue', 'NIR', 'SWIR']
        
        result = {
            'available_bands': [],
            'missing_bands': [],
            'has_all_required': False,
            'band_mapping': {}
        }
        
        try:
            with rasterio.open(file_path) as src:
                # Get band descriptions
                bands = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" 
                        for i in range(src.count)]
                
                result['available_bands'] = bands
                
                # Check required bands
                for req in required_bands:
                    if req in bands:
                        result['band_mapping'][req] = bands.index(req)
                    else:
                        result['missing_bands'].append(req)
                
                result['has_all_required'] = len(result['missing_bands']) == 0
                
        except Exception as e:
            result['error'] = str(e)
        
        return result


# ============================================
# Test Function
# ============================================

def test_format_checker():
    """Test the format checker module"""
    print("🧪 Testing FormatChecker...")
    print("=" * 60)
    
    # Create a dummy image for testing
    import tempfile
    from PIL import Image
    
    # Create a sample image
    sample_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(sample_img)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Test format check
        print("\n📋 Testing check_file_format:")
        result = FormatChecker.check_file_format(tmp_path)
        print(f"   Format: {result['format_type']}")
        print(f"   Valid: {result['is_valid']}")
        print(f"   Status: {result['status']}")
        print(f"   Metadata: {result['metadata']}")
        
        # Test compatibility
        print("\n📋 Testing check_compatibility:")
        compat = FormatChecker.check_compatibility([tmp_path, tmp_path])
        print(f"   Compatible: {compat['is_compatible']}")
        
        # Test supported formats
        print("\n📋 Supported Formats:")
        print(f"   {list(FormatChecker.SUPPORTED_FORMATS.keys())}")
        
    finally:
        os.unlink(tmp_path)
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_format_checker()