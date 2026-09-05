"""
Validation Service for SatQuery AI Backend

This service handles all validation operations:
1. Image format validation (GeoTIFF, TIFF, PNG, JPEG)
2. Image compatibility validation
3. Geospatial metadata validation
4. File size and integrity validation
5. Input query validation
6. Cross-modal and bi-temporal pair validation

Features:
- Comprehensive image validation
- Geospatial metadata checking
- Format compatibility validation
- Query validation
- Error reporting with details
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

# Import data processing modules
try:
    from data_processing import FormatChecker, ImageLoader, GeospatialUtils
except ImportError:
    # Fallback for testing
    class FormatChecker:
        @staticmethod
        def check_file_format(file_path):
            return {'is_valid': True, 'format_type': 'GeoTIFF', 'metadata': {}, 'errors': [], 'warnings': []}
        @staticmethod
        def validate_pair(file1, file2):
            return {'is_compatible': True, 'issues': []}
        @staticmethod
        def check_compatibility(file_paths):
            return {'is_compatible': True}
    
    class ImageLoader:
        @staticmethod
        def get_image_info(file_path):
            return {'width': 1024, 'height': 1024, 'bands': 3}
    
    class GeospatialUtils:
        @staticmethod
        def get_bounds_from_image(file_path):
            return {'left': 0, 'bottom': 0, 'right': 1, 'top': 1}
        @staticmethod
        def get_crs_info(crs):
            return {'epsg': 4326, 'is_geographic': True}

logger = logging.getLogger(__name__)


# ============================================
# Validation Service Class
# ============================================

class ValidationService:
    """
    Validation Service for all input validation
    
    Features:
    1. Image format validation
    2. Geospatial metadata validation
    3. Image compatibility validation
    4. Query validation
    5. Cross-modal pair validation
    6. Bi-temporal pair validation
    """
    
    def __init__(
        self,
        max_file_size: int = 100 * 1024 * 1024,  # 100 MB
        supported_formats: Optional[List[str]] = None,
        require_geospatial: bool = False
    ):
        """
        Initialize Validation Service
        
        Args:
            max_file_size: Maximum file size in bytes
            supported_formats: List of supported formats
            require_geospatial: Whether geospatial metadata is required
        """
        self.max_file_size = max_file_size
        self.supported_formats = supported_formats or ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.require_geospatial = require_geospatial
        
        logger.info("✅ ValidationService initialized")
    
    # ============================================
    # Image Validation Methods
    # ============================================
    
    def validate_image(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single image file
        
        Args:
            file_path: Path to image file
        
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': False,
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'errors': [],
            'warnings': [],
            'metadata': {},
            'format_type': 'unknown'
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            result['errors'].append(f"File not found: {file_path}")
            return result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            result['errors'].append(f"File too large: {file_size} bytes (max: {self.max_file_size} bytes)")
        
        # Check format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            result['errors'].append(f"Unsupported format: {ext}. Supported: {', '.join(self.supported_formats)}")
            return result
        
        # Check if file is a valid image
        try:
            format_result = FormatChecker.check_file_format(file_path)
            result['format_type'] = format_result.get('format_type', 'unknown')
            result['metadata'] = format_result.get('metadata', {})
            result['geospatial'] = format_result.get('geospatial', {})
            
            if not format_result.get('is_valid', False):
                result['errors'].extend(format_result.get('errors', []))
            else:
                result['warnings'].extend(format_result.get('warnings', []))
                
                # Check geospatial metadata if required
                if self.require_geospatial:
                    if not result['geospatial'].get('is_georeferenced', False):
                        result['errors'].append("Image is not georeferenced (CRS missing)")
            
            # Check if image is readable
            try:
                ImageLoader.get_image_info(file_path)
            except Exception as e:
                result['errors'].append(f"Image reading failed: {str(e)}")
            
        except Exception as e:
            result['errors'].append(f"Format check failed: {str(e)}")
            return result
        
        # Determine validity
        result['valid'] = len(result['errors']) == 0
        
        return result
    
    def validate_images(self, file_paths: List[str], image_type: str) -> Tuple[bool, List[str]]:
        """
        Validate multiple images for a specific image type
        
        Args:
            file_paths: List of image paths
            image_type: Type of image(s)
        
        Returns:
            Tuple of (is_valid, issues_list)
        """
        issues = []
        
        # Check number of images
        if image_type in ['single_optical', 'single_sar']:
            if len(file_paths) != 1:
                issues.append(f"Single image type requires exactly 1 image, got {len(file_paths)}")
        elif image_type in ['cross_modal', 'bi_temporal']:
            if len(file_paths) != 2:
                issues.append(f"Multi-image type requires exactly 2 images, got {len(file_paths)}")
        
        # Validate each image
        for i, file_path in enumerate(file_paths):
            result = self.validate_image(file_path)
            if not result['valid']:
                issues.append(f"Image {i+1} invalid: {', '.join(result['errors'])}")
            elif result['warnings']:
                issues.extend([f"Image {i+1}: {w}" for w in result['warnings']])
        
        # Check image compatibility
        if len(file_paths) >= 2:
            compat_result = self.validate_pair(file_paths[0], file_paths[1])
            if not compat_result['is_compatible']:
                issues.extend(compat_result['issues'])
        
        return len(issues) == 0, issues
    
    def validate_image_type(self, image_type: str) -> Tuple[bool, str]:
        """
        Validate image type string
        
        Args:
            image_type: Image type to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_types = ['single_optical', 'single_sar', 'cross_modal', 'bi_temporal']
        
        if image_type not in valid_types:
            return False, f"Invalid image_type: {image_type}. Must be one of: {valid_types}"
        
        return True, ""
    
    # ============================================
    # Pair Validation Methods
    # ============================================
    
    def validate_pair(self, file1_path: str, file2_path: str) -> Dict[str, Any]:
        """
        Validate a pair of images (for cross-modal or bi-temporal analysis)
        
        Args:
            file1_path: First image path
            file2_path: Second image path
        
        Returns:
            Validation result dictionary
        """
        result = {
            'is_compatible': False,
            'file1': os.path.basename(file1_path),
            'file2': os.path.basename(file2_path),
            'issues': [],
            'differences': {},
            'recommendations': []
        }
        
        # Validate individual images
        result1 = self.validate_image(file1_path)
        result2 = self.validate_image(file2_path)
        
        if not result1['valid']:
            result['issues'].append(f"Image 1 invalid: {', '.join(result1['errors'])}")
        if not result2['valid']:
            result['issues'].append(f"Image 2 invalid: {', '.join(result2['errors'])}")
        
        if not result1['valid'] or not result2['valid']:
            return result
        
        # Compare formats
        if result1['format_type'] != result2['format_type']:
            result['differences']['format'] = {
                'file1': result1['format_type'],
                'file2': result2['format_type']
            }
            result['issues'].append("Format mismatch")
        
        # Compare dimensions
        dim1 = (result1['metadata'].get('width', 0), result1['metadata'].get('height', 0))
        dim2 = (result2['metadata'].get('width', 0), result2['metadata'].get('height', 0))
        
        if dim1 != dim2 and all(d > 0 for d in dim1 + dim2):
            result['differences']['dimensions'] = {
                'file1': dim1,
                'file2': dim2
            }
            result['issues'].append(f"Dimension mismatch: {dim1} vs {dim2}")
            result['recommendations'].append("Resize images to same dimensions")
        
        # Compare CRS (if available)
        crs1 = result1['geospatial'].get('crs') if 'geospatial' in result1 else None
        crs2 = result2['geospatial'].get('crs') if 'geospatial' in result2 else None
        
        if crs1 and crs2:
            if crs1 != crs2:
                result['differences']['crs'] = {
                    'file1': crs1,
                    'file2': crs2
                }
                result['issues'].append("CRS mismatch")
                result['recommendations'].append("Reproject images to same CRS")
        
        # Check overlap (for georeferenced images)
        if crs1 and crs2:
            bounds1 = result1['geospatial'].get('bounds')
            bounds2 = result2['geospatial'].get('bounds')
            
            if bounds1 and bounds2:
                overlap = self._calculate_overlap(bounds1, bounds2)
                if overlap < 0.9:
                    result['issues'].append(f"Low overlap: {overlap:.2%}")
                    result['recommendations'].append("Ensure images are co-registered")
                result['overlap'] = overlap
        
        # Check bands
        bands1 = result1['metadata'].get('bands', 0)
        bands2 = result2['metadata'].get('bands', 0)
        if bands1 != bands2:
            result['differences']['bands'] = {
                'file1': bands1,
                'file2': bands2
            }
            result['issues'].append(f"Band count mismatch: {bands1} vs {bands2}")
        
        # Determine compatibility
        result['is_compatible'] = len(result['issues']) == 0
        
        return result
    
    def _calculate_overlap(self, bounds1: Dict, bounds2: Dict) -> float:
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
    
    # ============================================
    # Query Validation Methods
    # ============================================
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate a query string
        
        Args:
            query: Query to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query.strip()) < 3:
            return False, "Query is too short (minimum 3 characters)"
        
        if len(query.strip()) > 1000:
            return False, "Query is too long (maximum 1000 characters)"
        
        # Check for invalid characters
        invalid_chars = ['\x00', '\x01', '\x02']
        for char in invalid_chars:
            if char in query:
                return False, f"Query contains invalid character: {repr(char)}"
        
        return True, ""
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query
        
        Args:
            query: Query string
        
        Returns:
            List of keywords
        """
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        words = cleaned.lower().split()
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'this', 'that', 'these', 'those',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'to', 'for', 'of', 'with', 'without', 'and', 'or', 'but',
                    'in', 'on', 'at', 'by', 'from', 'up', 'down'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    # ============================================
    # Format Validation Methods
    # ============================================
    
    def validate_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file format
        
        Args:
            file_path: Path to file
        
        Returns:
            Format validation result
        """
        result = {
            'is_valid': False,
            'format': 'unknown',
            'supported': False,
            'details': {}
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        result['format'] = ext
        
        if ext in self.supported_formats:
            result['supported'] = True
        else:
            result['details']['error'] = f"Unsupported format: {ext}"
            return result
        
        # Check using FormatChecker
        try:
            check_result = FormatChecker.check_file_format(file_path)
            result['is_valid'] = check_result.get('is_valid', False)
            result['details']['metadata'] = check_result.get('metadata', {})
            result['details']['format_type'] = check_result.get('format_type', 'unknown')
            result['details']['warnings'] = check_result.get('warnings', [])
            result['details']['errors'] = check_result.get('errors', [])
        except Exception as e:
            result['details']['error'] = str(e)
        
        return result
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def get_validation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of validation results
        
        Args:
            results: List of validation results
        
        Returns:
            Summary dictionary
        """
        total = len(results)
        valid = sum(1 for r in results if r.get('valid', False))
        
        errors = []
        warnings = []
        
        for r in results:
            errors.extend(r.get('errors', []))
            warnings.extend(r.get('warnings', []))
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'error_summary': errors[:10],
            'warning_summary': warnings[:10]
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats"""
        return self.supported_formats
    
    def add_supported_format(self, format_ext: str):
        """Add a format to supported formats"""
        if not format_ext.startswith('.'):
            format_ext = '.' + format_ext
        if format_ext not in self.supported_formats:
            self.supported_formats.append(format_ext)
            logger.info(f"Added supported format: {format_ext}")


# ============================================
# Test Functions
# ============================================

def test_validation_service():
    """Test the validation service"""
    print("🧪 Testing ValidationService...")
    print("=" * 60)
    
    # Create service
    service = ValidationService()
    print(f"✅ Service initialized")
    print(f"   Supported formats: {service.get_supported_formats()}")
    
    # Test query validation
    print("\n📋 Query Validation:")
    test_queries = [
        ("What is the land-cover?", True),
        ("", False),
        ("   ", False),
        ("Hi", False),
        ("A" * 1001, False)
    ]
    
    for query, expected in test_queries:
        is_valid, error = service.validate_query(query)
        status = "✅" if is_valid == expected else "❌"
        print(f"  {status} Query: '{query[:30]}...' -> Valid: {is_valid}")
    
    # Test image type validation
    print("\n📋 Image Type Validation:")
    test_types = [
        ('single_optical', True),
        ('single_sar', True),
        ('cross_modal', True),
        ('bi_temporal', True),
        ('invalid_type', False)
    ]
    
    for img_type, expected in test_types:
        is_valid, error = service.validate_image_type(img_type)
        status = "✅" if is_valid == expected else "❌"
        print(f"  {status} Type: {img_type} -> Valid: {is_valid}")
    
    # Test keyword extraction
    print("\n📋 Keyword Extraction:")
    query = "What is the land-cover and urban area percentage in this image?"
    keywords = service.extract_keywords(query)
    print(f"  Query: {query}")
    print(f"  Keywords: {keywords}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_validation_service()