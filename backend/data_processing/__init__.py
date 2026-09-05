"""
Data Processing Module for SatQuery AI Backend

This module provides comprehensive data processing capabilities for satellite imagery:
1. Image Loading - Load various image formats (GeoTIFF, TIFF, PNG, JPEG)
2. Format Checking - Validate image formats and extract metadata
3. Image Processing - Resize, normalize, and enhance images
4. Geospatial Operations - CRS handling, reprojection, bounds extraction
5. Band Operations - Band extraction, NDVI/NDWI calculation
6. Image Registration - Co-registration of image pairs

All modules are designed to be:
- Modular and reusable
- Error-handled with proper logging
- Compatible with remote sensing workflows
"""

import logging
from typing import Dict, Any, Optional, List

# Setup logging
logger = logging.getLogger(__name__)

# ============================================
# Import Data Processing Modules
# ============================================

# Try importing each module with error handling
try:
    from .image_loader import ImageLoader
    logger.info("✅ ImageLoader imported")
except ImportError as e:
    logger.warning(f"⚠️ ImageLoader import failed: {e}")
    ImageLoader = None

try:
    from .format_checker import FormatChecker
    logger.info("✅ FormatChecker imported")
except ImportError as e:
    logger.warning(f"⚠️ FormatChecker import failed: {e}")
    FormatChecker = None

try:
    from .image_utils import ImageUtils, InterpolationMethod, NormalizationMethod, ColorSpace
    logger.info("✅ ImageUtils imported")
except ImportError as e:
    logger.warning(f"⚠️ ImageUtils import failed: {e}")
    ImageUtils = None
    InterpolationMethod = None
    NormalizationMethod = None
    ColorSpace = None

try:
    from .geospatial_utils import GeospatialUtils
    logger.info("✅ GeospatialUtils imported")
except ImportError as e:
    logger.warning(f"⚠️ GeospatialUtils import failed: {e}")
    GeospatialUtils = None

try:
    from .band_operations import BandOperations
    logger.info("✅ BandOperations imported")
except ImportError as e:
    logger.warning(f"⚠️ BandOperations import failed: {e}")
    BandOperations = None

try:
    from .image_registration import ImageRegistration
    logger.info("✅ ImageRegistration imported")
except ImportError as e:
    logger.warning(f"⚠️ ImageRegistration import failed: {e}")
    ImageRegistration = None

try:
    from .patch_extractor import PatchExtractor
    logger.info("✅ PatchExtractor imported")
except ImportError as e:
    logger.warning(f"⚠️ PatchExtractor import failed: {e}")
    PatchExtractor = None

try:
    from .data_augmentation import DataAugmentation
    logger.info("✅ DataAugmentation imported")
except ImportError as e:
    logger.warning(f"⚠️ DataAugmentation import failed: {e}")
    DataAugmentation = None

try:
    from .visualizer import Visualizer
    logger.info("✅ Visualizer imported")
except ImportError as e:
    logger.warning(f"⚠️ Visualizer import failed: {e}")
    Visualizer = None

# ============================================
# Data Processing Registry
# ============================================

class DataProcessingRegistry:
    """
    Registry for all data processing modules
    Provides module creation and management
    """
    
    _modules = {
        'image_loader': {
            'class': ImageLoader,
            'name': 'ImageLoader',
            'description': 'Load various image formats (GeoTIFF, TIFF, PNG, JPEG)',
            'functions': ['load_image', 'load_images_from_directory', 'get_image_info']
        },
        'format_checker': {
            'class': FormatChecker,
            'name': 'FormatChecker',
            'description': 'Validate image formats and extract metadata',
            'functions': ['check_file_format', 'validate_pair', 'extract_geotiff_metadata']
        },
        'image_utils': {
            'class': ImageUtils,
            'name': 'ImageUtils',
            'description': 'Resize, normalize, and enhance images',
            'functions': ['resize', 'normalize', 'enhance_contrast', 'convert_color_space']
        },
        'geospatial_utils': {
            'class': GeospatialUtils,
            'name': 'GeospatialUtils',
            'description': 'CRS handling, reprojection, bounds extraction',
            'functions': ['get_crs_info', 'get_bounds_from_image', 'reproject_image']
        },
        'band_operations': {
            'class': BandOperations,
            'name': 'BandOperations',
            'description': 'Band extraction, NDVI/NDWI calculation',
            'functions': ['extract_bands', 'calculate_ndvi', 'calculate_ndwi', 'calculate_ndbi']
        },
        'image_registration': {
            'class': ImageRegistration,
            'name': 'ImageRegistration',
            'description': 'Co-registration of image pairs',
            'functions': ['register_images', 'compute_transform', 'apply_transform']
        },
        'patch_extractor': {
            'class': PatchExtractor,
            'name': 'PatchExtractor',
            'description': 'Extract patches from large images',
            'functions': ['extract_patches', 'extract_random_patches', 'extract_center_patch']
        },
        'data_augmentation': {
            'class': DataAugmentation,
            'name': 'DataAugmentation',
            'description': 'Data augmentation for training',
            'functions': ['rotate', 'flip', 'scale', 'add_noise', 'adjust_brightness']
        },
        'visualizer': {
            'class': Visualizer,
            'name': 'Visualizer',
            'description': 'Visualization utilities for images',
            'functions': ['display_image', 'display_bands', 'display_ndvi', 'display_change_map']
        }
    }
    
    @classmethod
    def get_module_names(cls) -> List[str]:
        """Get list of available module names"""
        return list(cls._modules.keys())
    
    @classmethod
    def get_module_info(cls, module_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific module"""
        return cls._modules.get(module_name)
    
    @classmethod
    def get_all_modules_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all modules"""
        return cls._modules
    
    @classmethod
    def is_available(cls, module_name: str) -> bool:
        """Check if a module is available"""
        if module_name not in cls._modules:
            return False
        return cls._modules[module_name]['class'] is not None

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Core Modules
    'ImageLoader',
    'FormatChecker',
    'ImageUtils',
    'GeospatialUtils',
    'BandOperations',
    'ImageRegistration',
    'PatchExtractor',
    'DataAugmentation',
    'Visualizer',
    
    # Enums and Constants
    'InterpolationMethod',
    'NormalizationMethod',
    'ColorSpace',
    
    # Registry
    'DataProcessingRegistry'
]

# ============================================
# Package Metadata
# ============================================

MODULE_FEATURES = {
    'image_loader': {
        'description': 'Load various image formats',
        'supported_formats': ['GeoTIFF', 'TIFF', 'PNG', 'JPEG'],
        'key_functions': ['load_image', 'get_image_info', 'load_images_from_directory']
    },
    'format_checker': {
        'description': 'Validate image formats and extract metadata',
        'key_functions': ['check_file_format', 'validate_pair', 'extract_geotiff_metadata']
    },
    'image_utils': {
        'description': 'Image preprocessing utilities',
        'key_functions': ['resize', 'normalize', 'enhance_contrast', 'convert_color_space']
    },
    'geospatial_utils': {
        'description': 'Geospatial operations',
        'key_functions': ['get_crs_info', 'reproject_image', 'get_bounds_from_image']
    },
    'band_operations': {
        'description': 'Band operations and indices',
        'key_functions': ['extract_bands', 'calculate_ndvi', 'calculate_ndwi']
    },
    'image_registration': {
        'description': 'Image co-registration',
        'key_functions': ['register_images', 'compute_transform', 'apply_transform']
    },
    'patch_extractor': {
        'description': 'Patch extraction from large images',
        'key_functions': ['extract_patches', 'extract_random_patches']
    },
    'data_augmentation': {
        'description': 'Data augmentation for training',
        'key_functions': ['rotate', 'flip', 'scale', 'add_noise']
    },
    'visualizer': {
        'description': 'Image visualization utilities',
        'key_functions': ['display_image', 'display_ndvi', 'display_change_map']
    }
}

# ============================================
# Test Function
# ============================================

def test_data_processing_package():
    """Test the data processing package imports"""
    print("🧪 Testing Data Processing Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Data Processing Registry:")
    print(f"  Available modules: {DataProcessingRegistry.get_module_names()}")
    
    for name, info in DataProcessingRegistry.get_all_modules_info().items():
        status = "✅" if info['class'] is not None else "❌"
        print(f"  {status} {name}: {info['description']}")
    
    # Test module features
    print("\n📋 Module Features:")
    for name, features in MODULE_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Description: {features['description']}")
        if 'supported_formats' in features:
            print(f"     Supported Formats: {', '.join(features['supported_formats'])}")
        if 'key_functions' in features:
            print(f"     Key Functions: {', '.join(features['key_functions'])}")
    
    # Test availability
    print("\n📋 Module Availability:")
    for name in DataProcessingRegistry.get_module_names():
        available = DataProcessingRegistry.is_available(name)
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    print("\n✅ Data Processing Package test complete!")


if __name__ == "__main__":
    test_data_processing_package()