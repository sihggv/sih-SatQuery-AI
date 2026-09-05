"""
Services Package for SatQuery AI Backend

This package contains all service modules that handle specific functionalities:
1. ImageService - Image upload, storage, and management
2. ValidationService - Input validation and format checking
3. ReportService - PDF and JSON report generation
4. FileService - File handling and cleanup
5. CacheService - Caching for performance

All services are designed to be:
- Modular and reusable
- Asynchronous for better performance
- Error-handled with proper logging
"""

import logging
from typing import Dict, Any, Optional, List

# Setup logging
logger = logging.getLogger(__name__)

# ============================================
# Import Services
# ============================================

# Try importing each service with error handling
try:
    from .image_service import ImageService
    logger.info("✅ ImageService imported")
except ImportError as e:
    logger.warning(f"⚠️ ImageService import failed: {e}")
    ImageService = None

try:
    from .validation_service import ValidationService
    logger.info("✅ ValidationService imported")
except ImportError as e:
    logger.warning(f"⚠️ ValidationService import failed: {e}")
    ValidationService = None

try:
    from .report_service import ReportService
    logger.info("✅ ReportService imported")
except ImportError as e:
    logger.warning(f"⚠️ ReportService import failed: {e}")
    ReportService = None

try:
    from .file_service import FileService
    logger.info("✅ FileService imported")
except ImportError as e:
    logger.warning(f"⚠️ FileService import failed: {e}")
    FileService = None

try:
    from .cache_service import CacheService
    logger.info("✅ CacheService imported")
except ImportError as e:
    logger.warning(f"⚠️ CacheService import failed: {e}")
    CacheService = None

# ============================================
# Service Registry
# ============================================

class ServiceRegistry:
    """
    Registry for all available services
    Provides service creation and management
    """
    
    _services = {
        'image': {
            'class': ImageService,
            'name': 'ImageService',
            'description': 'Image upload, storage, and management'
        },
        'validation': {
            'class': ValidationService,
            'name': 'ValidationService',
            'description': 'Input validation and format checking'
        },
        'report': {
            'class': ReportService,
            'name': 'ReportService',
            'description': 'PDF and JSON report generation'
        },
        'file': {
            'class': FileService,
            'name': 'FileService',
            'description': 'File handling and cleanup'
        },
        'cache': {
            'class': CacheService,
            'name': 'CacheService',
            'description': 'Caching for performance'
        }
    }
    
    @classmethod
    def get_service_names(cls) -> List[str]:
        """Get list of available service names"""
        return list(cls._services.keys())
    
    @classmethod
    def get_service_info(cls, service_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific service"""
        return cls._services.get(service_name)
    
    @classmethod
    def get_all_services_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all services"""
        return cls._services
    
    @classmethod
    def create_service(cls, service_name: str, **kwargs) -> Optional[Any]:
        """
        Create an instance of a service
        
        Args:
            service_name: Name of the service
            **kwargs: Additional arguments for service initialization
        
        Returns:
            Service instance or None if creation fails
        """
        service_info = cls._services.get(service_name)
        if not service_info:
            logger.error(f"Service '{service_name}' not found")
            return None
        
        service_class = service_info.get('class')
        if not service_class:
            logger.error(f"Service class for '{service_name}' not available")
            return None
        
        try:
            service = service_class(**kwargs)
            logger.info(f"✅ Created service: {service_name}")
            return service
        except Exception as e:
            logger.error(f"Failed to create service '{service_name}': {e}")
            return None
    
    @classmethod
    def is_available(cls, service_name: str) -> bool:
        """Check if a service is available"""
        if service_name not in cls._services:
            return False
        return cls._services[service_name]['class'] is not None

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Services
    'ImageService',
    'ValidationService',
    'ReportService',
    'FileService',
    'CacheService',
    
    # Registry
    'ServiceRegistry'
]

# ============================================
# Package Metadata
# ============================================

SERVICE_FEATURES = {
    'image': {
        'description': 'Handles image upload, storage, and management',
        'features': [
            'Upload images',
            'Validate formats',
            'Store temporarily',
            'Cleanup files'
        ]
    },
    'validation': {
        'description': 'Validates input images and formats',
        'features': [
            'Check file formats',
            'Validate GeoTIFF',
            'Extract metadata',
            'Check compatibility'
        ]
    },
    'report': {
        'description': 'Generates PDF and JSON reports',
        'features': [
            'PDF generation',
            'JSON export',
            'Visual evidence',
            'Confidence scores'
        ]
    },
    'file': {
        'description': 'Handles file operations and cleanup',
        'features': [
            'File saving',
            'Temp file management',
            'File deletion',
            'Directory management'
        ]
    },
    'cache': {
        'description': 'Manages caching for performance',
        'features': [
            'In-memory cache',
            'Cache invalidation',
            'TTL management',
            'LRU eviction'
        ]
    }
}

# ============================================
# Test Function
# ============================================

def test_services_package():
    """Test the services package imports"""
    print("🧪 Testing Services Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Service Registry:")
    print(f"  Available services: {ServiceRegistry.get_service_names()}")
    
    for name, info in ServiceRegistry.get_all_services_info().items():
        status = "✅" if info['class'] is not None else "❌"
        print(f"  {status} {name}: {info['description']}")
    
    # Test service features
    print("\n📋 Service Features:")
    for name, features in SERVICE_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Description: {features['description']}")
        for feature in features['features']:
            print(f"     - {feature}")
    
    # Test creating services
    print("\n📋 Creating Services:")
    for name in ServiceRegistry.get_service_names()[:3]:  # Test first 3 services
        service = ServiceRegistry.create_service(name)
        if service:
            print(f"  ✅ {name}: Created successfully")
        else:
            print(f"  ❌ {name}: Creation failed")
    
    print("\n✅ Services Package test complete!")


if __name__ == "__main__":
    test_services_package()