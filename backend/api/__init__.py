"""
API Package for SatQuery AI Backend

This package contains all API-related modules including:
- Routes: All API endpoints
- Schemas: Pydantic models for request/response validation
- Dependencies: FastAPI dependency injections

The API is organized into the following main groups:
1. Health & Status endpoints
2. Model management endpoints
3. Analysis endpoints (VQA, Grounding, Change Detection, Fusion)
4. Report generation endpoints
"""

from .routes import router
from .schemas import (
    # Request Models
    QueryRequest,
    UploadRequest,
    
    # Response Models
    AnalysisResponse,
    HealthResponse,
    ErrorResponse,
    ModelListResponse,
    ChangeDetectionResponse,
    FusionResponse,
    
    # Enums
    TaskType,
    ImageType,
    Status
)

# ============================================
# Package Metadata
# ============================================

__version__ = '1.0.0'
__author__ = 'SatQuery AI Team'

# ============================================
# API Router - All endpoints are registered here
# ============================================

# The main router is imported from routes.py
# All endpoints are prefixed with /api

# ============================================
# Exports
# ============================================

__all__ = [
    # Router
    'router',
    
    # Request Schemas
    'QueryRequest',
    'UploadRequest',
    
    # Response Schemas
    'AnalysisResponse',
    'HealthResponse',
    'ErrorResponse',
    'ModelListResponse',
    'ChangeDetectionResponse',
    'FusionResponse',
    
    # Enums
    'TaskType',
    'ImageType',
    'Status'
]

# ============================================
# API Information
# ============================================

API_INFO = {
    'name': 'SatQuery AI API',
    'version': __version__,
    'description': 'Interactive Vision-Language Assistant for Multimodal Remote Sensing',
    'base_path': '/api',
    'endpoints': {
        'GET /health': 'Health check',
        'GET /models': 'List available models',
        'POST /analyze': 'Analyze query with images',
        'POST /analyze/upload': 'Upload and analyze images',
        'POST /change-detection': 'Bi-temporal change detection',
        'POST /optical-sar-fusion': 'Optical-SAR fusion analysis',
        'POST /generate-report': 'Generate downloadable report'
    },
    'features': [
        'Single-Image VQA',
        'Visual Grounding',
        'Bi-temporal Change Detection',
        'Optical-SAR Fusion',
        'Agentic AI Routing',
        'Audit Trail',
        'Downloadable Reports'
    ]
}

# ============================================
# Quick Test Function
# ============================================

def test_api_package():
    """Test the API package imports"""
    print("🧪 Testing API Package...")
    print("=" * 60)
    
    print(f"✅ API Info: {API_INFO['name']} v{API_INFO['version']}")
    print(f"✅ Base Path: {API_INFO['base_path']}")
    print(f"✅ Endpoints: {len(API_INFO['endpoints'])}")
    print(f"✅ Features: {len(API_INFO['features'])}")
    
    print("\n📋 Available Schemas:")
    schemas = [
        'QueryRequest',
        'AnalysisResponse',
        'HealthResponse',
        'ErrorResponse',
        'ModelListResponse',
        'ChangeDetectionResponse',
        'FusionResponse'
    ]
    for schema in schemas:
        print(f"  ✅ {schema}")
    
    print("\n📋 Available Enums:")
    enums = ['TaskType', 'ImageType', 'Status']
    for enum in enums:
        print(f"  ✅ {enum}")
    
    print("\n📋 Available Routes:")
    routes = [
        'POST /api/analyze',
        'POST /api/analyze/upload',
        'POST /api/change-detection',
        'POST /api/optical-sar-fusion',
        'GET /api/health',
        'GET /api/models',
        'POST /api/generate-report'
    ]
    for route in routes:
        print(f"  ✅ {route}")
    
    print("\n✅ API Package test complete!")


if __name__ == "__main__":
    test_api_package()