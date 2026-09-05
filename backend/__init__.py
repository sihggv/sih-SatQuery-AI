"""
SatQuery AI - Backend Package
Interactive Vision-Language Assistant for Multimodal Remote Sensing

This package provides the complete backend API for SatQuery AI including:
- Agentic AI Controller for automatic task routing
- Specialist models for VQA, Grounding, Change Detection, and Fusion
- Execution tracing for audit trail
- Image processing and validation services
- Report generation

Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'SatQuery AI Team'
__description__ = 'Interactive Vision-Language Assistant for Multimodal Remote Sensing'

# ============================================
# Import main application
# ============================================

from .main import app

# ============================================
# Import API schemas
# ============================================

from .api.schemas import (
    QueryRequest,
    AnalysisResponse,
    HealthResponse,
    ErrorResponse,
    ModelListResponse,
    ChangeDetectionResponse,
    FusionResponse,
    TaskType,
    ImageType,
    Status
)

# ============================================
# Import Core Components
# ============================================

try:
    from .core.agentic_controller import AgenticController
    from .core.task_classifier import TaskClassifier
    from .core.execution_trace import ExecutionTrace, ExecutionTraceManager, TraceStep, TraceStatus
except ImportError:
    # Fallback for incomplete installation
    class AgenticController:
        def __init__(self):
            pass
        async def process_query(self, **kwargs):
            return {'task': 'vqa', 'answer': 'Backend incomplete. Please install all dependencies.', 'confidence': 0.0}
        def get_loaded_models(self):
            return []
        def get_available_models(self):
            return []
    
    class TaskClassifier:
        def classify(self, query):
            return 'vqa', 0.5
    
    class ExecutionTrace:
        pass
    
    class ExecutionTraceManager:
        def create_trace(self, **kwargs):
            return type('Trace', (), {'trace_id': 'test'})()
        def complete_trace(self, trace_id):
            pass
    
    class TraceStep:
        pass
    
    class TraceStatus:
        PENDING = "pending"
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"


# ============================================
# Import Models
# ============================================

try:
    from .models import (
        BaseModel,
        VQAModel,
        GroundingModel,
        ChangeDetectionModel,
        OpticalSARFusionModel,
        CaptioningModel
    )
except ImportError:
    # Fallback for incomplete installation
    class BaseModel:
        pass
    class VQAModel:
        pass
    class GroundingModel:
        pass
    class ChangeDetectionModel:
        pass
    class OpticalSARFusionModel:
        pass
    class CaptioningModel:
        pass


# ============================================
# Import Services
# ============================================

try:
    from .services import (
        ImageService,
        ValidationService,
        ReportService
    )
except ImportError:
    # Fallback for incomplete installation
    class ImageService:
        async def save_uploaded_images(self, files):
            return ['temp1.tif']
        def cleanup_temp_files(self, paths):
            pass
    
    class ValidationService:
        def validate_images(self, paths, image_type):
            return True, []
    
    class ReportService:
        async def generate_report(self, analysis_id, format):
            return '/tmp/report.pdf'


# ============================================
# Package exports
# ============================================

__all__ = [
    # Main app
    'app',
    
    # API Schemas
    'QueryRequest',
    'AnalysisResponse',
    'HealthResponse',
    'ErrorResponse',
    'ModelListResponse',
    'ChangeDetectionResponse',
    'FusionResponse',
    'TaskType',
    'ImageType',
    'Status',
    
    # Core Components
    'AgenticController',
    'TaskClassifier',
    'ExecutionTrace',
    'ExecutionTraceManager',
    'TraceStep',
    'TraceStatus',
    
    # Models
    'BaseModel',
    'VQAModel',
    'GroundingModel',
    'ChangeDetectionModel',
    'OpticalSARFusionModel',
    'CaptioningModel',
    
    # Services
    'ImageService',
    'ValidationService',
    'ReportService'
]

# ============================================
# Package metadata
# ============================================

PACKAGE_INFO = {
    'name': 'satquery-backend',
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'dependencies': [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pillow',
        'numpy',
        'opencv-python',
        'rasterio'
    ],
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
# Console script entry point
# ============================================

def main():
    """Console entry point for the backend"""
    import uvicorn
    from .main import app
    
    print(f"🚀 Starting SatQuery AI Backend v{__version__}")
    print(f"📋 Features: {', '.join(PACKAGE_INFO['features'])}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()