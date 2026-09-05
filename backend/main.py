"""
SatQuery AI - Main FastAPI Application Entry Point

This is the main entry point for the SatQuery AI backend server.
It initializes the FastAPI application, configures middleware,
and sets up all API routes.

Features:
- FastAPI with automatic OpenAPI documentation
- CORS middleware for frontend communication
- Agentic AI controller for intelligent task routing
- Health checks and monitoring
- Error handling and logging
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import time
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Import Routes and Schemas
# ============================================

from api.routes import router
from api.schemas import HealthResponse, ErrorResponse

# ============================================
# Create FastAPI App
# ============================================

app = FastAPI(
    title="SatQuery AI API",
    description="""
    ## 🛰️ SatQuery AI - Interactive Vision-Language Assistant for Multimodal Remote Sensing
    
    SatQuery AI is an intelligent assistant that allows users to analyze satellite imagery 
    through natural language queries. No GIS expertise required!
    
    ### Key Capabilities:
    
    - **Single-Image VQA**: Ask questions about any satellite image in plain English
    - **Visual Grounding**: Highlight specific objects and regions with bounding boxes
    - **Bi-temporal Change Detection**: Compare two dates and see what changed
    - **Optical-SAR Fusion**: Combine optical and radar imagery for complete analysis
    - **Agentic AI**: Automatic task routing - no manual model selection needed
    - **Audit Trail**: Complete execution trace for transparency and trust
    
    ### How to Use:
    
    1. Upload satellite images (GeoTIFF, TIFF, PNG, JPEG)
    2. Ask a question in natural language
    3. Get answers with visual evidence and confidence scores
    4. Download reports for documentation
    
    ### Target Users:
    
    - Disaster Management Authorities
    - Urban Planners and Municipal Corporations
    - Agriculture and Forest Departments
    - Defense and Border Security Analysts
    - Environmental Researchers and Policy Makers
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints"
        },
        {
            "name": "Models",
            "description": "AI model management endpoints"
        },
        {
            "name": "Analysis",
            "description": "Main analysis endpoints for image querying"
        },
        {
            "name": "Change Detection",
            "description": "Specialized change detection endpoints"
        },
        {
            "name": "Fusion",
            "description": "Optical-SAR fusion endpoints"
        },
        {
            "name": "Reports",
            "description": "Report generation endpoints"
        }
    ]
)

# ============================================
# CORS Middleware Configuration
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8502",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8502",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================
# Request Logging Middleware
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing"""
    start_time = time.time()
    
    # Get request details
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log
    logger.info(
        f"{client_host} - {method} {url} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    # Add duration header
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response

# ============================================
# Include API Routes
# ============================================

app.include_router(router)

# ============================================
# Root Endpoint
# ============================================

@app.get(
    "/",
    tags=["Root"],
    summary="API Information",
    description="Get information about the SatQuery AI API"
)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SatQuery AI API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "description": "Interactive Vision-Language Assistant for Multimodal Remote Sensing",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "analyze": "/api/analyze",
            "upload": "/api/analyze/upload",
            "change_detection": "/api/change-detection",
            "fusion": "/api/optical-sar-fusion",
            "report": "/api/generate-report"
        },
        "features": [
            "Single-Image VQA",
            "Visual Grounding",
            "Bi-temporal Change Detection",
            "Optical-SAR Fusion",
            "Agentic AI Routing",
            "Audit Trail",
            "Downloadable Reports"
        ]
    }

# ============================================
# Health Check Endpoint
# ============================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API is healthy and all models are loaded"
)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check if models are loaded
        from core.agentic_controller import AgenticController
        controller = AgenticController()
        models_loaded = controller.get_loaded_models()
        
        # Check if services are available
        services_available = True
        try:
            from services.image_service import ImageService
            ImageService()
        except:
            services_available = False
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            version="1.0.0",
            services_available=services_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            models_loaded=[],
            version="1.0.0"
        )

# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    errors = []
    for error in exc.errors():
        errors.append(f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}")
    
    error_msg = "; ".join(errors)
    logger.error(f"Validation Error: {error_msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=f"Validation error: {error_msg}",
            error_code="VALIDATION_ERROR",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=f"Internal server error: {str(exc)}",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ============================================
# Startup and Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    logger.info("=" * 60)
    logger.info("🚀 Starting SatQuery AI Backend Server")
    logger.info(f"📅 Started at: {datetime.now().isoformat()}")
    logger.info(f"📖 Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)
    
    # Initialize Agentic Controller
    try:
        from core.agentic_controller import AgenticController
        controller = AgenticController()
        logger.info(f"✅ Agentic Controller initialized with {len(controller.get_loaded_models())} models")
    except Exception as e:
        logger.warning(f"⚠️ Agentic Controller initialization failed: {e}")
    
    # Check services
    try:
        from services.image_service import ImageService
        logger.info("✅ Image Service initialized")
    except Exception as e:
        logger.warning(f"⚠️ Image Service initialization failed: {e}")
    
    # Check data processing
    try:
        from data_processing import ImageLoader
        logger.info("✅ Data Processing module initialized")
    except Exception as e:
        logger.warning(f"⚠️ Data Processing initialization failed: {e}")
    
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown"""
    logger.info("=" * 60)
    logger.info("🛑 Shutting down SatQuery AI Backend Server")
    logger.info(f"📅 Shutdown at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

# ============================================
# CLI Entry Point
# ============================================

def main():
    """Main entry point for running the server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=[str(Path(__file__).parent)]
    )

if __name__ == "__main__":
    main()