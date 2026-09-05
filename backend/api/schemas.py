from pydantic import BaseModel, Field, validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    CAPTIONING = "captioning"
    UNKNOWN = "unknown"

class ImageType(str, Enum):
    SINGLE_OPTICAL = "single_optical"
    SINGLE_SAR = "single_sar"
    CROSS_MODAL = "cross_modal"
    BI_TEMPORAL = "bi_temporal"

class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    image_type: ImageType = Field(...)
    num_images: int = Field(..., ge=1, le=4)
    task_scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    entities: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('num_images')
    def validate_num_images(cls, v, values):
        image_type = values.get('image_type')
        if image_type in [ImageType.SINGLE_OPTICAL, ImageType.SINGLE_SAR]:
            if v != 1:
                raise ValueError("Single image requires exactly 1 image")
        elif image_type in [ImageType.CROSS_MODAL, ImageType.BI_TEMPORAL]:
            if v != 2:
                raise ValueError("Multi-image requires exactly 2 images")
        return v
    
    @model_validator(mode='after')
    def validate_num_images_new(self):
        if self.image_type in [ImageType.SINGLE_OPTICAL, ImageType.SINGLE_SAR]:
            if self.num_images != 1:
                raise ValueError("Single image requires exactly 1 image")
        elif self.image_type in [ImageType.CROSS_MODAL, ImageType.BI_TEMPORAL]:
            if self.num_images != 2:
                raise ValueError("Multi-image requires exactly 2 images")
        return self

class UploadRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    image_type: ImageType = Field(...)

class AnalysisResponse(BaseModel):
    success: bool = Field(True)
    task: str = Field(...)
    answer: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    models_used: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    visual_evidence: Optional[Dict[str, Any]] = Field(None)
    change_map: Optional[Dict[str, Any]] = Field(None)
    execution_time: float = Field(0.0)
    execution_trace: Optional[Dict[str, Any]] = Field(None)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthResponse(BaseModel):
    status: str = Field("healthy")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    models_loaded: List[str] = Field(default_factory=list)
    version: str = Field("1.0.0")

class ErrorResponse(BaseModel):
    success: bool = Field(False)
    error: str = Field(...)
    error_code: Optional[str] = Field(None)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ModelListResponse(BaseModel):
    models: List[Dict[str, str]]
    loaded: List[str]
    total_models: int
    loaded_count: int

class ChangeDetectionResponse(BaseModel):
    success: bool = Field(True)
    result: Dict[str, Any]
    execution_time: float = Field(0.0)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FusionResponse(BaseModel):
    success: bool = Field(True)
    result: Dict[str, Any]
    execution_time: float = Field(0.0)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())