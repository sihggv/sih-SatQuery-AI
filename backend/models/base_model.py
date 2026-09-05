"""
Base Model Class for SatQuery AI

This module provides the abstract base class for all specialist models.
All models inherit from BaseModel and implement:
1. load() - Load model weights and resources
2. predict() - Run inference on input data
3. unload() - Unload model from memory
4. get_info() - Get model metadata

Features:
- Standardized input/output interfaces
- Automatic device detection (CPU/GPU)
- Performance tracking
- Model versioning
- Status management
"""

import logging
import time
import abc
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================
# Enums
# ============================================

class ModelStatus(str, Enum):
    """Status of a model"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADED = "unloaded"


class ModelType(str, Enum):
    """Type of model"""
    VISION_LANGUAGE = "vision-language"
    CHANGE_DETECTION = "change-detection"
    FUSION = "fusion"
    GROUNDING = "grounding"
    CAPTIONING = "captioning"
    VQA = "vqa"


# ============================================
# Data Classes
# ============================================

@dataclass
class ModelInput:
    """
    Standardized input for all models
    
    Attributes:
        query: Optional text query
        image_paths: List of image file paths
        image_type: Type of image(s)
        parameters: Additional parameters
        metadata: Optional metadata
    """
    query: Optional[str] = None
    image_paths: Optional[List[str]] = None
    image_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'image_paths': self.image_paths,
            'image_type': self.image_type,
            'parameters': self.parameters,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInput':
        """Create from dictionary"""
        return cls(
            query=data.get('query'),
            image_paths=data.get('image_paths'),
            image_type=data.get('image_type'),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ModelOutput:
    """
    Standardized output from all models
    
    Attributes:
        result: The main result (answer, description, etc.)
        confidence: Confidence score (0-1)
        models_used: List of model names used
        execution_time: Time taken in seconds
        visual_evidence: Optional visual evidence (bboxes, masks)
        change_map: Optional change map
        error: Optional error message
        metadata: Additional metadata
    """
    result: Any
    confidence: float = 0.0
    models_used: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    visual_evidence: Optional[Dict[str, Any]] = None
    change_map: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'result': self.result,
            'confidence': self.confidence,
            'models_used': self.models_used,
            'execution_time': self.execution_time,
            'visual_evidence': self.visual_evidence,
            'change_map': self.change_map,
            'error': self.error,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelOutput':
        """Create from dictionary"""
        return cls(
            result=data.get('result'),
            confidence=data.get('confidence', 0.0),
            models_used=data.get('models_used', []),
            execution_time=data.get('execution_time', 0.0),
            visual_evidence=data.get('visual_evidence'),
            change_map=data.get('change_map'),
            error=data.get('error'),
            metadata=data.get('metadata', {})
        )
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful"""
        return self.error is None
    
    @property
    def confidence_level(self) -> str:
        """Get confidence level as string"""
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "low"


# ============================================
# Base Model Class
# ============================================

class BaseModel(abc.ABC):
    """
    Abstract base class for all models
    
    All specialist models must implement:
    1. load() - Load model weights
    2. predict() - Run inference
    3. unload() - Unload model
    4. get_info() - Get model info
    
    Features:
    - Device management (CPU/GPU)
    - Performance tracking
    - Status management
    - Model versioning
    """
    
    def __init__(
        self,
        name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        device: Optional[str] = None
    ):
        """
        Initialize base model
        
        Args:
            name: Model name
            model_type: Type of model
            version: Model version
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.name = name
        self.model_type = model_type
        self.version = version
        self.status = ModelStatus.UNINITIALIZED
        self.device = device or self._get_default_device()
        
        self.is_loaded = False
        self.is_trained = False
        self.load_time = None
        self.last_inference_time = None
        self.total_inferences = 0
        
        # Performance tracking
        self._performance = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'min_inference_time': float('inf'),
            'max_inference_time': 0.0
        }
        
        # Metadata
        self.metadata = {
            'name': name,
            'type': self.model_type.value if isinstance(model_type, ModelType) else model_type,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'device': self.device
        }
        
        logger.info(f"✅ Initialized {name} v{version} ({self.model_type})")
    
    def _get_default_device(self) -> str:
        """Get default device (GPU if available)"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    # ============================================
    # Abstract Methods
    # ============================================
    
    @abc.abstractmethod
    def load(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Load model weights and resources
        
        Args:
            model_path: Path to model weights (optional)
            **kwargs: Additional loading arguments
        
        Returns:
            True if loaded successfully
        """
        pass
    
    @abc.abstractmethod
    def predict(self, model_input: ModelInput) -> ModelOutput:
        """
        Run inference on input data
        
        Args:
            model_input: Standardized input
        
        Returns:
            ModelOutput: Standardized output
        """
        pass
    
    @abc.abstractmethod
    def unload(self) -> bool:
        """
        Unload model from memory
        
        Returns:
            True if unloaded successfully
        """
        pass
    
    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        pass
    
    # ============================================
    # Optional Methods (can be overridden)
    # ============================================
    
    def prepare_input(self, model_input: ModelInput) -> Dict[str, Any]:
        """
        Prepare input for model inference
        
        Override this method for custom input preparation
        
        Args:
            model_input: Raw input
        
        Returns:
            Prepared input for model
        """
        return model_input.to_dict()
    
    def post_process(self, raw_output: Any) -> ModelOutput:
        """
        Post-process raw model output
        
        Override this method for custom post-processing
        
        Args:
            raw_output: Raw output from model
        
        Returns:
            Standardized ModelOutput
        """
        if isinstance(raw_output, ModelOutput):
            return raw_output
        
        return ModelOutput(
            result=raw_output,
            confidence=0.5,
            models_used=[self.name]
        )
    
    def validate_input(self, model_input: ModelInput) -> Tuple[bool, Optional[str]]:
        """
        Validate input before inference
        
        Args:
            model_input: Input to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if query exists
        if not model_input.query and not model_input.image_paths:
            return False, "Either query or images must be provided"
        
        # Check image paths
        if model_input.image_paths:
            for path in model_input.image_paths:
                if not Path(path).exists():
                    return False, f"Image not found: {path}"
        
        return True, None
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.status == ModelStatus.READY and self.is_loaded
    
    def is_busy(self) -> bool:
        """Check if model is currently busy"""
        return self.status == ModelStatus.BUSY
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status.value if isinstance(self.status, ModelStatus) else self.status,
            'is_loaded': self.is_loaded,
            'is_trained': self.is_trained,
            'device': self.device,
            'load_time': self.load_time,
            'total_inferences': self.total_inferences
        }
    
    def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'inference_count': self._performance['inference_count'],
            'total_inference_time': self._performance['total_inference_time'],
            'avg_inference_time': self._performance['avg_inference_time'],
            'min_inference_time': self._performance['min_inference_time'],
            'max_inference_time': self._performance['max_inference_time']
        }
    
    def _track_inference(self, execution_time: float):
        """Track inference performance"""
        self._performance['inference_count'] += 1
        self._performance['total_inference_time'] += execution_time
        self._performance['avg_inference_time'] = (
            self._performance['total_inference_time'] / 
            self._performance['inference_count']
        )
        self._performance['min_inference_time'] = min(
            self._performance['min_inference_time'], 
            execution_time
        )
        self._performance['max_inference_time'] = max(
            self._performance['max_inference_time'], 
            execution_time
        )
    
    def _measure_time(self, func, *args, **kwargs) -> tuple:
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    def _format_answer(self, answer: str, max_length: int = 500) -> str:
        """Truncate answer if too long"""
        if len(answer) > max_length:
            return answer[:max_length] + "..."
        return answer
    
    def to_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'name': self.name,
            'type': self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            'version': self.version,
            'device': self.device,
            'metadata': self.metadata
        }
    
    def save_config(self, filepath: str) -> bool:
        """Save model configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_config(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    # ============================================
    # Magic Methods
    # ============================================
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.model_type})"
    
    def __repr__(self) -> str:
        return f"<Model {self.name} v{self.version} status={self.status} loaded={self.is_loaded}>"


# ============================================
# Example Usage
# ============================================

def test_base_model():
    """Test the base model functionality"""
    print("🧪 Testing BaseModel...")
    print("=" * 60)
    
    # Create a dummy model class for testing
    class DummyModel(BaseModel):
        def __init__(self):
            super().__init__(
                name="DummyModel",
                model_type=ModelType.VQA,
                version="1.0.0"
            )
        
        def load(self, model_path: Optional[str] = None, **kwargs) -> bool:
            self.is_loaded = True
            self.status = ModelStatus.READY
            return True
        
        def predict(self, model_input: ModelInput) -> ModelOutput:
            return ModelOutput(
                result="This is a test prediction",
                confidence=0.95,
                models_used=[self.name],
                execution_time=0.1
            )
        
        def unload(self) -> bool:
            self.is_loaded = False
            self.status = ModelStatus.UNLOADED
            return True
        
        def get_info(self) -> Dict[str, Any]:
            return {
                'name': self.name,
                'type': self.model_type,
                'version': self.version,
                'status': self.status,
                'is_loaded': self.is_loaded
            }
    
    # Test the dummy model
    model = DummyModel()
    print(f"Model: {model}")
    print(f"Status: {model.get_status()}")
    
    # Load model
    model.load()
    print(f"Loaded: {model.is_loaded}")
    
    # Predict
    model_input = ModelInput(query="What is in this image?", image_paths=["test.tif"])
    output = model.predict(model_input)
    print(f"Output: {output.result}")
    print(f"Confidence: {output.confidence}")
    
    # Get info
    info = model.get_info()
    print(f"Info: {info}")
    
    # Unload
    model.unload()
    print(f"Unloaded: {not model.is_loaded}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_base_model()