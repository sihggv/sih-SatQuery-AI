"""
Models Package for SatQuery AI

This package contains all specialist AI models used by SatQuery AI:
1. VQAModel - Visual Question Answering
2. GroundingModel - Text-guided Region Grounding
3. ChangeDetectionModel - Bi-temporal Change Detection
4. OpticalSARFusionModel - Optical-SAR Fusion
5. CaptioningModel - Scene Captioning

Each model inherits from BaseModel and implements:
- load() - Load model weights
- predict() - Run inference
- get_info() - Get model metadata
"""

import logging
from typing import Dict, Any, Optional, List

# Setup logging
logger = logging.getLogger(__name__)

# ============================================
# Import Base Model
# ============================================

from .base_model import (
    BaseModel,
    ModelInput,
    ModelOutput,
    ModelStatus
)

# ============================================
# Import Specialist Models
# ============================================

# Try importing each model with error handling
try:
    from .vqa_model import VQAModel
    logger.info("✅ VQAModel imported")
except ImportError as e:
    logger.warning(f"⚠️ VQAModel import failed: {e}")
    VQAModel = None

try:
    from .grounding_model import GroundingModel
    logger.info("✅ GroundingModel imported")
except ImportError as e:
    logger.warning(f"⚠️ GroundingModel import failed: {e}")
    GroundingModel = None

try:
    from .change_model import ChangeDetectionModel
    logger.info("✅ ChangeDetectionModel imported")
except ImportError as e:
    logger.warning(f"⚠️ ChangeDetectionModel import failed: {e}")
    ChangeDetectionModel = None

try:
    from .fusion_model import OpticalSARFusionModel
    logger.info("✅ OpticalSARFusionModel imported")
except ImportError as e:
    logger.warning(f"⚠️ OpticalSARFusionModel import failed: {e}")
    OpticalSARFusionModel = None

try:
    from .captioning_model import CaptioningModel
    logger.info("✅ CaptioningModel imported")
except ImportError as e:
    logger.warning(f"⚠️ CaptioningModel import failed: {e}")
    CaptioningModel = None

# ============================================
# Model Registry
# ============================================

class ModelRegistry:
    """
    Registry for all available models
    Provides model creation and management
    """
    
    _models = {
        'vqa': {
            'class': VQAModel,
            'name': 'SatQuery-VQA',
            'description': 'Visual Question Answering on satellite images',
            'version': '1.0.0'
        },
        'grounding': {
            'class': GroundingModel,
            'name': 'SatQuery-Grounding',
            'description': 'Text-guided region grounding with bounding boxes',
            'version': '1.0.0'
        },
        'change_detection': {
            'class': ChangeDetectionModel,
            'name': 'SatQuery-Change',
            'description': 'Bi-temporal change detection and analysis',
            'version': '1.0.0'
        },
        'optical_sar_fusion': {
            'class': OpticalSARFusionModel,
            'name': 'SatQuery-Fusion',
            'description': 'Optical-SAR fusion for enhanced analysis',
            'version': '1.0.0'
        },
        'captioning': {
            'class': CaptioningModel,
            'name': 'SatQuery-Caption',
            'description': 'Scene description and captioning',
            'version': '1.0.0'
        }
    }
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return cls._models.get(model_name)
    
    @classmethod
    def get_all_models_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all models"""
        return cls._models
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> Optional[BaseModel]:
        """
        Create an instance of a model
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Model instance or None if creation fails
        """
        model_info = cls._models.get(model_name)
        if not model_info:
            logger.error(f"Model '{model_name}' not found")
            return None
        
        model_class = model_info.get('class')
        if not model_class:
            logger.error(f"Model class for '{model_name}' not available")
            return None
        
        try:
            model = model_class(**kwargs)
            logger.info(f"✅ Created model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model '{model_name}': {e}")
            return None
    
    @classmethod
    def is_available(cls, model_name: str) -> bool:
        """Check if a model is available"""
        if model_name not in cls._models:
            return False
        return cls._models[model_name]['class'] is not None

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Base Model
    'BaseModel',
    'ModelInput',
    'ModelOutput',
    'ModelStatus',
    
    # Specialist Models
    'VQAModel',
    'GroundingModel',
    'ChangeDetectionModel',
    'OpticalSARFusionModel',
    'CaptioningModel',
    
    # Registry
    'ModelRegistry'
]

# ============================================
# Package Metadata
# ============================================

MODEL_FEATURES = {
    'vqa': {
        'input': 'Image + Text Query',
        'output': 'Text Answer',
        'confidence': '0.85+',
        'use_case': 'Answer questions about satellite images'
    },
    'grounding': {
        'input': 'Image + Text Query',
        'output': 'Bounding Boxes + Labels',
        'confidence': '0.80+',
        'use_case': 'Highlight specific objects/regions'
    },
    'change_detection': {
        'input': 'Two Images (Before/After)',
        'output': 'Change Map + Description',
        'confidence': '0.78+',
        'use_case': 'Detect changes between time periods'
    },
    'optical_sar_fusion': {
        'input': 'Optical + SAR Images',
        'output': 'Fused Analysis + Features',
        'confidence': '0.88+',
        'use_case': 'Combine optical and radar imagery'
    },
    'captioning': {
        'input': 'Image',
        'output': 'Scene Description',
        'confidence': '0.82+',
        'use_case': 'Generate descriptive captions'
    }
}

# ============================================
# Test Function
# ============================================

def test_models_package():
    """Test the models package imports"""
    print("🧪 Testing Models Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Model Registry:")
    print(f"  Available models: {ModelRegistry.get_model_names()}")
    
    for name, info in ModelRegistry.get_all_models_info().items():
        status = "✅" if info['class'] is not None else "❌"
        print(f"  {status} {name}: {info['description']}")
    
    # Test model features
    print("\n📋 Model Features:")
    for name, features in MODEL_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        for key, value in features.items():
            print(f"     {key}: {value}")
    
    # Test creating models
    print("\n📋 Creating Models:")
    for name in ModelRegistry.get_model_names()[:3]:  # Test first 3 models
        model = ModelRegistry.create_model(name)
        if model:
            print(f"  ✅ {name}: Created successfully")
        else:
            print(f"  ❌ {name}: Creation failed")
    
    print("\n✅ Models Package test complete!")


if __name__ == "__main__":
    test_models_package()