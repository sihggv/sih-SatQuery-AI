"""
Training and Fine-tuning Module for SatQuery AI

This module provides comprehensive training and fine-tuning capabilities:
1. Model Training - Train models from scratch
2. Fine-tuning - Fine-tune pre-trained models on remote sensing data
3. Dataset Management - Load and manage datasets (BigEarthNet, RSVQA, CDVQA)
4. Training Pipeline - Complete training pipeline with validation
5. Evaluation - Model evaluation and metrics
6. Checkpoint Management - Save and load model checkpoints

Features:
- Support for multiple datasets (BigEarthNet, RSVQA, CDVQA, VRSBench)
- Domain adaptation for remote sensing
- Transfer learning from pre-trained models
- Comprehensive evaluation metrics
- Checkpoint management
- GPU/CPU support
"""

import logging
from typing import Dict, Any, Optional, List, Union

# Setup logging
logger = logging.getLogger(__name__)

# ============================================
# Import Training Modules
# ============================================

try:
    from .config import (
        TrainingConfig,
        ModelConfig,
        DataConfig,
        BigEarthNetConfig,
        VQADatasetConfig,
        ChangeDetectionConfig,
        DEFAULT_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_DATA,
        get_config
    )
    logger.info("✅ Config modules imported")
except ImportError as e:
    logger.warning(f"⚠️ Config modules import failed: {e}")
    TrainingConfig = None
    ModelConfig = None
    DataConfig = None
    BigEarthNetConfig = None
    VQADatasetConfig = None
    ChangeDetectionConfig = None
    DEFAULT_CONFIG = None
    DEFAULT_MODEL = None
    DEFAULT_DATA = None
    get_config = None

try:
    from .dataset import (
        RemoteSensingDataset,
        BigEarthNetDataset,
        VQADataset,
        ChangeDetectionDataset,
        get_dataloader
    )
    logger.info("✅ Dataset modules imported")
except ImportError as e:
    logger.warning(f"⚠️ Dataset modules import failed: {e}")
    RemoteSensingDataset = None
    BigEarthNetDataset = None
    VQADataset = None
    ChangeDetectionDataset = None
    get_dataloader = None

try:
    from .trainer import Trainer
    logger.info("✅ Trainer module imported")
except ImportError as e:
    logger.warning(f"⚠️ Trainer module import failed: {e}")
    Trainer = None

try:
    from .evaluation import Evaluator
    logger.info("✅ Evaluator module imported")
except ImportError as e:
    logger.warning(f"⚠️ Evaluator module import failed: {e}")
    Evaluator = None

try:
    from .utils import (
        set_seed,
        get_device,
        save_checkpoint,
        load_checkpoint,
        get_optimizer,
        get_scheduler,
        get_loss_function
    )
    logger.info("✅ Utils modules imported")
except ImportError as e:
    logger.warning(f"⚠️ Utils modules import failed: {e}")
    set_seed = None
    get_device = None
    save_checkpoint = None
    load_checkpoint = None
    get_optimizer = None
    get_scheduler = None
    get_loss_function = None

# ============================================
# Dataset Loaders
# ============================================

try:
    from .dataset_loaders import (
        BaseDatasetLoader,
        BigEarthNetLoader,
        RSVQALoader,
        VRSBenchLoader,
        CDVQALoader,
        FusionDatasetLoader,
        DatasetFactory
    )
    logger.info("✅ Dataset loaders imported")
except ImportError as e:
    logger.warning(f"⚠️ Dataset loaders import failed: {e}")
    BaseDatasetLoader = None
    BigEarthNetLoader = None
    RSVQALoader = None
    VRSBenchLoader = None
    CDVQALoader = None
    FusionDatasetLoader = None
    DatasetFactory = None

# ============================================
# Training Registry
# ============================================

class TrainingRegistry:
    """
    Registry for all training components
    """
    
    _components = {
        'config': {
            'name': 'Config',
            'description': 'Training configuration',
            'classes': ['TrainingConfig', 'ModelConfig', 'DataConfig']
        },
        'dataset': {
            'name': 'Dataset',
            'description': 'Dataset loading and management',
            'classes': ['RemoteSensingDataset', 'BigEarthNetDataset', 'VQADataset', 'ChangeDetectionDataset']
        },
        'trainer': {
            'name': 'Trainer',
            'description': 'Training loop and optimization',
            'classes': ['Trainer']
        },
        'evaluation': {
            'name': 'Evaluation',
            'description': 'Model evaluation and metrics',
            'classes': ['Evaluator']
        },
        'utils': {
            'name': 'Utils',
            'description': 'Training utilities',
            'classes': ['set_seed', 'get_device', 'save_checkpoint', 'load_checkpoint']
        },
        'dataset_loaders': {
            'name': 'Dataset Loaders',
            'description': 'Dataset loaders for remote sensing datasets',
            'classes': ['BigEarthNetLoader', 'RSVQALoader', 'VRSBenchLoader', 'CDVQALoader', 'DatasetFactory']
        }
    }
    
    @classmethod
    def get_component_names(cls) -> List[str]:
        """Get list of available component names"""
        return list(cls._components.keys())
    
    @classmethod
    def get_component_info(cls, component_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific component"""
        return cls._components.get(component_name)
    
    @classmethod
    def get_all_components_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all components"""
        return cls._components
    
    @classmethod
    def is_available(cls, component_name: str) -> bool:
        """Check if a component is available"""
        return component_name in cls._components

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Config
    'TrainingConfig',
    'ModelConfig',
    'DataConfig',
    'BigEarthNetConfig',
    'VQADatasetConfig',
    'ChangeDetectionConfig',
    'DEFAULT_CONFIG',
    'DEFAULT_MODEL',
    'DEFAULT_DATA',
    'get_config',
    
    # Dataset
    'RemoteSensingDataset',
    'BigEarthNetDataset',
    'VQADataset',
    'ChangeDetectionDataset',
    'get_dataloader',
    
    # Trainer
    'Trainer',
    
    # Evaluation
    'Evaluator',
    
    # Utils
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'get_optimizer',
    'get_scheduler',
    'get_loss_function',
    
    # Dataset Loaders
    'BaseDatasetLoader',
    'BigEarthNetLoader',
    'RSVQALoader',
    'VRSBenchLoader',
    'CDVQALoader',
    'FusionDatasetLoader',
    'DatasetFactory',
    
    # Registry
    'TrainingRegistry'
]

# ============================================
# Package Metadata
# ============================================

TRAINING_FEATURES = {
    'config': {
        'description': 'Training configuration management',
        'features': ['Model config', 'Data config', 'Training config']
    },
    'dataset': {
        'description': 'Dataset loading and preprocessing',
        'features': ['BigEarthNet', 'RSVQA', 'CDVQA', 'VRSBench']
    },
    'trainer': {
        'description': 'Training loop and optimization',
        'features': ['Training loop', 'Validation', 'Checkpointing']
    },
    'evaluation': {
        'description': 'Model evaluation and metrics',
        'features': ['Accuracy', 'F1', 'Precision', 'Recall']
    },
    'utils': {
        'description': 'Training utilities',
        'features': ['Seed setting', 'Device management', 'Checkpoint management']
    },
    'dataset_loaders': {
        'description': 'Dataset loaders for remote sensing',
        'features': ['BigEarthNet', 'RSVQA', 'VRSBench', 'CDVQA', 'Fusion']
    }
}

# ============================================
# Test Function
# ============================================

def test_training_package():
    """Test the training package imports"""
    print("🧪 Testing Training Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Training Registry:")
    print(f"  Available components: {TrainingRegistry.get_component_names()}")
    
    for name, info in TrainingRegistry.get_all_components_info().items():
        print(f"  ✅ {name}: {info['description']}")
        print(f"     Classes: {', '.join(info['classes'])}")
    
    # Test features
    print("\n📋 Training Features:")
    for name, features in TRAINING_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Description: {features['description']}")
        for feature in features['features']:
            print(f"     - {feature}")
    
    # Test component availability
    print("\n📋 Component Availability:")
    for name in TrainingRegistry.get_component_names():
        available = TrainingRegistry.is_available(name)
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    print("\n✅ Training Package test complete!")


if __name__ == "__main__":
    test_training_package()