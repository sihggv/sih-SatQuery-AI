"""
Dataset Loaders Package for SatQuery AI

This package provides dataset loaders for various remote sensing datasets:
1. BigEarthNet - Multi-label land cover classification
2. RSVQA - Visual Question Answering on satellite images
3. VRSBench - Captioning and grounding benchmark
4. CDVQA - Change Detection VQA
5. Fusion - Optical-SAR fusion datasets

Features:
- Standardized dataset loading interface
- Support for multiple datasets
- Train/val/test splits
- Data preprocessing and augmentation
- Batch loading
- Multi-modal data support
"""

import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# ============================================
# Import Dataset Loaders
# ============================================

try:
    from .base_loader import BaseDatasetLoader
    logger.info("✅ BaseDatasetLoader imported")
except ImportError as e:
    logger.warning(f"⚠️ BaseDatasetLoader import failed: {e}")
    BaseDatasetLoader = None

try:
    from .bigearthnet_loader import BigEarthNetLoader
    logger.info("✅ BigEarthNetLoader imported")
except ImportError as e:
    logger.warning(f"⚠️ BigEarthNetLoader import failed: {e}")
    BigEarthNetLoader = None

try:
    from .rsvqa_loader import RSVQALoader
    logger.info("✅ RSVQALoader imported")
except ImportError as e:
    logger.warning(f"⚠️ RSVQALoader import failed: {e}")
    RSVQALoader = None

try:
    from .vrsbench_loader import VRSBenchLoader
    logger.info("✅ VRSBenchLoader imported")
except ImportError as e:
    logger.warning(f"⚠️ VRSBenchLoader import failed: {e}")
    VRSBenchLoader = None

try:
    from .cdvqa_loader import CDVQALoader
    logger.info("✅ CDVQALoader imported")
except ImportError as e:
    logger.warning(f"⚠️ CDVQALoader import failed: {e}")
    CDVQALoader = None

try:
    from .fusion_loader import FusionDatasetLoader
    logger.info("✅ FusionDatasetLoader imported")
except ImportError as e:
    logger.warning(f"⚠️ FusionDatasetLoader import failed: {e}")
    FusionDatasetLoader = None

try:
    from .dataset_factory import DatasetFactory
    logger.info("✅ DatasetFactory imported")
except ImportError as e:
    logger.warning(f"⚠️ DatasetFactory import failed: {e}")
    DatasetFactory = None

# ============================================
# Dataset Loaders Registry
# ============================================

class DatasetLoaderRegistry:
    """
    Registry for all dataset loaders
    """
    
    _loaders = {
        'bigearthnet': {
            'class': BigEarthNetLoader,
            'name': 'BigEarthNetLoader',
            'description': 'Multi-label land cover classification dataset',
            'datasets': ['BigEarthNet']
        },
        'rsvqa': {
            'class': RSVQALoader,
            'name': 'RSVQALoader',
            'description': 'Visual Question Answering on satellite images',
            'datasets': ['RSVQA']
        },
        'vrsbench': {
            'class': VRSBenchLoader,
            'name': 'VRSBenchLoader',
            'description': 'Captioning and grounding benchmark',
            'datasets': ['VRSBench']
        },
        'cdvqa': {
            'class': CDVQALoader,
            'name': 'CDVQALoader',
            'description': 'Change Detection Visual Question Answering',
            'datasets': ['CDVQA']
        },
        'fusion': {
            'class': FusionDatasetLoader,
            'name': 'FusionDatasetLoader',
            'description': 'Optical-SAR fusion datasets',
            'datasets': ['Fusion']
        }
    }
    
    @classmethod
    def get_loader_names(cls) -> List[str]:
        """Get list of available loader names"""
        return list(cls._loaders.keys())
    
    @classmethod
    def get_loader_info(cls, loader_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific loader"""
        return cls._loaders.get(loader_name)
    
    @classmethod
    def get_all_loaders_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaders"""
        return cls._loaders
    
    @classmethod
    def is_available(cls, loader_name: str) -> bool:
        """Check if a loader is available"""
        if loader_name not in cls._loaders:
            return False
        return cls._loaders[loader_name]['class'] is not None

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Base Loader
    'BaseDatasetLoader',
    
    # Dataset Loaders
    'BigEarthNetLoader',
    'RSVQALoader',
    'VRSBenchLoader',
    'CDVQALoader',
    'FusionDatasetLoader',
    
    # Factory
    'DatasetFactory',
    
    # Registry
    'DatasetLoaderRegistry'
]

# ============================================
# Package Metadata
# ============================================

DATASET_FEATURES = {
    'bigearthnet': {
        'description': 'Multi-label land cover classification',
        'features': [
            '43 land cover classes',
            'Sentinel-1 and Sentinel-2 data',
            'Multi-label classification'
        ],
        'num_classes': 43
    },
    'rsvqa': {
        'description': 'Visual Question Answering on satellite images',
        'features': [
            'Multiple question types',
            'Image-Text pairs',
            'VQA evaluation'
        ],
        'question_types': ['yesno', 'number', 'other']
    },
    'vrsbench': {
        'description': 'Captioning and grounding benchmark',
        'features': [
            'Image captioning',
            'Region grounding',
            'Multi-task evaluation'
        ]
    },
    'cdvqa': {
        'description': 'Change Detection Visual Question Answering',
        'features': [
            'Bi-temporal image pairs',
            'Change VQA',
            'Change detection'
        ]
    },
    'fusion': {
        'description': 'Optical-SAR fusion datasets',
        'features': [
            'Optical imagery',
            'SAR imagery',
            'Multi-modal fusion'
        ]
    }
}

# ============================================
# Test Function
# ============================================

def test_dataset_loaders_package():
    """Test the dataset loaders package imports"""
    print("🧪 Testing Dataset Loaders Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Dataset Loader Registry:")
    print(f"  Available loaders: {DatasetLoaderRegistry.get_loader_names()}")
    
    for name, info in DatasetLoaderRegistry.get_all_loaders_info().items():
        status = "✅" if info['class'] is not None else "❌"
        print(f"  {status} {name}: {info['description']}")
        print(f"     Datasets: {', '.join(info['datasets'])}")
    
    # Test features
    print("\n📋 Dataset Features:")
    for name, features in DATASET_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Description: {features['description']}")
        for feature in features['features']:
            print(f"     - {feature}")
    
    # Test availability
    print("\n📋 Loader Availability:")
    for name in DatasetLoaderRegistry.get_loader_names():
        available = DatasetLoaderRegistry.is_available(name)
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    print("\n✅ Dataset Loaders Package test complete!")


if __name__ == "__main__":
    test_dataset_loaders_package()