"""
Dataset Factory for SatQuery AI

This module provides a factory pattern for creating dataset loaders:
1. Centralized dataset creation
2. Dynamic loader selection
3. Configuration-based initialization
4. Support for multiple datasets
5. Easy extensibility

Features:
- Factory pattern for dataset creation
- Support for all dataset loaders
- Configuration-based initialization
- Dynamic loader selection
- Error handling
- Easy extension for new datasets
"""

import logging
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path

from .base_loader import BaseDatasetLoader
from .bigearthnet_loader import BigEarthNetLoader
from .rsvqa_loader import RSVQALoader
from .vrsbench_loader import VRSBenchLoader
from .cdvqa_loader import CDVQALoader
from .fusion_loader import FusionDatasetLoader

logger = logging.getLogger(__name__)


# ============================================
# Dataset Factory Class
# ============================================

class DatasetFactory:
    """
    Factory for creating dataset loaders
    
    Features:
    - Centralized dataset creation
    - Dynamic loader selection
    - Configuration-based initialization
    - Error handling
    - Easy extension
    """
    
    # Registry of dataset loaders
    _registry: Dict[str, Type[BaseDatasetLoader]] = {
        'bigearthnet': BigEarthNetLoader,
        'rsvqa': RSVQALoader,
        'vrsbench': VRSBenchLoader,
        'cdvqa': CDVQALoader,
        'fusion': FusionDatasetLoader
    }
    
    # Dataset descriptions
    _descriptions: Dict[str, Dict[str, Any]] = {
        'bigearthnet': {
            'name': 'BigEarthNet',
            'description': 'Multi-label land cover classification dataset',
            'task': 'classification',
            'num_classes': 43,
            'data_type': 'multi-label'
        },
        'rsvqa': {
            'name': 'RSVQA',
            'description': 'Visual Question Answering on satellite images',
            'task': 'vqa',
            'question_types': ['yesno', 'number', 'other']
        },
        'vrsbench': {
            'name': 'VRSBench',
            'description': 'Captioning and grounding benchmark',
            'task': 'multi-task',
            'subtasks': ['captioning', 'grounding']
        },
        'cdvqa': {
            'name': 'CDVQA',
            'description': 'Change Detection Visual Question Answering',
            'task': 'change_vqa',
            'change_types': 11
        },
        'fusion': {
            'name': 'Fusion Dataset',
            'description': 'Optical-SAR fusion for enhanced analysis',
            'task': 'fusion',
            'modalities': ['optical', 'sar']
        }
    }
    
    @classmethod
    def create(
        cls,
        dataset_name: str,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        **kwargs
    ) -> BaseDatasetLoader:
        """
        Create a dataset loader
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Directory containing dataset
            split: 'train', 'val', 'test'
            transform: Transform for images
            target_transform: Transform for targets
            **kwargs: Additional arguments for the loader
        
        Returns:
            Dataset loader instance
        
        Raises:
            ValueError: If dataset is not supported
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._registry:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available datasets: {list(cls._registry.keys())}"
            )
        
        loader_class = cls._registry[dataset_name]
        
        try:
            loader = loader_class(
                data_dir=data_dir,
                split=split,
                transform=transform,
                target_transform=target_transform,
                **kwargs
            )
            
            logger.info(f"✅ Created dataset loader: {dataset_name} ({split})")
            return loader
            
        except Exception as e:
            logger.error(f"Failed to create dataset loader '{dataset_name}': {e}")
            raise
    
    @classmethod
    def create_from_config(
        cls,
        config: Dict[str, Any]
    ) -> BaseDatasetLoader:
        """
        Create a dataset loader from configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Dataset loader instance
        """
        dataset_name = config.get('dataset_name')
        if not dataset_name:
            raise ValueError("Configuration must contain 'dataset_name'")
        
        data_dir = config.get('data_dir', './data')
        split = config.get('split', 'train')
        transform = config.get('transform')
        target_transform = config.get('target_transform')
        
        # Extract additional kwargs
        kwargs = {k: v for k, v in config.items() 
                 if k not in ['dataset_name', 'data_dir', 'split', 'transform', 'target_transform']}
        
        return cls.create(
            dataset_name=dataset_name,
            data_dir=data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            **kwargs
        )
    
    @classmethod
    def register(cls, name: str, loader_class: Type[BaseDatasetLoader]):
        """
        Register a new dataset loader
        
        Args:
            name: Dataset name
            loader_class: Dataset loader class
        """
        cls._registry[name.lower()] = loader_class
        logger.info(f"✅ Registered dataset loader: {name}")
    
    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """
        Get list of available datasets
        
        Returns:
            List of dataset names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a dataset
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            Dataset information dictionary
        """
        return cls._descriptions.get(dataset_name.lower())
    
    @classmethod
    def get_all_dataset_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all datasets
        
        Returns:
            Dictionary mapping dataset names to their info
        """
        return cls._descriptions
    
    @classmethod
    def is_supported(cls, dataset_name: str) -> bool:
        """
        Check if a dataset is supported
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            True if supported
        """
        return dataset_name.lower() in cls._registry
    
    @classmethod
    def get_loader_class(cls, dataset_name: str) -> Optional[Type[BaseDatasetLoader]]:
        """
        Get loader class for a dataset
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            Loader class or None
        """
        return cls._registry.get(dataset_name.lower())
    
    @classmethod
    def get_default_args(cls, dataset_name: str) -> Dict[str, Any]:
        """
        Get default arguments for a dataset loader
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            Dictionary of default arguments
        """
        dataset_name = dataset_name.lower()
        
        default_args = {
            'bigearthnet': {
                'use_s1': False,
                'use_s2': True,
                'selected_bands': None,
                'patch_size': 120,
                'normalize': True,
                'labels': None
            },
            'rsvqa': {
                'question_type': 'all',
                'max_question_len': 50,
                'max_answer_len': 20,
                'use_pretrained_vocab': True
            },
            'vrsbench': {
                'task': 'captioning',
                'max_caption_len': 100,
                'num_objects': 10,
                'use_bbox': True
            },
            'cdvqa': {
                'img_size': 224,
                'include_mask': True,
                'include_questions': True
            },
            'fusion': {
                'optical_bands': None,
                'sar_bands': None,
                'img_size': 224,
                'include_land_cover': True,
                'include_features': True
            }
        }
        
        return default_args.get(dataset_name, {})


# ============================================
# Test Function
# ============================================

def test_dataset_factory():
    """Test the dataset factory"""
    print("🧪 Testing DatasetFactory...")
    print("=" * 60)
    
    # Get available datasets
    print(f"\n📋 Available datasets: {DatasetFactory.get_available_datasets()}")
    
    # Get dataset info
    print("\n📋 Dataset Information:")
    for name, info in DatasetFactory.get_all_dataset_info().items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Name: {info['name']}")
        print(f"     Description: {info['description']}")
        print(f"     Task: {info['task']}")
    
    # Test creating each dataset
    print("\n📋 Creating datasets:")
    
    # BigEarthNet
    try:
        loader = DatasetFactory.create(
            dataset_name='bigearthnet',
            data_dir='./data/BigEarthNet',
            split='train',
            use_s2=True
        )
        print(f"  ✅ BigEarthNet: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ BigEarthNet: {e}")
    
    # RSVQA
    try:
        loader = DatasetFactory.create(
            dataset_name='rsvqa',
            data_dir='./data/RSVQA',
            split='train',
            question_type='all'
        )
        print(f"  ✅ RSVQA: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ RSVQA: {e}")
    
    # VRSBench
    try:
        loader = DatasetFactory.create(
            dataset_name='vrsbench',
            data_dir='./data/VRSBench',
            split='train',
            task='captioning'
        )
        print(f"  ✅ VRSBench: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ VRSBench: {e}")
    
    # CDVQA
    try:
        loader = DatasetFactory.create(
            dataset_name='cdvqa',
            data_dir='./data/CDVQA',
            split='train',
            include_mask=True
        )
        print(f"  ✅ CDVQA: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ CDVQA: {e}")
    
    # Fusion
    try:
        loader = DatasetFactory.create(
            dataset_name='fusion',
            data_dir='./data/Fusion',
            split='train',
            include_features=True
        )
        print(f"  ✅ Fusion: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ Fusion: {e}")
    
    # Test from config
    print("\n📋 Creating from config:")
    config = {
        'dataset_name': 'bigearthnet',
        'data_dir': './data/BigEarthNet',
        'split': 'train',
        'use_s2': True,
        'patch_size': 120
    }
    
    try:
        loader = DatasetFactory.create_from_config(config)
        print(f"  ✅ Created from config: {loader.__class__.__name__}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test default args
    print("\n📋 Default arguments:")
    for dataset in DatasetFactory.get_available_datasets():
        args = DatasetFactory.get_default_args(dataset)
        print(f"  {dataset}: {list(args.keys())}")
    
    print("\n✅ Test complete!")


