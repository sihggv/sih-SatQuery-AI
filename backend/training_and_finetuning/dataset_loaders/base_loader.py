"""
Base Dataset Loader for SatQuery AI

This module provides the base class for all dataset loaders:
1. Standardized dataset loading interface
2. Train/val/test split management
3. Data preprocessing and augmentation
4. Batch loading
5. Metadata extraction
6. Dataset information

Features:
- Abstract base class for all loaders
- Standardized interface
- Split management
- Preprocessing pipeline
- Metadata extraction
- Data validation
"""

import logging
import abc
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np

logger = logging.getLogger(__name__)


# ============================================
# Base Dataset Loader Class
# ============================================

class BaseDatasetLoader(abc.ABC):
    """
    Abstract base class for dataset loaders
    
    All dataset loaders must implement:
    1. load_dataset() - Load the dataset
    2. get_class_names() - Get class names
    3. get_info() - Get dataset information
    
    Features:
    - Standardized interface
    - Split management
    - DataLoader creation
    - Data validation
    - Preprocessing support
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing dataset
            split: 'train', 'val', 'test'
            transform: Transform to apply to data
            target_transform: Transform to apply to targets
            **kwargs: Additional arguments
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        
        self.dataset = None
        self.num_classes = 0
        self.class_names = []
        self._validate_split()
        
        logger.info(f"✅ Initialized {self.__class__.__name__} for {split}")
    
    def _validate_split(self):
        """Validate split parameter"""
        valid_splits = ['train', 'val', 'test', 'all']
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {valid_splits}")
    
    @abc.abstractmethod
    def load_dataset(self) -> Dataset:
        """
        Load the dataset
        
        Returns:
            PyTorch Dataset object
        """
        pass
    
    @abc.abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Get class names
        
        Returns:
            List of class names
        """
        pass
    
    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset information
        
        Returns:
            Dictionary with dataset info
        """
        pass
    
    # ============================================
    # DataLoader Creation
    # ============================================
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Get DataLoader for the dataset
        
        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            drop_last: Drop last incomplete batch
            collate_fn: Custom collate function
        
        Returns:
            DataLoader
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        # For validation and test, set shuffle=False
        if self.split in ['val', 'test']:
            shuffle = False
        
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    
    # ============================================
    # Split Management
    # ============================================
    
    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, Dataset]:
        """
        Get train/val/test splits
        
        Args:
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            random_seed: Random seed for reproducibility
        
        Returns:
            Dictionary with splits
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        total = len(self.dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        # Set seed for reproducibility
        torch.manual_seed(random_seed)
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    # ============================================
    # Data Validation
    # ============================================
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate dataset
        
        Returns:
            Validation results
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        results = {
            'is_valid': True,
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'issues': []
        }
        
        # Check if dataset is empty
        if len(self.dataset) == 0:
            results['is_valid'] = False
            results['issues'].append('Dataset is empty')
        
        # Check if classes are defined
        if not self.class_names:
            results['is_valid'] = False
            results['issues'].append('Class names are not defined')
        
        # Check if class names match num_classes
        if self.class_names and len(self.class_names) != self.num_classes:
            results['is_valid'] = False
            results['issues'].append(
                f'Class names length ({len(self.class_names)}) does not match num_classes ({self.num_classes})'
            )
        
        return results
    
    def validate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single sample
        
        Args:
            sample: Sample to validate
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': []
        }
        
        # Check required keys
        required_keys = ['image', 'label']
        for key in required_keys:
            if key not in sample:
                results['is_valid'] = False
                results['issues'].append(f'Missing required key: {key}')
        
        # Check image shape
        if 'image' in sample:
            image = sample['image']
            if isinstance(image, torch.Tensor):
                if len(image.shape) not in [3, 4]:
                    results['is_valid'] = False
                    results['issues'].append(f'Invalid image shape: {image.shape}')
        
        return results
    
    # ============================================
    # Data Preprocessing
    # ============================================
    
    def apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to sample
        
        Args:
            sample: Sample to transform
        
        Returns:
            Transformed sample
        """
        if self.transform and 'image' in sample:
            sample['image'] = self.transform(sample['image'])
        
        if self.target_transform and 'label' in sample:
            sample['label'] = self.target_transform(sample['label'])
        
        return sample
    
    # ============================================
    # Statistics Methods
    # ============================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dataset statistics
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        stats = {
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'split': self.split,
            'class_names': self.class_names
        }
        
        # Compute class distribution if possible
        try:
            labels = []
            for i in range(min(1000, len(self.dataset))):
                sample = self.dataset[i]
                if 'label' in sample:
                    label = sample['label']
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    labels.append(label)
            
            if labels:
                unique, counts = np.unique(labels, return_counts=True)
                stats['class_distribution'] = {
                    int(u): int(c) for u, c in zip(unique, counts)
                }
        except:
            pass
        
        return stats
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def __len__(self) -> int:
        """Get dataset size"""
        if self.dataset is None:
            self.dataset = self.load_dataset()
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample"""
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        sample = self.dataset[idx]
        return self.apply_transforms(sample)
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_info()
        return f"{self.__class__.__name__}(split={self.split}, num_samples={info.get('dataset_size', 0)}, num_classes={self.num_classes})"


# ============================================
# Test Function
# ============================================

def test_base_loader():
    """Test the base dataset loader"""
    print("🧪 Testing BaseDatasetLoader...")
    print("=" * 60)
    
    # Create a concrete implementation for testing
    class DummyLoader(BaseDatasetLoader):
        def load_dataset(self) -> Dataset:
            # Create a dummy dataset
            class DummyDataset(Dataset):
                def __len__(self):
                    return 100
                
                def __getitem__(self, idx):
                    return {
                        'image': torch.randn(3, 224, 224),
                        'label': torch.tensor(idx % 10)
                    }
            
            self.dataset = DummyDataset()
            self.num_classes = 10
            self.class_names = [f'Class_{i}' for i in range(10)]
            return self.dataset
        
        def get_class_names(self) -> List[str]:
            return self.class_names
        
        def get_info(self) -> Dict[str, Any]:
            return {
                'name': 'DummyLoader',
                'split': self.split,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'dataset_size': len(self.dataset) if self.dataset else 0
            }
    
    # Create loader
    loader = DummyLoader(
        data_dir='./data/dummy',
        split='train'
    )
    
    print(f"✅ Loader created: {loader}")
    
    # Load dataset
    dataset = loader.load_dataset()
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Get dataloader
    dataloader = loader.get_dataloader(batch_size=16)
    print(f"✅ DataLoader created: {len(dataloader)} batches")
    
    # Get sample
    sample = loader[0]
    print(f"✅ Sample: image shape {sample['image'].shape}, label {sample['label']}")
    
    # Get info
    info = loader.get_info()
    print(f"✅ Info: {info}")
    
    # Validate
    validation = loader.validate_data()
    print(f"✅ Validation: {validation}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_base_loader()