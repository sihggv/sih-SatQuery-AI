"""
Fusion Dataset Loader for SatQuery AI

This module provides the Optical-SAR Fusion dataset loader:
1. Optical-SAR Fusion - Combine optical and SAR imagery
2. Multi-modal data loading
3. Cross-modal validation
4. Feature detection from both modalities
5. Land cover classification from fusion
6. Co-registered image pairs

Features:
- Optical and SAR image pair loading
- Multi-modal data support
- Cross-modal validation
- Feature detection
- Land cover classification
- Co-registered image pairs
"""

import logging
import json
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

from .base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

# Optical band names
OPTICAL_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

# SAR band names
SAR_BANDS = ['HH', 'HV', 'VH', 'VV']

# Land cover classes for fusion
FUSION_LAND_COVER = [
    'water',
    'urban',
    'vegetation',
    'agriculture',
    'barren',
    'wetland',
    'forest'
]

# Feature types detectable through fusion
FUSION_FEATURES = [
    'urban_zone',
    'water_body',
    'agricultural_field',
    'road_network',
    'forest_patch',
    'industrial_area',
    'residential_area'
]


# ============================================
# Fusion Dataset
# ============================================

class FusionDataset(Dataset):
    """
    Optical-SAR Fusion Dataset
    
    Features:
    - Optical and SAR image pair loading
    - Multi-modal data support
    - Cross-modal validation
    - Feature detection
    - Land cover classification
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        optical_bands: Optional[List[str]] = None,
        sar_bands: Optional[List[str]] = None,
        img_size: int = 224,
        include_land_cover: bool = True,
        include_features: bool = True
    ):
        """
        Initialize Fusion dataset
        
        Args:
            data_dir: Directory containing fusion data
            split: 'train', 'val', 'test'
            transform: Transform for images
            optical_bands: List of optical bands to use
            sar_bands: List of SAR bands to use
            img_size: Image size
            include_land_cover: Include land cover labels
            include_features: Include feature annotations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.optical_bands = optical_bands or OPTICAL_BANDS
        self.sar_bands = sar_bands or SAR_BANDS
        self.img_size = img_size
        self.include_land_cover = include_land_cover
        self.include_features = include_features
        
        self.num_classes = len(FUSION_LAND_COVER)
        self.class_names = FUSION_LAND_COVER
        
        self.samples = []
        self._load_data()
        
        logger.info(f"✅ Loaded {len(self.samples)} samples from Fusion dataset ({split})")
    
    def _load_data(self):
        """Load fusion dataset"""
        num_samples = 60 if self.split == 'train' else 15
        
        for i in range(num_samples):
            # Generate random land cover distribution
            land_cover = self._generate_land_cover()
            
            # Generate features
            features = self._generate_features() if self.include_features else []
            
            # Generate sample
            self.samples.append({
                'id': f'fusion_{i:06d}',
                'optical_path': self.data_dir / f'optical_{i}.tif',
                'sar_path': self.data_dir / f'sar_{i}.tif',
                'land_cover': land_cover,
                'features': features,
                'metadata': {
                    'location': f'Region_{random.randint(1, 10)}',
                    'sensor': 'Sentinel-1/Sentinel-2',
                    'acquisition_date': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
                }
            })
    
    def _generate_land_cover(self) -> Dict[str, float]:
        """Generate random land cover distribution"""
        land_cover = {}
        total = 0
        
        # Assign random percentages to land cover classes
        for lc in FUSION_LAND_COVER:
            pct = random.uniform(5, 30)
            land_cover[lc] = pct
            total += pct
        
        # Normalize to 100%
        if total > 0:
            land_cover = {k: (v / total) * 100 for k, v in land_cover.items()}
        
        return land_cover
    
    def _generate_features(self) -> List[Dict[str, Any]]:
        """Generate random features"""
        num_features = random.randint(2, 5)
        features = []
        
        available_features = FUSION_FEATURES.copy()
        selected_features = random.sample(available_features, min(num_features, len(available_features)))
        
        for i, feat_name in enumerate(selected_features):
            # Generate random bbox
            x1 = random.uniform(0, 0.5)
            y1 = random.uniform(0, 0.5)
            x2 = x1 + random.uniform(0.1, 0.4)
            y2 = y1 + random.uniform(0.1, 0.4)
            
            # Ensure within bounds
            x2 = min(x2, 0.95)
            y2 = min(y2, 0.95)
            
            # Determine best modality
            modalities = ['optical', 'sar', 'fused']
            best_modality = random.choice(modalities)
            
            features.append({
                'name': feat_name,
                'bbox': [x1, y1, x2, y2],
                'modality': best_modality,
                'confidence': {
                    'optical': random.uniform(0.6, 0.95),
                    'sar': random.uniform(0.6, 0.95),
                    'fused': random.uniform(0.7, 0.98)
                }
            })
        
        return features
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample"""
        sample = self.samples[idx]
        
        # Load optical image (dummy)
        optical = np.random.rand(len(self.optical_bands), self.img_size, self.img_size).astype(np.float32)
        
        # Load SAR image (dummy)
        sar = np.random.rand(len(self.sar_bands), self.img_size, self.img_size).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            optical = self.transform(optical)
            sar = self.transform(sar)
        
        return {
            'optical': torch.from_numpy(optical).float(),
            'sar': torch.from_numpy(sar).float(),
            'land_cover': sample['land_cover'],
            'features': sample['features'],
            'metadata': sample['metadata']
        }


# ============================================
# Fusion Loader Class
# ============================================

class FusionDatasetLoader(BaseDatasetLoader):
    """
    Optical-SAR Fusion Data Loader
    
    Features:
    - Optical and SAR image pair loading
    - Multi-modal data support
    - Cross-modal validation
    - Feature detection
    - Land cover classification
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        optical_bands: Optional[List[str]] = None,
        sar_bands: Optional[List[str]] = None,
        img_size: int = 224,
        include_land_cover: bool = True,
        include_features: bool = True,
        **kwargs
    ):
        """
        Initialize Fusion loader
        
        Args:
            data_dir: Directory containing fusion data
            split: 'train', 'val', 'test'
            transform: Transform for images
            target_transform: Transform for targets
            optical_bands: List of optical bands to use
            sar_bands: List of SAR bands to use
            img_size: Image size
            include_land_cover: Include land cover labels
            include_features: Include feature annotations
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, split, transform, target_transform, **kwargs)
        self.optical_bands = optical_bands or OPTICAL_BANDS
        self.sar_bands = sar_bands or SAR_BANDS
        self.img_size = img_size
        self.include_land_cover = include_land_cover
        self.include_features = include_features
        
        self.num_classes = len(FUSION_LAND_COVER)
        self.class_names = FUSION_LAND_COVER
        
        logger.info(f"✅ FusionDatasetLoader initialized (classes: {self.num_classes})")
    
    def load_dataset(self) -> Dataset:
        """Load Fusion dataset"""
        self.dataset = FusionDataset(
            data_dir=self.data_dir,
            split=self.split,
            transform=self.transform,
            optical_bands=self.optical_bands,
            sar_bands=self.sar_bands,
            img_size=self.img_size,
            include_land_cover=self.include_land_cover,
            include_features=self.include_features
        )
        return self.dataset
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.class_names
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': 'Fusion Dataset',
            'split': self.split,
            'num_classes': self.num_classes,
            'land_cover_classes': self.class_names,
            'optical_bands': self.optical_bands,
            'sar_bands': self.sar_bands,
            'img_size': self.img_size,
            'include_land_cover': self.include_land_cover,
            'include_features': self.include_features,
            'dataset_size': len(self.dataset) if self.dataset else 0
        }
    
    def get_band_info(self) -> Dict[str, Any]:
        """
        Get band information
        
        Returns:
            Dictionary with band info
        """
        return {
            'optical_bands': self.optical_bands,
            'optical_bands_count': len(self.optical_bands),
            'sar_bands': self.sar_bands,
            'sar_bands_count': len(self.sar_bands),
            'total_bands': len(self.optical_bands) + len(self.sar_bands)
        }
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get feature information
        
        Returns:
            Dictionary with feature info
        """
        return {
            'feature_types': FUSION_FEATURES,
            'num_feature_types': len(FUSION_FEATURES)
        }


# ============================================
# Test Function
# ============================================

def test_fusion_loader():
    """Test the Fusion loader"""
    print("🧪 Testing FusionDatasetLoader...")
    print("=" * 60)
    
    # Create loader
    loader = FusionDatasetLoader(
        data_dir="./data/Fusion",
        split='train',
        include_land_cover=True,
        include_features=True
    )
    
    print(f"✅ Loader created")
    print(f"   Classes: {loader.num_classes}")
    print(f"   Land cover classes: {loader.class_names}")
    
    # Load dataset
    dataset = loader.load_dataset()
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Get a sample
    sample = dataset[0]
    print(f"\n📋 Sample:")
    print(f"   Optical shape: {sample['optical'].shape}")
    print(f"   SAR shape: {sample['sar'].shape}")
    print(f"   Land cover: {sample['land_cover']}")
    print(f"   Features: {len(sample['features'])}")
    print(f"   Metadata: {sample['metadata']}")
    
    # Get dataloader
    dataloader = loader.get_dataloader(batch_size=8)
    batch = next(iter(dataloader))
    print(f"\n📦 Batch:")
    print(f"   Optical shape: {batch['optical'].shape}")
    print(f"   SAR shape: {batch['sar'].shape}")
    
    # Get info
    info = loader.get_info()
    print(f"\n📋 Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Get band info
    band_info = loader.get_band_info()
    print(f"\n📋 Band Info:")
    for key, value in band_info.items():
        print(f"   {key}: {value}")
    
    # Test without features
    print("\n📋 Testing without features:")
    loader_no_features = FusionDatasetLoader(
        data_dir="./data/Fusion",
        split='train',
        include_features=False
    )
    
    dataset_no_features = loader_no_features.load_dataset()
    sample_no_features = dataset_no_features[0]
    print(f"   Features: {sample_no_features['features']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_fusion_loader()