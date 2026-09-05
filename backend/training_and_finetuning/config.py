"""
Configuration for Training and Fine-tuning

This module contains all configuration classes for:
1. Model Configuration - Model architecture and hyperparameters
2. Data Configuration - Dataset paths and preprocessing
3. Training Configuration - Training loop parameters
4. Dataset-specific Configurations - BigEarthNet, RSVQA, CDVQA

Features:
- Dataclass-based configurations
- Default values for all parameters
- Task-specific configurations
- Easy to extend and modify
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import os


# ============================================
# Model Configuration
# ============================================

@dataclass
class ModelConfig:
    """
    Model configuration for training
    
    Attributes:
        name: Model name
        model_type: Type of model (vision-language, change-detection, fusion, etc.)
        backbone: Backbone architecture (resnet50, vit, etc.)
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone during training
        num_classes: Number of output classes
        hidden_size: Hidden layer size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for text
    """
    name: str = "satquery-vqa"
    model_type: str = "vision-language"
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 100
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 128
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'backbone': self.backbone,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**data)


# ============================================
# Data Configuration
# ============================================

@dataclass
class DataConfig:
    """
    Data configuration for training
    
    Attributes:
        dataset_name: Name of the dataset
        data_dir: Directory containing dataset
        image_size: Image size for preprocessing
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        val_split: Validation split ratio
        test_split: Test split ratio
        shuffle: Whether to shuffle data
        bands: Band names to use
        band_indices: Band indices to use
    """
    dataset_name: str = "BigEarthNet"
    data_dir: str = "./data"
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    bands: List[str] = field(default_factory=lambda: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 11, 12])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dataset_name': self.dataset_name,
            'data_dir': self.data_dir,
            'image_size': self.image_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'shuffle': self.shuffle,
            'bands': self.bands,
            'band_indices': self.band_indices
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary"""
        return cls(**data)


# ============================================
# Training Configuration
# ============================================

@dataclass
class TrainingConfig:
    """
    Training configuration
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        optimizer: Optimizer type (adam, adamw, sgd)
        scheduler: Scheduler type (cosine, step, plateau)
        loss_fn: Loss function (cross_entropy, mse, bce)
        dropout: Dropout rate
        label_smoothing: Label smoothing factor
        use_amp: Whether to use mixed precision
        log_interval: Logging interval
        eval_interval: Evaluation interval
        save_interval: Checkpoint save interval
        output_dir: Output directory
        checkpoint_dir: Checkpoint directory
        log_dir: Log directory
        seed: Random seed
        device: Device to use (auto, cpu, cuda)
    """
    # Basic training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    loss_fn: str = "cross_entropy"
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Other
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'loss_fn': self.loss_fn,
            'dropout': self.dropout,
            'label_smoothing': self.label_smoothing,
            'use_amp': self.use_amp,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'save_interval': self.save_interval,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'seed': self.seed,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**data)


# ============================================
# Dataset-specific Configurations
# ============================================

@dataclass
class BigEarthNetConfig:
    """
    BigEarthNet dataset configuration
    
    Attributes:
        data_dir: Directory containing BigEarthNet data
        labels: List of labels to use
        num_classes: Number of classes
        use_s1: Whether to use Sentinel-1 data
        use_s2: Whether to use Sentinel-2 data
        img_size: Image size
        patch_size: Patch size
        num_patches: Number of patches
    """
    data_dir: str = "./data/BigEarthNet"
    labels: List[str] = field(default_factory=lambda: [
        'Agricultural', 'Forest', 'Water', 'Urban', 'Barren',
        'Grassland', 'Wetland', 'Shrubland'
    ])
    num_classes: int = 43
    use_s1: bool = False
    use_s2: bool = True
    img_size: int = 120
    patch_size: int = 12
    num_patches: int = 100


@dataclass
class VQADatasetConfig:
    """
    VQA dataset configuration
    
    Attributes:
        vqa_dataset: VQA dataset name
        data_dir: Directory containing VQA data
        question_type: Question type (all, yesno, number, other)
        max_question_len: Maximum question length
        max_answer_len: Maximum answer length
        answer_vocab_size: Answer vocabulary size
    """
    vqa_dataset: str = "RSVQA"
    data_dir: str = "./data/RSVQA"
    question_type: str = "all"
    max_question_len: int = 50
    max_answer_len: int = 20
    answer_vocab_size: int = 1000


@dataclass
class VRSBenchConfig:
    """
    VRSBench dataset configuration
    
    Attributes:
        data_dir: Directory containing VRSBench data
        task: Task type (captioning, grounding)
        max_caption_len: Maximum caption length
        num_objects: Number of objects
    """
    data_dir: str = "./data/VRSBench"
    task: str = "captioning"
    max_caption_len: int = 100
    num_objects: int = 10


@dataclass
class CDVQAConfig:
    """
    CDVQA dataset configuration
    
    Attributes:
        data_dir: Directory containing CDVQA data
        temporal_window: Temporal window in days
        min_change_area: Minimum change area
        change_types: List of change types
    """
    data_dir: str = "./data/CDVQA"
    temporal_window: int = 365
    min_change_area: float = 100.0
    change_types: List[str] = field(default_factory=lambda: [
        'urban_expansion', 'agricultural_loss', 'forest_loss', 'water_gain'
    ])


@dataclass
class FusionConfig:
    """
    Optical-SAR Fusion dataset configuration
    
    Attributes:
        data_dir: Directory containing fusion data
        optical_bands: Optical bands to use
        sar_bands: SAR bands to use
        fusion_method: Fusion method (concat, weighted, attention)
    """
    data_dir: str = "./data/Fusion"
    optical_bands: List[str] = field(default_factory=lambda: ['B02', 'B03', 'B04', 'B08'])
    sar_bands: List[str] = field(default_factory=lambda: ['HH', 'HV'])
    fusion_method: str = "concat"


# ============================================
# Default Configurations
# ============================================

DEFAULT_CONFIG = TrainingConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_DATA = DataConfig()

VQA_CONFIG = VQADatasetConfig()
VRSBENCH_CONFIG = VRSBenchConfig()
CHANGE_CONFIG = CDVQAConfig()
FUSION_CONFIG = FusionConfig()
BIGEARTHNET_CONFIG = BigEarthNetConfig()


# ============================================
# Configuration Factory
# ============================================

def get_config(task: str) -> Dict[str, Any]:
    """
    Get configuration for specific task
    
    Args:
        task: Task name (vqa, captioning, grounding, change_detection, fusion)
    
    Returns:
        Dictionary with configuration
    """
    configs = {
        'vqa': {
            'model': ModelConfig(name='satquery-vqa', model_type='vision-language'),
            'data': DataConfig(dataset_name='RSVQA', data_dir='./data/RSVQA'),
            'training': TrainingConfig(epochs=30, batch_size=16)
        },
        'captioning': {
            'model': ModelConfig(name='satquery-caption', model_type='vision-language'),
            'data': DataConfig(dataset_name='VRSBench', data_dir='./data/VRSBench'),
            'training': TrainingConfig(epochs=30, batch_size=16)
        },
        'grounding': {
            'model': ModelConfig(name='satquery-grounding', model_type='vision-language'),
            'data': DataConfig(dataset_name='VRSBench', data_dir='./data/VRSBench'),
            'training': TrainingConfig(epochs=40, batch_size=12)
        },
        'change_detection': {
            'model': ModelConfig(name='satquery-change', model_type='change-detection'),
            'data': DataConfig(dataset_name='CDVQA', data_dir='./data/CDVQA'),
            'training': TrainingConfig(epochs=40, batch_size=8)
        },
        'fusion': {
            'model': ModelConfig(name='satquery-fusion', model_type='fusion'),
            'data': DataConfig(dataset_name='BigEarthNet', data_dir='./data/Fusion'),
            'training': TrainingConfig(epochs=50, batch_size=8)
        }
    }
    
    return configs.get(task, {})


def get_task_config(task: str) -> Dict[str, Any]:
    """
    Get full configuration for a task
    
    Args:
        task: Task name
    
    Returns:
        Full configuration dictionary
    """
    config = get_config(task)
    
    if not config:
        return {
            'model': DEFAULT_MODEL,
            'data': DEFAULT_DATA,
            'training': DEFAULT_CONFIG
        }
    
    return config


# ============================================
# Utility Functions
# ============================================

def save_config(config: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        filepath: Output path
    
    Returns:
        True if saved successfully
    """
    import json
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        filepath: Input path
    
    Returns:
        Configuration dictionary
    """
    import json
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}


def create_config_dict(
    model: Optional[ModelConfig] = None,
    data: Optional[DataConfig] = None,
    training: Optional[TrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a configuration dictionary
    
    Args:
        model: ModelConfig object
        data: DataConfig object
        training: TrainingConfig object
        **kwargs: Additional arguments
    
    Returns:
        Configuration dictionary
    """
    return {
        'model': model.to_dict() if model else DEFAULT_MODEL.to_dict(),
        'data': data.to_dict() if data else DEFAULT_DATA.to_dict(),
        'training': training.to_dict() if training else DEFAULT_CONFIG.to_dict(),
        **kwargs
    }


# ============================================
# Test Function
# ============================================

def test_config():
    """Test the configuration module"""
    print("🧪 Testing Config...")
    print("=" * 60)
    
    # Test default configs
    print("\n📋 Default Configurations:")
    print(f"  Model: {DEFAULT_MODEL.name} v{DEFAULT_MODEL.hidden_size}")
    print(f"  Data: {DEFAULT_DATA.dataset_name} (size: {DEFAULT_DATA.image_size})")
    print(f"  Training: {DEFAULT_CONFIG.epochs} epochs, {DEFAULT_CONFIG.batch_size} batch")
    
    # Test task configs
    print("\n📋 Task Configurations:")
    tasks = ['vqa', 'captioning', 'grounding', 'change_detection', 'fusion']
    for task in tasks:
        config = get_config(task)
        if config:
            print(f"  ✅ {task}: {config.get('model', {}).get('name', 'N/A')}")
        else:
            print(f"  ❌ {task}: Not found")
    
    # Test VQA config
    print("\n📋 VQA Dataset Config:")
    print(f"  Dataset: {VQA_CONFIG.vqa_dataset}")
    print(f"  Question type: {VQA_CONFIG.question_type}")
    print(f"  Max question length: {VQA_CONFIG.max_question_len}")
    
    # Test CDVQA config
    print("\n📋 CDVQA Config:")
    print(f"  Temporal window: {CHANGE_CONFIG.temporal_window} days")
    print(f"  Min change area: {CHANGE_CONFIG.min_change_area}")
    print(f"  Change types: {CHANGE_CONFIG.change_types}")
    
    # Test config dictionary creation
    print("\n📋 Config Dictionary:")
    config_dict = create_config_dict(
        model=ModelConfig(name='test_model', num_classes=50),
        data=DataConfig(dataset_name='TestDataset'),
        training=TrainingConfig(epochs=10)
    )
    print(f"  Model: {config_dict['model']['name']}")
    print(f"  Data: {config_dict['data']['dataset_name']}")
    print(f"  Training: {config_dict['training']['epochs']} epochs")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_config()