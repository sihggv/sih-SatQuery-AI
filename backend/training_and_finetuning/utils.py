"""
Utilities for Training and Fine-tuning

This module provides utility functions for:
1. Device Management - GPU/CPU detection and selection
2. Seed Setting - Reproducibility
3. Checkpoint Management - Save and load checkpoints
4. Optimizer Creation - Create optimizers
5. Scheduler Creation - Create learning rate schedulers
6. Loss Function Creation - Create loss functions
7. Model Initialization - Initialize models
8. Data Augmentation - Augmentation utilities

Features:
- Comprehensive utility functions
- Reproducibility support
- Checkpoint management
- Device management
- Model initialization
"""

import os
import random
import logging
import json
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================
# Device Management
# ============================================

def get_device(device: Optional[str] = None) -> str:
    """
    Get device for training
    
    Args:
        device: Device preference ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        Device string
    """
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Get device information
    
    Returns:
        Dictionary with device information
    """
    device = get_device()
    
    info = {
        'device': device,
        'available': True
    }
    
    if device == 'cuda':
        info['name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
        info['memory_reserved'] = torch.cuda.memory_reserved(0)
        info['max_memory'] = torch.cuda.get_device_properties(0).total_memory
    elif device == 'mps':
        info['name'] = 'Apple MPS'
        info['memory_allocated'] = 0
        info['memory_reserved'] = 0
        info['max_memory'] = 0
    else:
        info['name'] = 'CPU'
        info['memory_allocated'] = 0
        info['memory_reserved'] = 0
        info['max_memory'] = 0
    
    return info


def to_device(data: Any, device: str) -> Any:
    """
    Move data to device
    
    Args:
        data: Data to move
        device: Device to move to
    
    Returns:
        Data on device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data


# ============================================
# Seed Setting
# ============================================

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"✅ Set random seed to {seed}")


def get_seed() -> int:
    """
    Get current random seed
    
    Returns:
        Current seed
    """
    return torch.initial_seed()


# ============================================
# Checkpoint Management
# ============================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    filepath: str = "checkpoint.pt",
    additional_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Training metrics
        config: Training configuration
        filepath: Path to save checkpoint
        additional_data: Additional data to save
    
    Returns:
        True if saved successfully
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'config', None),
            'metrics': metrics or {},
            'config': config or {}
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_data:
            checkpoint.update(additional_data)
        
        # Create directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"✅ Checkpoint saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        filepath: Path to checkpoint
        optimizer: Optimizer to load state
        scheduler: Scheduler to load state
        device: Device to load to
        strict: Strict loading
    
    Returns:
        Dictionary with loaded data
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        device = device or get_device()
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"✅ Checkpoint loaded from {filepath}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {}


def save_best_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    filepath: str = "best_model.pt",
    **kwargs
) -> bool:
    """
    Save best model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        config: Training configuration
        filepath: Path to save checkpoint
        **kwargs: Additional data
    
    Returns:
        True if saved successfully
    """
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        config=config,
        filepath=filepath,
        additional_data=kwargs
    )


# ============================================
# Optimizer Creation
# ============================================

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
    **kwargs
) -> Optimizer:
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        optimizer_name: Optimizer name (adam, adamw, sgd, rmsprop)
        learning_rate: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        **kwargs: Additional arguments
    
    Returns:
        Optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum, **kwargs)
    elif optimizer_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adagrad':
        return torch.optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)


# ============================================
# Scheduler Creation
# ============================================

def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 50,
    warmup_steps: int = 0,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler name (cosine, step, plateau, linear)
        epochs: Number of epochs
        warmup_steps: Number of warmup steps
        **kwargs: Additional arguments
    
    Returns:
        Scheduler
    """
    if scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6, **kwargs
        )
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, **kwargs
        )
    elif scheduler_name.lower() == 'plateau':
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor, **kwargs
        )
    elif scheduler_name.lower() == 'linear':
        return _create_linear_scheduler(optimizer, epochs, warmup_steps, **kwargs)
    elif scheduler_name.lower() == 'cosine_warmup':
        return _create_cosine_warmup_scheduler(optimizer, epochs, warmup_steps, **kwargs)
    else:
        return None


def _create_linear_scheduler(
    optimizer: Optimizer,
    epochs: int,
    warmup_steps: int,
    **kwargs
) -> _LRScheduler:
    """Create linear scheduler with warmup"""
    total_steps = epochs * 1000  # Assuming 1000 steps per epoch
    
    def lambda_func(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_func)


def _create_cosine_warmup_scheduler(
    optimizer: Optimizer,
    epochs: int,
    warmup_steps: int,
    **kwargs
) -> _LRScheduler:
    """Create cosine scheduler with warmup"""
    total_steps = epochs * 1000
    
    def lambda_func(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_func)


# ============================================
# Loss Function Creation
# ============================================

def get_loss_function(
    loss_name: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
    **kwargs
) -> nn.Module:
    """
    Create loss function
    
    Args:
        loss_name: Loss name (cross_entropy, mse, bce, l1, etc.)
        label_smoothing: Label smoothing factor
        reduction: Reduction method
        **kwargs: Additional arguments
    
    Returns:
        Loss function
    """
    if loss_name.lower() == 'cross_entropy':
        if label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction, **kwargs)
        return nn.CrossEntropyLoss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'mse':
        return nn.MSELoss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'bce':
        return nn.BCEWithLogitsLoss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'bce_loss':
        return nn.BCELoss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'l1':
        return nn.L1Loss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'smooth_l1':
        return nn.SmoothL1Loss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'kl_div':
        return nn.KLDivLoss(reduction=reduction, **kwargs)
    
    elif loss_name.lower() == 'triplet':
        margin = kwargs.get('margin', 1.0)
        return nn.TripletMarginLoss(margin=margin, reduction=reduction, **kwargs)
    
    else:
        return nn.CrossEntropyLoss(reduction=reduction, **kwargs)


# ============================================
# Model Initialization
# ============================================

def init_model_weights(model: nn.Module, method: str = 'xavier'):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model
        method: Initialization method (xavier, kaiming, normal, uniform)
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif method == 'uniform':
                nn.init.uniform_(m.weight, -0.1, 0.1)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Conv2d):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif method == 'uniform':
                nn.init.uniform_(m.weight, -0.1, 0.1)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    logger.info(f"✅ Initialized model weights with {method}")


# ============================================
# Model Utilities
# ============================================

def count_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_str(model: nn.Module) -> str:
    """
    Count number of trainable parameters as string
    
    Args:
        model: PyTorch model
    
    Returns:
        Formatted parameter count
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def get_model_summary(model: nn.Module, input_size: Optional[tuple] = None) -> str:
    """
    Get model summary
    
    Args:
        model: PyTorch model
        input_size: Input size for summary
    
    Returns:
        Model summary string
    """
    from torchsummary import summary
    
    if input_size:
        return str(summary(model, input_size))
    else:
        return str(model)


# ============================================
# Data Utilities
# ============================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader
    
    Args:
        batch: List of samples
    
    Returns:
        Collated batch
    """
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (list, tuple)):
            collated[key] = [sample[key] for sample in batch]
        elif isinstance(batch[0][key], dict):
            collated[key] = {k: [sample[key][k] for sample in batch] for k in batch[0][key]}
        else:
            collated[key] = [sample[key] for sample in batch]
    
    return collated


def normalize_tensor(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Normalize tensor
    
    Args:
        tensor: Input tensor
        mean: Mean values
        std: Standard deviation values
    
    Returns:
        Normalized tensor
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(tensor.device)
    if isinstance(std, list):
        std = torch.tensor(std).to(tensor.device)
    
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Denormalize tensor
    
    Args:
        tensor: Input tensor
        mean: Mean values
        std: Standard deviation values
    
    Returns:
        Denormalized tensor
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(tensor.device)
    if isinstance(std, list):
        std = torch.tensor(std).to(tensor.device)
    
    return tensor * std + mean


# ============================================
# Test Function
# ============================================

def test_utils():
    """Test the utils module"""
    print("🧪 Testing Utils...")
    print("=" * 60)
    
    # Test device
    print("\n📋 Testing device:")
    device = get_device()
    info = get_device_info()
    print(f"  Device: {device}")
    print(f"  Info: {info}")
    
    # Test seed
    print("\n📋 Testing seed:")
    set_seed(42)
    seed = get_seed()
    print(f"  Seed: {seed}")
    
    # Test checkpoint
    print("\n📋 Testing checkpoint:")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        filepath = tmp.name
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics={'accuracy': 0.95},
        filepath=filepath
    )
    print(f"  Saved checkpoint to {filepath}")
    
    # Load checkpoint
    loaded = load_checkpoint(model, filepath, optimizer)
    print(f"  Loaded checkpoint (epoch: {loaded.get('epoch', 'N/A')})")
    
    # Test optimizer creation
    print("\n📋 Testing optimizer:")
    opt = get_optimizer(model, 'adamw', lr=1e-4)
    print(f"  Optimizer: {type(opt).__name__}")
    
    # Test loss function
    print("\n📋 Testing loss:")
    loss_fn = get_loss_function('cross_entropy')
    print(f"  Loss function: {type(loss_fn).__name__}")
    
    # Test parameter counting
    print("\n📋 Testing parameter counting:")
    num_params = count_parameters(model)
    num_params_str = count_parameters_str(model)
    print(f"  Parameters: {num_params} ({num_params_str})")
    
    os.unlink(filepath)
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_utils()