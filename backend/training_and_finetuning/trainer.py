"""
Training Loop for SatQuery AI Models

This module provides a comprehensive training framework:
1. Training loop with progress tracking
2. Validation and evaluation
3. Checkpoint management
4. Learning rate scheduling
5. Mixed precision training
6. TensorBoard logging
7. Early stopping
8. Gradient accumulation

Features:
- GPU/CPU support
- Mixed precision training
- Automatic checkpointing
- Comprehensive logging
- Early stopping
- Resume from checkpoint
"""

import os
import logging
import time
import json
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainingConfig, ModelConfig, DataConfig
from .utils import set_seed, get_device, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


# ============================================
# Trainer Class
# ============================================

class Trainer:
    """
    Training class for SatQuery AI models
    
    Features:
    1. Training and validation loops
    2. Mixed precision training
    3. Checkpoint management
    4. TensorBoard logging
    5. Learning rate scheduling
    6. Early stopping
    7. Resume from checkpoint
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Union[TrainingConfig, Dict[str, Any]],
        output_dir: str = "./outputs",
        device: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        **kwargs
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            output_dir: Output directory
            device: Device to use
            model_config: Model configuration
            **kwargs: Additional arguments
        """
        # Convert config to dict if needed
        if isinstance(config, TrainingConfig):
            self.config = config.to_dict()
        else:
            self.config = config
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        
        # Set device
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.criterion = self._create_loss()
        
        # Mixed precision
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        logger.info(f"✅ Trainer initialized on {self.device}")
        logger.info(f"   Epochs: {self.config.get('epochs', 50)}")
        logger.info(f"   Batch size: {self.config.get('batch_size', 32)}")
        logger.info(f"   Learning rate: {self.config.get('learning_rate', 1e-4)}")
    
    # ============================================
    # Component Creation Methods
    # ============================================
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        # Filter parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 50)
        warmup_steps = self.config.get('warmup_steps', 0)
        
        if scheduler_name.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        elif scheduler_name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=epochs // 3, gamma=0.1
            )
        elif scheduler_name.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler_name.lower() == 'linear':
            scheduler = self._create_linear_scheduler(epochs, warmup_steps)
        else:
            scheduler = None
        
        return scheduler
    
    def _create_linear_scheduler(self, epochs: int, warmup_steps: int) -> optim.lr_scheduler.LambdaLR:
        """Create linear scheduler with warmup"""
        total_steps = epochs * len(self.train_loader)
        
        def lambda_func(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lambda_func)
    
    def _create_loss(self) -> nn.Module:
        """Create loss function"""
        loss_name = self.config.get('loss_fn', 'cross_entropy')
        label_smoothing = self.config.get('label_smoothing', 0.0)
        
        if loss_name.lower() == 'cross_entropy':
            if label_smoothing > 0:
                return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            return nn.CrossEntropyLoss()
        elif loss_name.lower() == 'mse':
            return nn.MSELoss()
        elif loss_name.lower() == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name.lower() == 'l1':
            return nn.L1Loss()
        else:
            return nn.CrossEntropyLoss()
    
    # ============================================
    # Training Methods
    # ============================================
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Epoch number
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler per step if using linear scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
            # Log
            if batch_idx % self.config.get('log_interval', 10) == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            epoch: Epoch number
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/accuracy', val_acc, epoch)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    # ============================================
    # Main Training Loop
    # ============================================
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            epochs: Number of epochs to train (uses config if None)
        
        Returns:
            Dictionary with training results
        """
        if epochs is None:
            epochs = self.config.get('epochs', 50)
        
        logger.info(f"🚀 Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Update scheduler if not per-step
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 500) == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(epochs - 1, val_metrics)
        self.save_history()
        
        total_time = time.time() - start_time
        
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'history': self.history
        }
        
        logger.info(f"✅ Training completed in {total_time:.2f}s")
        logger.info(f"   Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch + 1})")
        
        self.writer.close()
        return results
    
    # ============================================
    # Checkpoint Methods
    # ============================================
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
        
        torch.save(checkpoint, path)
        logger.debug(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            True if loaded successfully
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.history = checkpoint.get('history', self.history)
            
            logger.info(f"✅ Checkpoint loaded: {checkpoint_path}")
            logger.info(f"   Resume from epoch {self.start_epoch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"💾 History saved: {history_path}")
    
    # ============================================
    # Evaluation Methods
    # ============================================
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test DataLoader
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': 100. * correct / total,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        try:
            metrics['classification_report'] = classification_report(
                all_labels, all_preds, output_dict=True
            )
            metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
        except:
            pass
        
        return metrics


# ============================================
# Test Function
# ============================================

def test_trainer():
    """Test the trainer module"""
    print("🧪 Testing Trainer...")
    print("=" * 60)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create dummy data
    class DummyDataset:
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {'image': torch.randn(3, 224, 224), 'label': torch.randint(0, 2, (1,)).item()}
    
    from torch.utils.data import DataLoader
    
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model and trainer
    model = DummyModel()
    config = TrainingConfig(epochs=3, batch_size=16)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir="./test_outputs"
    )
    
    print(f"✅ Trainer created")
    print(f"   Device: {trainer.device}")
    print(f"   Epochs: {config.epochs}")
    
    # Run training
    results = trainer.train()
    
    print(f"\n📊 Results:")
    print(f"   Best Accuracy: {results['best_val_acc']:.2f}%")
    print(f"   Best Epoch: {results['best_epoch']}")
    print(f"   Total Time: {results['total_time']:.2f}s")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_trainer()