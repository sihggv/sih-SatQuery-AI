"""
Model Registry for SatQuery AI

This module manages all AI models in the system:
1. Model Registration - Register new models
2. Model Discovery - Find models by name, type, task
3. Model Lifecycle - Load, unload, update models
4. Model Versioning - Track model versions
5. Model Metadata - Store and retrieve model information

Features:
- Centralized model management
- Version tracking
- Status monitoring
- Model metadata storage
- Discovery by task/type
- Health checking
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib

logger = logging.getLogger(__name__)


# ============================================
# Enums and Data Classes
# ============================================

class ModelStatus(str, Enum):
    """Status of a model"""
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADED = "unloaded"
    DEPRECATED = "deprecated"


class ModelType(str, Enum):
    """Type of model"""
    VISION_LANGUAGE = "vision-language"
    CHANGE_DETECTION = "change-detection"
    FUSION = "fusion"
    GROUNDING = "grounding"
    CAPTIONING = "captioning"
    VQA = "vqa"
    UNKNOWN = "unknown"


class ModelTask(str, Enum):
    """Task the model performs"""
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    CAPTIONING = "captioning"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"


@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    description: str
    model_type: ModelType
    task: ModelTask
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.REGISTERED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    is_loaded: bool = False


# ============================================
# Model Registry Class
# ============================================

class ModelRegistry:
    """
    Central registry for all AI models
    
    Features:
    1. Register models with metadata
    2. Load/unload models
    3. Discover models by name, type, task
    4. Track model versions
    5. Model health checking
    6. Model lifecycle management
    """
    
    def __init__(self):
        """Initialize the model registry"""
        self._models: Dict[str, ModelInfo] = {}
        self._model_classes: Dict[str, type] = {}
        self._loaded_models: List[str] = []
        self._model_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("✅ ModelRegistry initialized")
    
    # ============================================
    # Registration Methods
    # ============================================
    
    def register_model(
        self,
        name: str,
        description: str,
        model_type: Union[ModelType, str],
        task: Union[ModelTask, str],
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_class: Optional[type] = None
    ) -> bool:
        """
        Register a model in the registry
        
        Args:
            name: Unique model name
            description: Human-readable description
            model_type: Type of model
            task: Task the model performs
            version: Model version
            tags: Tags for categorization
            dependencies: List of model names this model depends on
            parameters: Model parameters
            metadata: Additional metadata
            model_class: Model class (for loading)
        
        Returns:
            True if registration succeeded
        """
        # Normalize types
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                model_type = ModelType.UNKNOWN
        
        if isinstance(task, str):
            try:
                task = ModelTask(task.lower())
            except ValueError:
                task = ModelTask.VQA
        
        # Check if already registered
        if name in self._models:
            logger.warning(f"Model '{name}' already registered, updating...")
        
        # Create model info
        model_info = ModelInfo(
            name=name,
            description=description,
            model_type=model_type,
            task=task,
            version=version,
            tags=tags or [],
            dependencies=dependencies or [],
            parameters=parameters or {},
            metadata=metadata or {},
            status=ModelStatus.REGISTERED
        )
        
        # Store model info
        self._models[name] = model_info
        
        # Store model class if provided
        if model_class:
            self._model_classes[name] = model_class
        
        logger.info(f"✅ Registered model: {name} v{version} ({task.value})")
        return True
    
    def register_model_from_class(
        self,
        model_class: type,
        name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Register a model from a class
        
        Args:
            model_class: Model class
            name: Model name (uses class name if not provided)
            **kwargs: Additional arguments for register_model
        
        Returns:
            True if registration succeeded
        """
        # Get name from class
        if name is None:
            name = model_class.__name__
        
        # Get description from docstring
        description = kwargs.pop('description', model_class.__doc__ or f"{name} model")
        
        # Register model
        return self.register_model(
            name=name,
            description=description,
            model_class=model_class,
            **kwargs
        )
    
    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model
        
        Args:
            name: Model name
        
        Returns:
            True if unregistered successfully
        """
        if name not in self._models:
            logger.warning(f"Model '{name}' not found")
            return False
        
        # Unload if loaded
        if name in self._loaded_models:
            self.unload_model(name)
        
        # Remove from registry
        del self._models[name]
        if name in self._model_classes:
            del self._model_classes[name]
        
        logger.info(f"🗑️ Unregistered model: {name}")
        return True
    
    # ============================================
    # Model Loading Methods
    # ============================================
    
    def load_model(self, name: str, **kwargs) -> bool:
        """
        Load a model
        
        Args:
            name: Model name
            **kwargs: Loading arguments
        
        Returns:
            True if loaded successfully
        """
        if name not in self._models:
            logger.error(f"Model '{name}' not found")
            return False
        
        model_info = self._models[name]
        
        # Check if already loaded
        if model_info.is_loaded:
            logger.info(f"Model '{name}' already loaded")
            return True
        
        # Check dependencies
        for dep in model_info.dependencies:
            if dep not in self._models:
                logger.error(f"Dependency '{dep}' not found for model '{name}'")
                return False
            if not self._models[dep].is_loaded:
                logger.error(f"Dependency '{dep}' not loaded for model '{name}'")
                return False
        
        try:
            model_info.status = ModelStatus.LOADING
            
            # Create model instance if class available
            if name in self._model_classes:
                model_class = self._model_classes[name]
                instance = model_class(**kwargs)
                model_info.instance = instance
            else:
                # For testing, create dummy instance
                model_info.instance = DummyModel(name)
            
            model_info.is_loaded = True
            model_info.status = ModelStatus.READY
            model_info.updated_at = datetime.now().isoformat()
            
            if name not in self._loaded_models:
                self._loaded_models.append(name)
            
            # Initialize stats
            self._model_stats[name] = {
                'load_count': 0,
                'inference_count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'error_count': 0
            }
            self._model_stats[name]['load_count'] += 1
            
            logger.info(f"✅ Loaded model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}")
            model_info.status = ModelStatus.ERROR
            return False
    
    def unload_model(self, name: str) -> bool:
        """
        Unload a model
        
        Args:
            name: Model name
        
        Returns:
            True if unloaded successfully
        """
        if name not in self._models:
            return False
        
        model_info = self._models[name]
        
        if not model_info.is_loaded:
            return True
        
        try:
            model_info.instance = None
            model_info.is_loaded = False
            model_info.status = ModelStatus.UNLOADED
            model_info.updated_at = datetime.now().isoformat()
            
            if name in self._loaded_models:
                self._loaded_models.remove(name)
            
            logger.info(f"🔧 Unloaded model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model '{name}': {e}")
            return False
    
    def reload_model(self, name: str, **kwargs) -> bool:
        """
        Reload a model
        
        Args:
            name: Model name
            **kwargs: Loading arguments
        
        Returns:
            True if reloaded successfully
        """
        if not self.unload_model(name):
            return False
        return self.load_model(name, **kwargs)
    
    # ============================================
    # Retrieval Methods
    # ============================================
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """
        Get model information
        
        Args:
            name: Model name
        
        Returns:
            ModelInfo or None
        """
        return self._models.get(name)
    
    def get_model_instance(self, name: str) -> Optional[Any]:
        """
        Get model instance
        
        Args:
            name: Model name
        
        Returns:
            Model instance or None
        """
        if name in self._models:
            return self._models[name].instance
        return None
    
    def get_model_class(self, name: str) -> Optional[type]:
        """
        Get model class
        
        Args:
            name: Model name
        
        Returns:
            Model class or None
        """
        return self._model_classes.get(name)
    
    def get_all_models(self) -> List[ModelInfo]:
        """
        Get all registered models
        
        Returns:
            List of ModelInfo
        """
        return list(self._models.values())
    
    def get_loaded_models(self) -> List[str]:
        """
        Get loaded model names
        
        Returns:
            List of loaded model names
        """
        return self._loaded_models.copy()
    
    def get_models_by_type(self, model_type: Union[ModelType, str]) -> List[ModelInfo]:
        """
        Get models by type
        
        Args:
            model_type: Model type
        
        Returns:
            List of ModelInfo
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                return []
        
        return [m for m in self._models.values() if m.model_type == model_type]
    
    def get_models_by_task(self, task: Union[ModelTask, str]) -> List[ModelInfo]:
        """
        Get models by task
        
        Args:
            task: Task
        
        Returns:
            List of ModelInfo
        """
        if isinstance(task, str):
            try:
                task = ModelTask(task.lower())
            except ValueError:
                return []
        
        return [m for m in self._models.values() if m.task == task]
    
    def get_models_by_tag(self, tag: str) -> List[ModelInfo]:
        """
        Get models by tag
        
        Args:
            tag: Tag to search
        
        Returns:
            List of ModelInfo
        """
        return [m for m in self._models.values() if tag in m.tags]
    
    def get_ready_models(self) -> List[ModelInfo]:
        """
        Get ready models
        
        Returns:
            List of ready ModelInfo
        """
        return [m for m in self._models.values() if m.status == ModelStatus.READY]
    
    # ============================================
    # Model Execution Methods
    # ============================================
    
    async def execute_model(
        self,
        name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a model
        
        Args:
            name: Model name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Execution result
        """
        import time
        
        if name not in self._models:
            return {'error': f"Model '{name}' not found"}
        
        model_info = self._models[name]
        
        if not model_info.is_loaded:
            if not self.load_model(name):
                return {'error': f"Failed to load model '{name}'"}
        
        try:
            model_info.status = ModelStatus.BUSY
            start_time = time.time()
            
            instance = model_info.instance
            if instance and hasattr(instance, 'predict'):
                result = await instance.predict(*args, **kwargs)
            elif instance and callable(instance):
                result = instance(*args, **kwargs)
            else:
                result = {'error': f"Model '{name}' cannot be executed"}
            
            execution_time = time.time() - start_time
            
            # Update stats
            if name in self._model_stats:
                self._model_stats[name]['inference_count'] += 1
                self._model_stats[name]['total_time'] += execution_time
                self._model_stats[name]['avg_time'] = (
                    self._model_stats[name]['total_time'] / 
                    self._model_stats[name]['inference_count']
                )
            
            model_info.status = ModelStatus.READY
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'model_name': name
            }
            
        except Exception as e:
            logger.error(f"Error executing model '{name}': {e}")
            model_info.status = ModelStatus.ERROR
            if name in self._model_stats:
                self._model_stats[name]['error_count'] += 1
            return {'error': str(e)}
    
    # ============================================
    # Statistics Methods
    # ============================================
    
    def get_model_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a model
        
        Args:
            name: Model name
        
        Returns:
            Statistics dictionary
        """
        return self._model_stats.get(name, {})
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all models
        
        Returns:
            Dictionary of model statistics
        """
        return self._model_stats
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Registry statistics
        """
        total = len(self._models)
        loaded = len(self._loaded_models)
        
        ready = sum(1 for m in self._models.values() if m.status == ModelStatus.READY)
        error = sum(1 for m in self._models.values() if m.status == ModelStatus.ERROR)
        
        by_type = {}
        by_task = {}
        
        for model in self._models.values():
            by_type[model.model_type.value] = by_type.get(model.model_type.value, 0) + 1
            by_task[model.task.value] = by_task.get(model.task.value, 0) + 1
        
        return {
            'total_models': total,
            'loaded_models': loaded,
            'ready_models': ready,
            'error_models': error,
            'by_type': by_type,
            'by_task': by_task
        }
    
    # ============================================
    # Health Check Methods
    # ============================================
    
    def check_model_health(self, name: str) -> Dict[str, Any]:
        """
        Check health of a model
        
        Args:
            name: Model name
        
        Returns:
            Health check result
        """
        if name not in self._models:
            return {'healthy': False, 'error': f"Model '{name}' not found"}
        
        model_info = self._models[name]
        
        health = {
            'name': name,
            'healthy': True,
            'status': model_info.status.value,
            'is_loaded': model_info.is_loaded,
            'version': model_info.version,
            'updated_at': model_info.updated_at
        }
        
        # Check if model is responsive
        if model_info.is_loaded and model_info.instance:
            try:
                # Simple health check
                if hasattr(model_info.instance, 'is_ready'):
                    health['ready'] = model_info.instance.is_ready()
                else:
                    health['ready'] = True
            except Exception as e:
                health['healthy'] = False
                health['error'] = str(e)
        
        return health
    
    def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all models
        
        Returns:
            Dictionary of health check results
        """
        return {name: self.check_model_health(name) for name in self._models}
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def model_exists(self, name: str) -> bool:
        """
        Check if model exists
        
        Args:
            name: Model name
        
        Returns:
            True if exists
        """
        return name in self._models
    
    def is_loaded(self, name: str) -> bool:
        """
        Check if model is loaded
        
        Args:
            name: Model name
        
        Returns:
            True if loaded
        """
        if name in self._models:
            return self._models[name].is_loaded
        return False
    
    def get_model_names(self) -> List[str]:
        """
        Get all model names
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models as list of dicts
        
        Returns:
            List of model information
        """
        return [
            {
                'name': name,
                'description': m.description,
                'type': m.model_type.value,
                'task': m.task.value,
                'version': m.version,
                'status': m.status.value,
                'is_loaded': m.is_loaded
            }
            for name, m in self._models.items()
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'models': {
                name: {
                    'description': m.description,
                    'type': m.model_type.value,
                    'task': m.task.value,
                    'version': m.version,
                    'status': m.status.value,
                    'is_loaded': m.is_loaded,
                    'tags': m.tags,
                    'dependencies': m.dependencies,
                    'created_at': m.created_at,
                    'updated_at': m.updated_at
                }
                for name, m in self._models.items()
            },
            'loaded': self._loaded_models,
            'stats': self.get_registry_stats()
        }
    
    def save_registry(self, filepath: str) -> bool:
        """
        Save registry to JSON file
        
        Args:
            filepath: Output path
        
        Returns:
            True if saved successfully
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"✅ Registry saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            return False
    
    def load_registry(self, filepath: str) -> bool:
        """
        Load registry from JSON file
        
        Args:
            filepath: Input path
        
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for name, info in data.get('models', {}).items():
                self.register_model(
                    name=name,
                    description=info['description'],
                    model_type=info['type'],
                    task=info['task'],
                    version=info['version'],
                    tags=info.get('tags', []),
                    dependencies=info.get('dependencies', [])
                )
            
            logger.info(f"✅ Registry loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False


# ============================================
# Dummy Model for Testing
# ============================================

class DummyModel:
    """Dummy model for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_ready = True
    
    def predict(self, *args, **kwargs):
        return {'result': f'Dummy prediction from {self.name}'}
    
    def is_ready(self):
        return self.is_ready


# ============================================
# Test Function
# ============================================

def test_model_registry():
    """Test the model registry"""
    print("🧪 Testing ModelRegistry...")
    print("=" * 60)
    
    registry = ModelRegistry()
    
    # Register models
    print("\n📋 Registering models:")
    registry.register_model(
        name='vqa',
        description='Visual Question Answering',
        model_type='vqa',
        task='vqa',
        version='1.0.0',
        tags=['vision', 'language']
    )
    
    registry.register_model(
        name='grounding',
        description='Text-guided Region Grounding',
        model_type='grounding',
        task='grounding',
        version='1.0.0',
        tags=['vision', 'localization']
    )
    
    registry.register_model(
        name='change_detection',
        description='Bi-temporal Change Detection',
        model_type='change-detection',
        task='change_detection',
        version='1.0.0',
        tags=['temporal', 'change']
    )
    
    # Get all models
    print("\n📋 All models:")
    for model in registry.get_available_models():
        print(f"  ✅ {model['name']}: {model['description']} ({model['status']})")
    
    # Load models
    print("\n📋 Loading models:")
    for name in ['vqa', 'grounding']:
        if registry.load_model(name):
            print(f"  ✅ Loaded: {name}")
    
    # Get loaded models
    print(f"\n📋 Loaded models: {registry.get_loaded_models()}")
    
    # Get registry stats
    print("\n📋 Registry stats:")
    stats = registry.get_registry_stats()
    print(f"  Total models: {stats['total_models']}")
    print(f"  Loaded models: {stats['loaded_models']}")
    print(f"  Ready models: {stats['ready_models']}")
    
    # Check health
    print("\n📋 Health check:")
    for name in ['vqa', 'grounding', 'change_detection']:
        health = registry.check_model_health(name)
        status = "✅" if health['healthy'] else "❌"
        print(f"  {status} {name}: {health['status']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_model_registry()