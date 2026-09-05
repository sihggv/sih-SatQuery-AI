"""
Utils Package for SatQuery AI Backend

This package provides utility functions and helpers used across the backend:
1. Logger - Custom logging setup
2. Helpers - Common utility functions
3. Validators - Input validation helpers
4. Converters - Data conversion utilities
5. Decorators - Custom decorators for functions
6. Exceptions - Custom exception classes

All utilities are designed to be:
- Reusable across modules
- Well-documented
- Error-handled
- Type-safe
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

# Setup logger for this package
logger = logging.getLogger(__name__)

# ============================================
# Import Utility Modules
# ============================================

# Try importing each utility with error handling
try:
    from .logger import setup_logger, get_logger, LoggerConfig
    logger.info("✅ Logger utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Logger utilities import failed: {e}")
    setup_logger = None
    get_logger = None
    LoggerConfig = None

try:
    from .helpers import (
        get_timestamp,
        ensure_dir,
        load_json,
        save_json,
        generate_id,
        format_bytes,
        get_file_extension,
        is_valid_url,
        truncate_text,
        safe_divide,
        calculate_percentage,
        chunk_list,
        flatten_dict,
        deep_merge
    )
    logger.info("✅ Helper utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Helper utilities import failed: {e}")
    get_timestamp = None
    ensure_dir = None
    load_json = None
    save_json = None
    generate_id = None
    format_bytes = None
    get_file_extension = None
    is_valid_url = None
    truncate_text = None
    safe_divide = None
    calculate_percentage = None
    chunk_list = None
    flatten_dict = None
    deep_merge = None

try:
    from .validators import (
        validate_email,
        validate_url,
        validate_date,
        validate_phone,
        validate_ip,
        validate_domain,
        is_valid_json,
        sanitize_string,
        validate_positive_number,
        validate_range
    )
    logger.info("✅ Validator utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Validator utilities import failed: {e}")
    validate_email = None
    validate_url = None
    validate_date = None
    validate_phone = None
    validate_ip = None
    validate_domain = None
    is_valid_json = None
    sanitize_string = None
    validate_positive_number = None
    validate_range = None

try:
    from .converters import (
        to_json,
        from_json,
        to_dict,
        from_dict,
        to_list,
        to_string,
        to_int,
        to_float,
        to_bool,
        parse_date,
        format_date
    )
    logger.info("✅ Converter utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Converter utilities import failed: {e}")
    to_json = None
    from_json = None
    to_dict = None
    from_dict = None
    to_list = None
    to_string = None
    to_int = None
    to_float = None
    to_bool = None
    parse_date = None
    format_date = None

try:
    from .decorators import (
        timer,
        log_execution,
        retry,
        cache,
        rate_limit,
        async_wrap,
        suppress_errors,
        validate_args,
        require_auth
    )
    logger.info("✅ Decorator utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Decorator utilities import failed: {e}")
    timer = None
    log_execution = None
    retry = None
    cache = None
    rate_limit = None
    async_wrap = None
    suppress_errors = None
    validate_args = None
    require_auth = None

try:
    from .exceptions import (
        SatQueryException,
        ValidationError,
        ImageProcessingError,
        ModelError,
        ServiceError,
        ConfigurationError,
        NotFoundError,
        UnauthorizedError,
        RateLimitError
    )
    logger.info("✅ Exception utilities imported")
except ImportError as e:
    logger.warning(f"⚠️ Exception utilities import failed: {e}")
    SatQueryException = None
    ValidationError = None
    ImageProcessingError = None
    ModelError = None
    ServiceError = None
    ConfigurationError = None
    NotFoundError = None
    UnauthorizedError = None
    RateLimitError = None

# ============================================
# Utility Registry
# ============================================

class UtilsRegistry:
    """
    Registry for all utility modules
    """
    
    _modules = {
        'logger': {
            'name': 'Logger',
            'description': 'Custom logging setup and management',
            'functions': ['setup_logger', 'get_logger']
        },
        'helpers': {
            'name': 'Helpers',
            'description': 'Common utility functions',
            'functions': ['get_timestamp', 'ensure_dir', 'load_json', 'save_json', 'generate_id']
        },
        'validators': {
            'name': 'Validators',
            'description': 'Input validation helpers',
            'functions': ['validate_email', 'validate_url', 'validate_date', 'is_valid_json']
        },
        'converters': {
            'name': 'Converters',
            'description': 'Data conversion utilities',
            'functions': ['to_json', 'from_json', 'to_dict', 'to_list']
        },
        'decorators': {
            'name': 'Decorators',
            'description': 'Custom decorators for functions',
            'functions': ['timer', 'log_execution', 'retry', 'cache']
        },
        'exceptions': {
            'name': 'Exceptions',
            'description': 'Custom exception classes',
            'functions': ['SatQueryException', 'ValidationError', 'ModelError']
        }
    }
    
    @classmethod
    def get_module_names(cls) -> List[str]:
        """Get list of available module names"""
        return list(cls._modules.keys())
    
    @classmethod
    def get_module_info(cls, module_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific module"""
        return cls._modules.get(module_name)
    
    @classmethod
    def get_all_modules_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all modules"""
        return cls._modules

# ============================================
# Package Exports
# ============================================

__all__ = [
    # Logger
    'setup_logger',
    'get_logger',
    'LoggerConfig',
    
    # Helpers
    'get_timestamp',
    'ensure_dir',
    'load_json',
    'save_json',
    'generate_id',
    'format_bytes',
    'get_file_extension',
    'is_valid_url',
    'truncate_text',
    'safe_divide',
    'calculate_percentage',
    'chunk_list',
    'flatten_dict',
    'deep_merge',
    
    # Validators
    'validate_email',
    'validate_url',
    'validate_date',
    'validate_phone',
    'validate_ip',
    'validate_domain',
    'is_valid_json',
    'sanitize_string',
    'validate_positive_number',
    'validate_range',
    
    # Converters
    'to_json',
    'from_json',
    'to_dict',
    'from_dict',
    'to_list',
    'to_string',
    'to_int',
    'to_float',
    'to_bool',
    'parse_date',
    'format_date',
    
    # Decorators
    'timer',
    'log_execution',
    'retry',
    'cache',
    'rate_limit',
    'async_wrap',
    'suppress_errors',
    'validate_args',
    'require_auth',
    
    # Exceptions
    'SatQueryException',
    'ValidationError',
    'ImageProcessingError',
    'ModelError',
    'ServiceError',
    'ConfigurationError',
    'NotFoundError',
    'UnauthorizedError',
    'RateLimitError',
    
    # Registry
    'UtilsRegistry'
]

# ============================================
# Quick Init Function
# ============================================

def init_utils(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    config_file: Optional[str] = None
):
    """
    Initialize utils package with configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path
        config_file: Configuration file path
    """
    # Setup logger
    if setup_logger:
        setup_logger(level=log_level, log_file=log_file)
    
    # Load config if provided
    if config_file and load_json:
        config = load_json(config_file)
        logger.info(f"Loaded config from {config_file}")
    
    logger.info("✅ Utils package initialized")

# ============================================
# Package Metadata
# ============================================

UTILS_FEATURES = {
    'logger': {
        'description': 'Custom logging with file and console output',
        'features': ['Log rotation', 'Multiple handlers', 'Log formatting']
    },
    'helpers': {
        'description': 'Common utility functions',
        'features': ['JSON handling', 'File operations', 'ID generation']
    },
    'validators': {
        'description': 'Input validation functions',
        'features': ['Email validation', 'URL validation', 'JSON validation']
    },
    'converters': {
        'description': 'Data conversion utilities',
        'features': ['JSON conversion', 'Type conversion', 'Date parsing']
    },
    'decorators': {
        'description': 'Function decorators',
        'features': ['Timing', 'Logging', 'Retry', 'Caching']
    },
    'exceptions': {
        'description': 'Custom exception classes',
        'features': ['Domain-specific exceptions', 'Error handling']
    }
}

# ============================================
# Test Function
# ============================================

def test_utils_package():
    """Test the utils package imports"""
    print("🧪 Testing Utils Package...")
    print("=" * 60)
    
    # Test registry
    print("\n📋 Utils Registry:")
    print(f"  Available modules: {UtilsRegistry.get_module_names()}")
    
    for name, info in UtilsRegistry.get_all_modules_info().items():
        print(f"  ✅ {name}: {info['description']}")
        print(f"     Functions: {', '.join(info['functions'])}")
    
    # Test features
    print("\n📋 Utils Features:")
    for name, features in UTILS_FEATURES.items():
        print(f"\n  ✅ {name.upper()}:")
        print(f"     Description: {features['description']}")
        for feature in features['features']:
            print(f"     - {feature}")
    
    # Test if functions are available
    print("\n📋 Function Availability:")
    test_functions = [
        ('get_timestamp', 'helpers'),
        ('validate_email', 'validators'),
        ('to_json', 'converters'),
        ('timer', 'decorators'),
        ('ValidationError', 'exceptions')
    ]
    
    for func_name, module in test_functions:
        func = globals().get(func_name)
        status = "✅" if func is not None else "❌"
        print(f"  {status} {func_name} (from {module})")
    
    print("\n✅ Utils Package test complete!")


if __name__ == "__main__":
    test_utils_package()