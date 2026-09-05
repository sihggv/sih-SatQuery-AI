"""
Logger Module for SatQuery AI Backend

This module provides comprehensive logging setup:
1. Custom log formatting
2. Multiple log handlers (console, file, rotating files)
3. Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
4. Log rotation
5. JSON log formatting
6. Context-aware logging
7. Performance logging

Features:
- Consistent log format across the application
- File and console output
- Log rotation to manage file sizes
- JSON format for structured logging
- Contextual logging with extra fields
- Performance timing decorators
"""

import logging
import sys
import json
import time
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import wraps
import traceback


# ============================================
# Configuration
# ============================================

class LoggerConfig:
    """Logger configuration"""
    
    DEFAULT_LEVEL = logging.INFO
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # JSON format for structured logging
    JSON_FORMAT = {
        'timestamp': '%(asctime)s',
        'level': '%(levelname)s',
        'module': '%(module)s',
        'function': '%(funcName)s',
        'line': '%(lineno)d',
        'message': '%(message)s',
        'logger_name': '%(name)s'
    }
    
    # Log file settings
    LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    LOG_FILE_COUNT = 5
    LOG_INTERVAL = 1  # Days for timed rotation
    
    # Colors for console output (ANSI)
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }


# ============================================
# Custom Formatters
# ============================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colors for console output
    """
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        fmt = fmt or LoggerConfig.DEFAULT_FORMAT
        datefmt = datefmt or LoggerConfig.DEFAULT_DATE_FORMAT
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        levelname = record.levelname
        color = LoggerConfig.COLORS.get(levelname, LoggerConfig.COLORS['RESET'])
        reset = LoggerConfig.COLORS['RESET']
        
        # Store original levelname
        original_levelname = record.levelname
        
        # Apply color to levelname
        record.levelname = f"{color}{levelname}{reset}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    """
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'logger_name': record.name,
            'process': record.process,
            'thread': record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_tb(record.exc_info[2]))
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data['extra'] = record.extra
        
        return json.dumps(log_data)


# ============================================
# Logger Class
# ============================================

class Logger:
    """
    Custom logger with enhanced functionality
    
    Features:
    - Multiple handlers (console, file)
    - Log rotation
    - JSON formatting
    - Color output
    - Context management
    - Performance logging
    """
    
    _instances: Dict[str, 'Logger'] = {}
    
    def __new__(cls, name: str, *args, **kwargs):
        """Singleton pattern for loggers"""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(
        self,
        name: str = 'satquery',
        level: Union[str, int] = LoggerConfig.DEFAULT_LEVEL,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        format_type: str = 'default',
        enable_json: bool = False,
        enable_colors: bool = True,
        max_size: int = LoggerConfig.LOG_FILE_SIZE,
        backup_count: int = LoggerConfig.LOG_FILE_COUNT
    ):
        """
        Initialize logger
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path
            log_dir: Log directory (for rotated logs)
            format_type: 'default' or 'json'
            enable_json: Enable JSON format
            enable_colors: Enable colors in console output
            max_size: Maximum log file size
            backup_count: Number of backup files to keep
        """
        # Check if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.name = name
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.log_file = log_file
        self.log_dir = log_dir or os.getcwd()
        self.format_type = format_type
        self.enable_json = enable_json
        self.enable_colors = enable_colors
        self.max_size = max_size
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Mark as initialized
        self._initialized = True
    
    def _setup_handlers(self):
        """Setup log handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        # Formatter
        if self.enable_json:
            formatter = JSONFormatter()
        else:
            formatter = ColoredFormatter() if self.enable_colors else logging.Formatter(
                LoggerConfig.DEFAULT_FORMAT,
                LoggerConfig.DEFAULT_DATE_FORMAT
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            self._setup_file_handlers()
    
    def _setup_file_handlers(self):
        """Setup file handlers with rotation"""
        log_path = Path(self.log_file)
        log_dir = log_path.parent if log_path.parent else Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        handler.setLevel(self.level)
        
        # Formatter for file (always use default, not JSON)
        formatter = logging.Formatter(
            LoggerConfig.DEFAULT_FORMAT,
            LoggerConfig.DEFAULT_DATE_FORMAT
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    # ============================================
    # Logging Methods
    # ============================================
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(msg, *args, **kwargs)
    
    # ============================================
    # Context Methods
    # ============================================
    
    def log_with_context(self, msg: str, context: Dict[str, Any], level: str = 'info'):
        """
        Log message with additional context
        
        Args:
            msg: Log message
            context: Context dictionary
            level: Log level
        """
        extra = {'extra': context}
        getattr(self, level)(msg, extra=extra)
    
    def log_start(self, operation: str, **kwargs):
        """
        Log the start of an operation
        
        Args:
            operation: Operation name
            **kwargs: Additional context
        """
        self.info(f"Starting {operation}", extra={'extra': {'operation': operation, **kwargs}})
    
    def log_end(self, operation: str, duration: float, **kwargs):
        """
        Log the end of an operation
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional context
        """
        self.info(
            f"Completed {operation} in {duration:.3f}s",
            extra={'extra': {'operation': operation, 'duration': duration, **kwargs}}
        )
    
    def log_error(self, operation: str, error: Exception, **kwargs):
        """
        Log an error with context
        
        Args:
            operation: Operation name
            error: Exception object
            **kwargs: Additional context
        """
        self.error(
            f"Error in {operation}: {str(error)}",
            extra={'extra': {'operation': operation, 'error_type': type(error).__name__, **kwargs}}
        )


# ============================================
# Module-Level Functions
# ============================================

_logger_instances: Dict[str, Logger] = {}


def setup_logger(
    name: str = 'satquery',
    level: Union[str, int] = 'INFO',
    log_file: Optional[str] = None,
    **kwargs
) -> Logger:
    """
    Setup a logger instance
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file path
        **kwargs: Additional arguments for Logger
    
    Returns:
        Logger instance
    """
    if name not in _logger_instances:
        _logger_instances[name] = Logger(name, level, log_file, **kwargs)
    return _logger_instances[name]


def get_logger(name: str = 'satquery') -> Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name not in _logger_instances:
        return setup_logger(name)
    return _logger_instances[name]


def log_execution(logger: Optional[Logger] = None, level: str = 'info'):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger instance (uses default if None)
        level: Log level
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger()
            
            # Log start
            if level == 'debug':
                _logger.debug(f"Executing {func.__name__}")
            else:
                _logger.info(f"Executing {func.__name__}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                _logger.info(f"Completed {func.__name__} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                _logger.error(f"Failed {func.__name__} in {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


# ============================================
# Quick Setup
# ============================================

def setup_default_logger(
    level: str = 'INFO',
    log_file: Optional[str] = None
) -> Logger:
    """
    Setup default logger for the application
    
    Args:
        level: Logging level
        log_file: Log file path
    
    Returns:
        Logger instance
    """
    return setup_logger(
        name='satquery',
        level=level,
        log_file=log_file,
        enable_colors=True
    )


# ============================================
# Test Function
# ============================================

def test_logger():
    """Test the logger module"""
    print("🧪 Testing Logger...")
    print("=" * 60)
    
    # Setup logger
    logger = setup_logger(
        name='test',
        level='DEBUG',
        enable_colors=True
    )
    
    print("\n📋 Testing log levels:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test context logging
    print("\n📋 Testing context logging:")
    logger.log_with_context(
        "User logged in",
        {'user_id': '123', 'ip': '192.168.1.1'},
        'info'
    )
    
    # Test operation logging
    print("\n📋 Testing operation logging:")
    logger.log_start("Processing image", image_id='img_001')
    import time
    time.sleep(0.5)
    logger.log_end("Processing image", 0.5, image_id='img_001')
    
    # Test decorator
    print("\n📋 Testing decorator:")
    
    @log_execution(logger)
    def test_function():
        time.sleep(0.2)
        return "Done"
    
    result = test_function()
    print(f"  Result: {result}")
    
    # Test exception logging
    print("\n📋 Testing exception logging:")
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error("Test operation", e)
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_logger()