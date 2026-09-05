"""
Helpers Module for SatQuery AI Backend

This module provides common utility functions used across the application:
1. File Operations - JSON handling, directory management
2. ID Generation - UUID, timestamp-based IDs
3. String Manipulation - Truncation, sanitization
4. Data Operations - Chunking, flattening, merging
5. Number Operations - Safe division, percentage calculation
6. Validation - URL validation, file extension validation
7. Time Operations - Timestamp formatting

Features:
- Reusable helper functions
- Error handling
- Type hints
- Comprehensive documentation
"""

import os
import json
import uuid
import re
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
import random
import string


# ============================================
# ID Generation
# ============================================

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part
    
    Returns:
        Unique ID string
    """
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def generate_uuid() -> str:
    """
    Generate a UUID
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def generate_timestamp_id(prefix: str = "") -> str:
    """
    Generate a timestamp-based ID
    
    Args:
        prefix: Optional prefix
    
    Returns:
        Timestamp-based ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    if prefix:
        return f"{prefix}_{timestamp}"
    return timestamp


# ============================================
# Time Operations
# ============================================

def get_timestamp(format: str = "iso") -> str:
    """
    Get current timestamp
    
    Args:
        format: 'iso', 'datetime', 'unix'
    
    Returns:
        Timestamp string
    """
    now = datetime.now()
    
    if format == "iso":
        return now.isoformat()
    elif format == "datetime":
        return now.strftime('%Y-%m-%d %H:%M:%S')
    elif format == "unix":
        return str(int(now.timestamp()))
    else:
        return now.isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


# ============================================
# File Operations
# ============================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output path
        indent: Indentation level
    
    Returns:
        True if successful
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        return False


def read_file(filepath: Union[str, Path]) -> Optional[str]:
    """
    Read file content
    
    Args:
        filepath: Path to file
    
    Returns:
        File content or None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def write_file(filepath: Union[str, Path], content: str) -> bool:
    """
    Write content to file
    
    Args:
        filepath: Output path
        content: Content to write
    
    Returns:
        True if successful
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def get_file_extension(filepath: Union[str, Path]) -> str:
    """
    Get file extension
    
    Args:
        filepath: File path
    
    Returns:
        File extension (lowercase, with dot)
    """
    return Path(filepath).suffix.lower()


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Args:
        filepath: File path
    
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size


def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        size: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


# ============================================
# String Operations
# ============================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to max length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix for truncated text
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_string(text: str) -> str:
    """
    Sanitize string (remove special characters)
    
    Args:
        text: Input string
    
    Returns:
        Sanitized string
    """
    # Remove special characters
    cleaned = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')


def slugify(text: str) -> str:
    """
    Convert text to slug
    
    Args:
        text: Input text
    
    Returns:
        Slug string
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-_')


def is_valid_url(url: str) -> bool:
    """
    Validate URL
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid
    """
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return pattern.match(url) is not None


# ============================================
# Data Operations
# ============================================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# ============================================
# Number Operations
# ============================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division (handles division by zero)
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    if b == 0:
        return default
    return a / b


def calculate_percentage(value: float, total: float, default: float = 0.0) -> float:
    """
    Calculate percentage
    
    Args:
        value: Value
        total: Total
        default: Default if total is zero
    
    Returns:
        Percentage
    """
    if total == 0:
        return default
    return (value / total) * 100


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


# ============================================
# Validation
# ============================================

def validate_email(email: str) -> bool:
    """
    Validate email address
    
    Args:
        email: Email to validate
    
    Returns:
        True if valid
    """
    pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return pattern.match(email) is not None


def validate_domain(domain: str) -> bool:
    """
    Validate domain name
    
    Args:
        domain: Domain to validate
    
    Returns:
        True if valid
    """
    pattern = re.compile(
        r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    )
    return pattern.match(domain) is not None


def validate_ip(ip: str) -> bool:
    """
    Validate IP address
    
    Args:
        ip: IP address to validate
    
    Returns:
        True if valid
    """
    pattern = re.compile(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    )
    return pattern.match(ip) is not None


def validate_positive_number(value: Any) -> bool:
    """
    Validate positive number
    
    Args:
        value: Value to validate
    
    Returns:
        True if positive number
    """
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def validate_range(value: float, min_val: float, max_val: float) -> bool:
    """
    Validate value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        True if within range
    """
    return min_val <= value <= max_val


# ============================================
# Dictionary Operations
# ============================================

def get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Get value from nested dictionary
    
    Args:
        data: Dictionary
        keys: List of keys
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


def set_nested_value(data: Dict[str, Any], keys: List[str], value: Any) -> Dict[str, Any]:
    """
    Set value in nested dictionary
    
    Args:
        data: Dictionary
        keys: List of keys
        value: Value to set
    
    Returns:
        Updated dictionary
    """
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return data


# ============================================
# Misc Operations
# ============================================

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable
    
    Args:
        key: Environment variable name
        default: Default value
    
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def is_debug_mode() -> bool:
    """
    Check if in debug mode
    
    Returns:
        True if debug mode
    """
    return os.environ.get('DEBUG', 'false').lower() == 'true'


# ============================================
# Test Function
# ============================================

def test_helpers():
    """Test the helpers module"""
    print("🧪 Testing Helpers...")
    print("=" * 60)
    
    # Test ID generation
    print("\n📋 Testing ID generation:")
    print(f"  generate_id: {generate_id()}")
    print(f"  generate_id('user'): {generate_id('user')}")
    print(f"  generate_uuid: {generate_uuid()}")
    print(f"  generate_timestamp_id: {generate_timestamp_id()}")
    
    # Test time operations
    print("\n📋 Testing time operations:")
    print(f"  get_timestamp(): {get_timestamp()}")
    print(f"  format_duration(125.5): {format_duration(125.5)}")
    
    # Test string operations
    print("\n📋 Testing string operations:")
    print(f"  truncate_text('Hello World', 5): {truncate_text('Hello World', 5)}")
    print(f"  sanitize_string('Hello @World!'): {sanitize_string('Hello @World!')}")
    print(f"  slugify('Hello World!'): {slugify('Hello World!')}")
    
    # Test validation
    print("\n📋 Testing validation:")
    print(f"  validate_email('test@example.com'): {validate_email('test@example.com')}")
    print(f"  validate_domain('example.com'): {validate_domain('example.com')}")
    print(f"  validate_ip('192.168.1.1'): {validate_ip('192.168.1.1')}")
    
    # Test data operations
    print("\n📋 Testing data operations:")
    print(f"  chunk_list([1,2,3,4,5], 2): {chunk_list([1,2,3,4,5], 2)}")
    
    nested = {'a': {'b': {'c': 1}}}
    print(f"  flatten_dict({nested}): {flatten_dict(nested)}")
    print(f"  get_nested_value({nested}, ['a','b','c']): {get_nested_value(nested, ['a','b','c'])}")
    
    # Test number operations
    print("\n📋 Testing number operations:")
    print(f"  safe_divide(10, 2): {safe_divide(10, 2)}")
    print(f"  safe_divide(10, 0): {safe_divide(10, 0)}")
    print(f"  calculate_percentage(25, 100): {calculate_percentage(25, 100)}")
    print(f"  clamp(15, 0, 10): {clamp(15, 0, 10)}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_helpers()