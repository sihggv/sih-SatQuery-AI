"""
Tool Registry for SatQuery AI

This module manages all tools, functions, and utilities in the system:
1. Tool Registration - Register new tools
2. Tool Discovery - Find tools by name, type, tag
3. Tool Execution - Execute tools with validation
4. Tool Lifecycle - Load, unload, update tools
5. Tool Metadata - Store and retrieve tool information

Features:
- Centralized tool management
- Multiple tool types (function, class, API)
- Dependency management
- Execution tracking
- Performance monitoring
- Health checking
"""

import logging
import time
import asyncio
import inspect
import functools
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================
# Enums and Data Classes
# ============================================

class ToolType(str, Enum):
    """Types of tools"""
    FUNCTION = "function"      # Python function
    CLASS = "class"           # Python class
    API = "api"               # External API
    PIPELINE = "pipeline"     # Multi-step pipeline
    UTILITY = "utility"       # Helper utility
    MODEL = "model"           # AI/ML Model


class ToolStatus(str, Enum):
    """Status of a tool"""
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADED = "unloaded"
    DEPRECATED = "deprecated"


@dataclass
class ToolInfo:
    """Tool information and metadata"""
    name: str
    description: str
    tool_type: ToolType
    version: str = "1.0.0"
    status: ToolStatus = ToolStatus.REGISTERED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    function: Optional[Callable] = None
    is_loaded: bool = False
    is_async: bool = False


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: Optional[str] = None
    input_params: Optional[Dict[str, Any]] = None


# ============================================
# Tool Registry Class
# ============================================

class ToolRegistry:
    """
    Central registry for all tools, functions, and utilities
    
    Features:
    1. Register tools (functions, classes, APIs)
    2. Discover tools by name, type, tag
    3. Execute tools with validation
    4. Track tool usage and performance
    5. Manage tool dependencies
    6. Health checking
    """
    
    def __init__(self):
        """Initialize the tool registry"""
        self._tools: Dict[str, ToolInfo] = {}
        self._loaded_tools: List[str] = []
        self._execution_history: List[ToolExecutionResult] = []
        self._performance_stats: Dict[str, Dict[str, float]] = {}
        self._max_history: int = 1000
        
        logger.info("✅ ToolRegistry initialized")
    
    # ============================================
    # Registration Methods
    # ============================================
    
    def register_tool(
        self,
        name: str,
        description: str,
        tool_type: Union[ToolType, str],
        function: Optional[Callable] = None,
        instance: Optional[Any] = None,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a tool in the registry
        
        Args:
            name: Unique tool name
            description: Human-readable description
            tool_type: Type of tool
            function: Callable function (for functions)
            instance: Tool instance (for classes)
            version: Tool version
            tags: Tags for categorization
            dependencies: List of tool names this tool depends on
            parameters: Expected parameters schema
            returns: Return value schema
            examples: Usage examples
            metadata: Additional metadata
        
        Returns:
            True if registration succeeded
        """
        # Normalize tool type
        if isinstance(tool_type, str):
            try:
                tool_type = ToolType(tool_type.lower())
            except ValueError:
                tool_type = ToolType.UTILITY
        
        # Check if already registered
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, updating...")
        
        # Check if function is async
        is_async = False
        if function:
            is_async = asyncio.iscoroutinefunction(function)
        
        # Auto-detect parameters if not provided
        if parameters is None and function:
            parameters = self._extract_parameters(function)
        
        # Create tool info
        tool_info = ToolInfo(
            name=name,
            description=description,
            tool_type=tool_type,
            version=version,
            tags=tags or [],
            dependencies=dependencies or [],
            parameters=parameters or {},
            returns=returns or {},
            examples=examples or [],
            metadata=metadata or {},
            function=function,
            instance=instance,
            is_async=is_async,
            status=ToolStatus.REGISTERED
        )
        
        # Store tool info
        self._tools[name] = tool_info
        
        logger.info(f"✅ Registered tool: {name} ({tool_type.value})")
        return True
    
    def register_function(
        self,
        name: str,
        description: str,
        function: Callable,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to register a function
        
        Args:
            name: Tool name
            description: Tool description
            function: Python function
            version: Tool version
            tags: Tags for categorization
            parameters: Expected parameters
            returns: Return value schema
            examples: Usage examples
            metadata: Additional metadata
        
        Returns:
            True if registration succeeded
        """
        return self.register_tool(
            name=name,
            description=description,
            tool_type=ToolType.FUNCTION,
            function=function,
            version=version,
            tags=tags,
            parameters=parameters,
            returns=returns,
            examples=examples,
            metadata=metadata
        )
    
    def register_class(
        self,
        name: str,
        description: str,
        class_obj: Type,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to register a class
        
        Args:
            name: Tool name
            description: Tool description
            class_obj: Python class
            version: Tool version
            tags: Tags for categorization
            parameters: Constructor parameters
            returns: Return value schema
            examples: Usage examples
            metadata: Additional metadata
        
        Returns:
            True if registration succeeded
        """
        return self.register_tool(
            name=name,
            description=description,
            tool_type=ToolType.CLASS,
            instance=class_obj,
            version=version,
            tags=tags,
            parameters=parameters,
            returns=returns,
            examples=examples,
            metadata=metadata
        )
    
    def register_api(
        self,
        name: str,
        description: str,
        api_url: str,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to register an external API
        
        Args:
            name: Tool name
            description: Tool description
            api_url: API endpoint URL
            version: Tool version
            tags: Tags for categorization
            parameters: API parameters
            returns: Response schema
            examples: Usage examples
            metadata: Additional metadata
        
        Returns:
            True if registration succeeded
        """
        # Create an async function for API call
        async def call_api(**kwargs):
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=kwargs) as response:
                    return await response.json()
        
        return self.register_tool(
            name=name,
            description=description,
            tool_type=ToolType.API,
            function=call_api,
            version=version,
            tags=tags,
            parameters=parameters,
            returns=returns,
            examples=examples,
            metadata={**metadata, 'api_url': api_url}
        )
    
    def _extract_parameters(self, function: Callable) -> Dict[str, Any]:
        """Extract parameter information from function signature"""
        signature = inspect.signature(function)
        params = {}
        
        for name, param in signature.parameters.items():
            if name in ['self', 'cls']:
                continue
            
            param_info = {
                'name': name,
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'required': param.default == inspect.Parameter.empty,
                'default': None if param.default == inspect.Parameter.empty else param.default
            }
            params[name] = param_info
        
        return params
    
    # ============================================
    # Tool Loading Methods
    # ============================================
    
    def load_tool(self, name: str) -> bool:
        """
        Load a tool
        
        Args:
            name: Tool name
        
        Returns:
            True if loaded successfully
        """
        if name not in self._tools:
            logger.error(f"Tool '{name}' not found")
            return False
        
        tool_info = self._tools[name]
        
        # Check if already loaded
        if tool_info.is_loaded:
            return True
        
        # Check dependencies
        for dep in tool_info.dependencies:
            if dep not in self._tools:
                logger.error(f"Dependency '{dep}' not found for tool '{name}'")
                return False
            if not self._tools[dep].is_loaded:
                logger.error(f"Dependency '{dep}' not loaded for tool '{name}'")
                return False
        
        try:
            tool_info.status = ToolStatus.LOADING
            
            # If class, create instance
            if tool_info.tool_type == ToolType.CLASS and tool_info.instance:
                if isinstance(tool_info.instance, type):
                    tool_info.instance = tool_info.instance()
            
            tool_info.is_loaded = True
            tool_info.status = ToolStatus.READY
            tool_info.updated_at = datetime.now().isoformat()
            
            if name not in self._loaded_tools:
                self._loaded_tools.append(name)
            
            # Initialize performance stats
            self._performance_stats[name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
            
            logger.info(f"✅ Loaded tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
            tool_info.status = ToolStatus.ERROR
            return False
    
    def unload_tool(self, name: str) -> bool:
        """
        Unload a tool
        
        Args:
            name: Tool name
        
        Returns:
            True if unloaded successfully
        """
        if name not in self._tools:
            return False
        
        tool_info = self._tools[name]
        
        if not tool_info.is_loaded:
            return True
        
        try:
            tool_info.is_loaded = False
            tool_info.status = ToolStatus.UNLOADED
            tool_info.updated_at = datetime.now().isoformat()
            
            if name in self._loaded_tools:
                self._loaded_tools.remove(name)
            
            logger.info(f"🔧 Unloaded tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload tool '{name}': {e}")
            return False
    
    def reload_tool(self, name: str) -> bool:
        """
        Reload a tool
        
        Args:
            name: Tool name
        
        Returns:
            True if reloaded successfully
        """
        if not self.unload_tool(name):
            return False
        return self.load_tool(name)
    
    # ============================================
    # Tool Execution Methods
    # ============================================
    
    async def execute_tool(
        self,
        name: str,
        *args,
        **kwargs
    ) -> ToolExecutionResult:
        """
        Execute a tool
        
        Args:
            name: Tool name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            ToolExecutionResult
        """
        start_time = time.time()
        
        # Check if tool exists
        if name not in self._tools:
            return ToolExecutionResult(
                success=False,
                result=None,
                error=f"Tool '{name}' not found",
                tool_name=name,
                input_params=kwargs
            )
        
        tool_info = self._tools[name]
        
        # Check if tool is ready
        if not tool_info.is_loaded:
            if not self.load_tool(name):
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"Tool '{name}' is not ready and could not be loaded",
                    tool_name=name,
                    input_params=kwargs
                )
        
        try:
            tool_info.status = ToolStatus.BUSY
            
            # Get tool function
            func = tool_info.function
            if not func:
                return ToolExecutionResult(
                    success=False,
                    result=None,
                    error=f"Tool '{name}' has no executable function",
                    tool_name=name,
                    input_params=kwargs
                )
            
            # Execute function
            if tool_info.is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance(name, execution_time, True)
            
            tool_info.status = ToolStatus.READY
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                tool_name=name,
                input_params=kwargs
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance(name, execution_time, False)
            
            tool_info.status = ToolStatus.READY
            
            return ToolExecutionResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=name,
                input_params=kwargs
            )
        finally:
            # Store history
            self._execution_history.append(ToolExecutionResult(
                success=True,
                result=None,
                tool_name=name
            ))
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]
    
    def execute_sync(self, name: str, *args, **kwargs) -> ToolExecutionResult:
        """
        Synchronous version of execute_tool
        
        Args:
            name: Tool name
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            ToolExecutionResult
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.execute_tool(name, *args, **kwargs))
        finally:
            loop.close()
    
    def _update_performance(self, tool_name: str, execution_time: float, success: bool):
        """Update performance statistics"""
        if tool_name not in self._performance_stats:
            return
        
        stats = self._performance_stats[tool_name]
        stats['total_calls'] += 1
        if success:
            stats['successful_calls'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_calls']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
    
    # ============================================
    # Discovery Methods
    # ============================================
    
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """
        Get tool information
        
        Args:
            name: Tool name
        
        Returns:
            ToolInfo or None
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[ToolInfo]:
        """
        Get all registered tools
        
        Returns:
            List of ToolInfo
        """
        return list(self._tools.values())
    
    def get_loaded_tools(self) -> List[str]:
        """
        Get loaded tool names
        
        Returns:
            List of loaded tool names
        """
        return self._loaded_tools.copy()
    
    def get_tools_by_type(self, tool_type: Union[ToolType, str]) -> List[ToolInfo]:
        """
        Get tools by type
        
        Args:
            tool_type: Tool type
        
        Returns:
            List of ToolInfo
        """
        if isinstance(tool_type, str):
            try:
                tool_type = ToolType(tool_type.lower())
            except ValueError:
                return []
        
        return [t for t in self._tools.values() if t.tool_type == tool_type]
    
    def get_tools_by_tag(self, tag: str) -> List[ToolInfo]:
        """
        Get tools by tag
        
        Args:
            tag: Tag to search
        
        Returns:
            List of ToolInfo
        """
        return [t for t in self._tools.values() if tag in t.tags]
    
    def get_tools_by_name_pattern(self, pattern: str) -> List[ToolInfo]:
        """
        Get tools by name pattern
        
        Args:
            pattern: Name pattern (case-insensitive)
        
        Returns:
            List of ToolInfo
        """
        pattern = pattern.lower()
        return [t for t in self._tools.values() if pattern in t.name.lower()]
    
    def get_ready_tools(self) -> List[ToolInfo]:
        """
        Get ready tools
        
        Returns:
            List of ready ToolInfo
        """
        return [t for t in self._tools.values() if t.status == ToolStatus.READY]
    
    # ============================================
    # Statistics Methods
    # ============================================
    
    def get_performance_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Args:
            tool_name: Tool name (None for all)
        
        Returns:
            Performance statistics
        """
        if tool_name:
            return self._performance_stats.get(tool_name, {})
        return self._performance_stats
    
    def get_execution_history(
        self,
        limit: int = 10,
        tool_name: Optional[str] = None,
        success_only: bool = False
    ) -> List[ToolExecutionResult]:
        """
        Get execution history
        
        Args:
            limit: Maximum number of results
            tool_name: Filter by tool name
            success_only: Only successful executions
        
        Returns:
            List of ToolExecutionResult
        """
        history = self._execution_history
        
        if tool_name:
            history = [h for h in history if h.tool_name == tool_name]
        
        if success_only:
            history = [h for h in history if h.success]
        
        return history[-limit:]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Registry statistics
        """
        total = len(self._tools)
        loaded = len(self._loaded_tools)
        
        ready = sum(1 for t in self._tools.values() if t.status == ToolStatus.READY)
        error = sum(1 for t in self._tools.values() if t.status == ToolStatus.ERROR)
        
        by_type = {}
        for tool in self._tools.values():
            by_type[tool.tool_type.value] = by_type.get(tool.tool_type.value, 0) + 1
        
        return {
            'total_tools': total,
            'loaded_tools': loaded,
            'ready_tools': ready,
            'error_tools': error,
            'by_type': by_type
        }
    
    # ============================================
    # Health Check Methods
    # ============================================
    
    def check_tool_health(self, name: str) -> Dict[str, Any]:
        """
        Check health of a tool
        
        Args:
            name: Tool name
        
        Returns:
            Health check result
        """
        if name not in self._tools:
            return {'healthy': False, 'error': f"Tool '{name}' not found"}
        
        tool_info = self._tools[name]
        
        health = {
            'name': name,
            'healthy': True,
            'status': tool_info.status.value,
            'is_loaded': tool_info.is_loaded,
            'version': tool_info.version,
            'updated_at': tool_info.updated_at
        }
        
        # Check if function exists
        if tool_info.is_loaded and not tool_info.function:
            health['healthy'] = False
            health['error'] = "No executable function"
        
        return health
    
    def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all tools
        
        Returns:
            Dictionary of health check results
        """
        return {name: self.check_tool_health(name) for name in self._tools}
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def tool_exists(self, name: str) -> bool:
        """
        Check if tool exists
        
        Args:
            name: Tool name
        
        Returns:
            True if exists
        """
        return name in self._tools
    
    def is_loaded(self, name: str) -> bool:
        """
        Check if tool is loaded
        
        Args:
            name: Tool name
        
        Returns:
            True if loaded
        """
        if name in self._tools:
            return self._tools[name].is_loaded
        return False
    
    def get_tool_names(self) -> List[str]:
        """
        Get all tool names
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def clear_history(self):
        """
        Clear execution history
        """
        self._execution_history = []
        logger.info("🗑️ Execution history cleared")
    
    def clear_all(self):
        """
        Clear all tools and history
        """
        self._tools.clear()
        self._loaded_tools.clear()
        self._execution_history.clear()
        self._performance_stats.clear()
        logger.info("🧹 Cleared all tools and history")


# ============================================
# Decorators for Tool Registration
# ============================================

class ToolRegistryDecorators:
    """Decorators for easy tool registration"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def tool(self, name: str, description: str, **kwargs):
        """
        Decorator to register a function as a tool
        
        Args:
            name: Tool name
            description: Tool description
            **kwargs: Additional arguments for register_function
        
        Returns:
            Decorator
        """
        def decorator(func):
            self.registry.register_function(
                name=name,
                description=description,
                function=func,
                **kwargs
            )
            return func
        return decorator


# ============================================
# Test Function
# ============================================

async def test_tool_registry():
    """Test the tool registry"""
    print("🧪 Testing ToolRegistry...")
    print("=" * 60)
    
    registry = ToolRegistry()
    
    # Register a function
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    registry.register_function(
        name='multiply',
        description='Multiply two numbers',
        function=multiply,
        tags=['math', 'utility']
    )
    
    # Register an async function
    async def greet(name: str) -> str:
        """Greet someone"""
        return f"Hello, {name}!"
    
    registry.register_function(
        name='greet',
        description='Greet someone',
        function=greet,
        tags=['utility']
    )
    
    # Register a class
    class Calculator:
        def add(self, a, b):
            return a + b
    
    registry.register_class(
        name='calculator',
        description='Calculator class',
        class_obj=Calculator,
        tags=['math']
    )
    
    # Get all tools
    print("\n📋 Registered tools:")
    for tool in registry.get_all_tools():
        print(f"  ✅ {tool.name}: {tool.description} ({tool.tool_type.value})")
    
    # Load tools
    print("\n📋 Loading tools:")
    for name in ['multiply', 'greet']:
        if registry.load_tool(name):
            print(f"  ✅ Loaded: {name}")
    
    # Execute tools
    print("\n📋 Executing tools:")
    
    # Execute multiply
    result = await registry.execute_tool('multiply', a=6, b=7)
    print(f"  multiply(6, 7) = {result.result} (time: {result.execution_time:.4f}s)")
    
    # Execute greet
    result = await registry.execute_tool('greet', name='SatQuery')
    print(f"  greet('SatQuery') = {result.result} (time: {result.execution_time:.4f}s)")
    
    # Get stats
    print("\n📋 Registry stats:")
    stats = registry.get_registry_stats()
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Loaded tools: {stats['loaded_tools']}")
    print(f"  Ready tools: {stats['ready_tools']}")
    
    # Get performance stats
    print("\n📋 Performance stats:")
    stats = registry.get_performance_stats('multiply')
    print(f"  multiply - calls: {stats.get('total_calls', 0)}, avg_time: {stats.get('avg_time', 0):.4f}s")
    
    # Health check
    print("\n📋 Health check:")
    for name in ['multiply', 'greet', 'calculator']:
        health = registry.check_tool_health(name)
        status = "✅" if health['healthy'] else "❌"
        print(f"  {status} {name}: {health['status']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_tool_registry())