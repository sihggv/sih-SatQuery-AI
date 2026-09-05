"""
Core Module for SatQuery AI Backend

This module contains the core components of the SatQuery AI system:
1. Agentic Controller - The brain that orchestrates everything
2. Task Classifier - Classifies user queries into tasks
3. Execution Trace - Audit trail for transparency

The core module is responsible for:
- Intelligent task routing (Agentic AI)
- Model orchestration and execution
- Audit trail and explainability
- Input validation and error handling
"""

from .agentic_controller import AgenticController
from .task_classifier import TaskClassifier
from .execution_trace import (
    ExecutionTrace,
    ExecutionTraceManager,
    TraceStep,
    TraceStatus
)

# ============================================
# Package Metadata
# ============================================

__version__ = '1.0.0'
__author__ = 'SatQuery AI Team'

# ============================================
# Core Features Description
# ============================================

CORE_FEATURES = {
    'agentic_controller': {
        'description': 'Orchestrates task classification, model selection, and execution',
        'key_features': [
            'Zero-GIS Barrier - Users don\'t need GIS knowledge',
            'Agentic Multi-Model - Coordinates multiple specialist models',
            'Explainable - Provides evidence and confidence scores',
            'Automatic Task Routing - No model selection needed'
        ]
    },
    'task_classifier': {
        'description': 'Classifies user queries into appropriate tasks',
        'supported_tasks': ['vqa', 'grounding', 'change_detection', 'optical_sar_fusion', 'captioning']
    },
    'execution_trace': {
        'description': 'Audit trail for transparency and trust',
        'features': [
            'Step-by-step execution logging',
            'Performance metrics per step',
            'Error tracking and recovery',
            'Export to JSON/CSV'
        ]
    }
}

# ============================================
# Exports
# ============================================

__all__ = [
    # Main Controller
    'AgenticController',
    
    # Task Classification
    'TaskClassifier',
    
    # Execution Trace
    'ExecutionTrace',
    'ExecutionTraceManager',
    'TraceStep',
    'TraceStatus',
    
    # Package Info
    'CORE_FEATURES'
]

# ============================================
# Quick Test Function
# ============================================

def test_core_package():
    """Test the core package imports"""
    print("🧪 Testing Core Package...")
    print("=" * 60)
    
    print("\n📋 Core Features:")
    for name, info in CORE_FEATURES.items():
        print(f"\n  ✅ {name}:")
        print(f"     Description: {info['description']}")
        if 'key_features' in info:
            for feature in info['key_features']:
                print(f"     - {feature}")
        if 'supported_tasks' in info:
            print(f"     Supported: {', '.join(info['supported_tasks'])}")
    
    # Test imports
    print("\n📋 Testing Imports:")
    try:
        from .agentic_controller import AgenticController
        print("  ✅ AgenticController imported")
    except ImportError as e:
        print(f"  ❌ AgenticController: {e}")
    
    try:
        from .task_classifier import TaskClassifier
        print("  ✅ TaskClassifier imported")
    except ImportError as e:
        print(f"  ❌ TaskClassifier: {e}")
    
    try:
        from .execution_trace import ExecutionTrace, ExecutionTraceManager, TraceStep, TraceStatus
        print("  ✅ ExecutionTrace components imported")
    except ImportError as e:
        print(f"  ❌ ExecutionTrace: {e}")
    
    print("\n✅ Core Package test complete!")


if __name__ == "__main__":
    test_core_package()