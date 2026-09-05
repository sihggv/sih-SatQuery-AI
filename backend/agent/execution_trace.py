"""
Execution Trace Module for SatQuery AI

This module provides comprehensive audit trail functionality:
1. Step-by-step execution logging
2. Performance metrics per step
3. Error tracking and recovery
4. Export to JSON/CSV
5. Visualization of execution flow
6. Trace management and querying

The execution trace is crucial for:
- Transparency and trust (explainable AI)
- Debugging and optimization
- Compliance and auditing
- Performance monitoring
"""

import logging
import time
import uuid
import json
import csv
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================
# Enums
# ============================================

class TraceStatus(str, Enum):
    """Status of a trace step"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TraceLevel(str, Enum):
    """Level of trace detail"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================
# Data Classes
# ============================================

@dataclass
class TraceStep:
    """
    Single step in the execution trace
    
    Each step represents a discrete action in the execution pipeline:
    - Task Classification
    - Input Validation
    - Model Loading
    - Model Execution
    - Response Formatting
    - etc.
    """
    step_id: str
    step_name: str
    status: TraceStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    children: List['TraceStep'] = field(default_factory=list)
    parent_id: Optional[str] = None
    level: TraceLevel = TraceLevel.INFO
    
    def __post_init__(self):
        if self.step_id is None:
            self.step_id = str(uuid.uuid4())[:8]
        if self.status is None:
            self.status = TraceStatus.PENDING
    
    def start(self):
        """Mark step as started"""
        self.status = TraceStatus.RUNNING
        self.start_time = time.time()
        logger.debug(f"Step started: {self.step_name} ({self.step_id})")
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None):
        """Mark step as completed successfully"""
        self.status = TraceStatus.SUCCESS
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.output_data = output_data or {}
        logger.debug(f"Step completed: {self.step_name} ({self.duration:.3f}s)")
    
    def fail(self, error: str):
        """Mark step as failed"""
        self.status = TraceStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.error = error
        logger.error(f"Step failed: {self.step_name} - {error}")
    
    def skip(self, reason: Optional[str] = None):
        """Mark step as skipped"""
        self.status = TraceStatus.SKIPPED
        self.end_time = time.time()
        if reason:
            self.metadata = self.metadata or {}
            self.metadata['skip_reason'] = reason
        logger.info(f"Step skipped: {self.step_name} - {reason or 'No reason provided'}")
    
    def add_child(self, child: 'TraceStep'):
        """Add a child step"""
        child.parent_id = self.step_id
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_id': self.step_id,
            'step_name': self.step_name,
            'status': self.status.value if isinstance(self.status, TraceStatus) else self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error': self.error,
            'metadata': self.metadata,
            'children': [child.to_dict() for child in self.children],
            'parent_id': self.parent_id,
            'level': self.level.value if isinstance(self.level, TraceLevel) else self.level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceStep':
        """Create from dictionary"""
        if 'children' in data:
            data['children'] = [cls.from_dict(child) for child in data['children']]
        return cls(**data)
    
    def get_duration_str(self) -> str:
        """Get duration as string"""
        if self.duration is None:
            return "N/A"
        if self.duration < 0.001:
            return f"{self.duration * 1000:.2f}ms"
        elif self.duration < 1:
            return f"{self.duration * 1000:.0f}ms"
        else:
            return f"{self.duration:.3f}s"
    
    def get_status_icon(self) -> str:
        """Get status icon"""
        icons = {
            TraceStatus.PENDING: "⏳",
            TraceStatus.RUNNING: "🔄",
            TraceStatus.SUCCESS: "✅",
            TraceStatus.FAILED: "❌",
            TraceStatus.SKIPPED: "⏭️",
            TraceStatus.TIMEOUT: "⏰"
        }
        return icons.get(self.status, "❓")


@dataclass
class ExecutionTrace:
    """
    Complete execution trace for a task
    
    This captures the entire execution flow including:
    - All steps performed
    - Timing information
    - Input/output data
    - Errors and warnings
    - Performance metrics
    """
    trace_id: str
    task: str
    query: str
    image_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: TraceStatus = TraceStatus.PENDING
    steps: List[TraceStep] = field(default_factory=list)
    root_step: Optional[TraceStep] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())[:8]
        if self.start_time is None:
            self.start_time = time.time()
    
    def start(self):
        """Start the trace"""
        self.status = TraceStatus.RUNNING
        self.start_time = time.time()
        self.root_step = TraceStep(
            step_id=f"root_{self.trace_id}",
            step_name=f"Task: {self.task}",
            status=TraceStatus.RUNNING,
            start_time=self.start_time,
            input_data={'query': self.query, 'image_type': self.image_type}
        )
        logger.info(f"Trace started: {self.trace_id} - {self.task}")
    
    def complete(self):
        """Complete the trace"""
        self.status = TraceStatus.SUCCESS
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.root_step:
            self.root_step.complete()
        self._generate_summary()
        logger.info(f"Trace completed: {self.trace_id} - Duration: {self.duration:.3f}s")
    
    def fail(self, error: str):
        """Fail the trace"""
        self.status = TraceStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.root_step:
            self.root_step.fail(error)
        self._generate_summary()
        logger.error(f"Trace failed: {self.trace_id} - {error}")
    
    def add_step(self, step: TraceStep) -> TraceStep:
        """Add a step to the trace"""
        if not self.root_step:
            self.start()
        self.root_step.add_child(step)
        self.steps.append(step)
        return step
    
    def create_step(
        self,
        step_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: TraceLevel = TraceLevel.INFO
    ) -> TraceStep:
        """Create and add a new step"""
        step = TraceStep(
            step_id=str(uuid.uuid4())[:8],
            step_name=step_name,
            status=TraceStatus.PENDING,
            start_time=time.time(),
            input_data=input_data,
            metadata=metadata,
            level=level
        )
        self.add_step(step)
        return step
    
    def get_step_by_id(self, step_id: str) -> Optional[TraceStep]:
        """Get a step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
            # Check children
            if step.children:
                for child in step.children:
                    if child.step_id == step_id:
                        return child
        return None
    
    def get_steps_by_status(self, status: TraceStatus) -> List[TraceStep]:
        """Get steps by status"""
        return [s for s in self.steps if s.status == status]
    
    def _generate_summary(self):
        """Generate summary statistics"""
        total_steps = len(self.steps)
        success_steps = sum(1 for s in self.steps if s.status == TraceStatus.SUCCESS)
        failed_steps = sum(1 for s in self.steps if s.status == TraceStatus.FAILED)
        skipped_steps = sum(1 for s in self.steps if s.status == TraceStatus.SKIPPED)
        
        total_duration = sum(s.duration or 0 for s in self.steps)
        avg_duration = total_duration / total_steps if total_steps > 0 else 0
        
        # Find slowest step
        slowest_step = max(self.steps, key=lambda s: s.duration or 0) if self.steps else None
        
        self.summary = {
            'trace_id': self.trace_id,
            'task': self.task,
            'status': self.status.value if isinstance(self.status, TraceStatus) else self.status,
            'total_duration': self.duration,
            'total_steps': total_steps,
            'success_steps': success_steps,
            'failed_steps': failed_steps,
            'skipped_steps': skipped_steps,
            'total_step_duration': total_duration,
            'avg_step_duration': avg_duration,
            'slowest_step': slowest_step.step_name if slowest_step else None,
            'slowest_step_duration': slowest_step.duration if slowest_step else None,
            'has_errors': failed_steps > 0,
            'success_rate': success_steps / total_steps if total_steps > 0 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trace_id': self.trace_id,
            'task': self.task,
            'query': self.query[:500],
            'image_type': self.image_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value if isinstance(self.status, TraceStatus) else self.status,
            'steps': [step.to_dict() for step in self.steps],
            'root_step': self.root_step.to_dict() if self.root_step else None,
            'metadata': self.metadata,
            'summary': self.summary,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionTrace':
        """Create from dictionary"""
        trace = cls(
            trace_id=data.get('trace_id', str(uuid.uuid4())[:8]),
            task=data.get('task', ''),
            query=data.get('query', ''),
            image_type=data.get('image_type', ''),
            start_time=data.get('start_time', time.time()),
            end_time=data.get('end_time'),
            duration=data.get('duration'),
            status=TraceStatus(data.get('status', 'pending')),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', [])
        )
        if 'steps' in data:
            trace.steps = [TraceStep.from_dict(s) for s in data['steps']]
        return trace
    
    def to_json(self, pretty: bool = True) -> str:
        """Export trace to JSON"""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_csv(self) -> str:
        """Export steps to CSV"""
        output = []
        output.append(['step_id', 'step_name', 'status', 'duration', 'error', 'start_time', 'end_time'])
        
        for step in self.steps:
            output.append([
                step.step_id,
                step.step_name,
                step.status.value if isinstance(step.status, TraceStatus) else step.status,
                step.duration or 0,
                step.error or '',
                step.start_time,
                step.end_time or ''
            ])
        
        return '\n'.join(','.join(str(v) for v in row) for row in output)


# ============================================
# Execution Trace Manager
# ============================================

class ExecutionTraceManager:
    """
    Manages execution traces with persistence and query capabilities
    
    Features:
    1. Store and retrieve traces
    2. Query traces by ID, task, status, etc.
    3. Export traces to JSON/CSV
    4. Generate reports
    5. Cleanup old traces
    6. Performance analytics
    """
    
    def __init__(self, storage_dir: Optional[str] = None, max_traces: int = 1000):
        """
        Initialize the trace manager
        
        Args:
            storage_dir: Directory to store trace files (optional)
            max_traces: Maximum number of traces to keep in memory
        """
        self._traces: Dict[str, ExecutionTrace] = {}
        self._storage_dir: Optional[Path] = None
        self._max_traces = max_traces
        
        if storage_dir:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_traces()
        
        logger.info(f"✅ ExecutionTraceManager initialized (max_traces: {max_traces})")
    
    # ============================================
    # Core Operations
    # ============================================
    
    def create_trace(
        self,
        task: str,
        query: str,
        image_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ExecutionTrace:
        """
        Create a new execution trace
        
        Args:
            task: Task name
            query: User query
            image_type: Type of image(s)
            metadata: Additional metadata
            tags: Tags for categorization
        
        Returns:
            ExecutionTrace object
        """
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4())[:8],
            task=task,
            query=query,
            image_type=image_type,
            metadata=metadata or {},
            tags=tags or []
        )
        trace.start()
        self._traces[trace.trace_id] = trace
        
        # Limit number of traces
        if len(self._traces) > self._max_traces:
            self._cleanup_oldest()
        
        logger.debug(f"Created trace: {trace.trace_id}")
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Get a trace by ID"""
        return self._traces.get(trace_id)
    
    def update_trace(self, trace: ExecutionTrace):
        """Update a trace"""
        self._traces[trace.trace_id] = trace
        if self._storage_dir:
            self._save_trace(trace)
    
    def complete_trace(self, trace_id: str) -> bool:
        """Complete a trace"""
        trace = self.get_trace(trace_id)
        if trace:
            trace.complete()
            if self._storage_dir:
                self._save_trace(trace)
            return True
        return False
    
    def fail_trace(self, trace_id: str, error: str) -> bool:
        """Mark a trace as failed"""
        trace = self.get_trace(trace_id)
        if trace:
            trace.fail(error)
            if self._storage_dir:
                self._save_trace(trace)
            return True
        return False
    
    def add_step_to_trace(
        self,
        trace_id: str,
        step_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: TraceLevel = TraceLevel.INFO
    ) -> Optional[TraceStep]:
        """Add a step to an existing trace"""
        trace = self.get_trace(trace_id)
        if trace:
            return trace.create_step(step_name, input_data, metadata, level)
        return None
    
    def complete_step(self, trace_id: str, step_id: str, output_data: Optional[Dict[str, Any]] = None) -> bool:
        """Complete a specific step"""
        trace = self.get_trace(trace_id)
        if trace:
            step = trace.get_step_by_id(step_id)
            if step:
                step.complete(output_data)
                if self._storage_dir:
                    self._save_trace(trace)
                return True
        return False
    
    def fail_step(self, trace_id: str, step_id: str, error: str) -> bool:
        """Fail a specific step"""
        trace = self.get_trace(trace_id)
        if trace:
            step = trace.get_step_by_id(step_id)
            if step:
                step.fail(error)
                if self._storage_dir:
                    self._save_trace(trace)
                return True
        return False
    
    # ============================================
    # Context Manager
    # ============================================
    
    def trace_context(
        self,
        task: str,
        query: str,
        image_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Context manager for automatic tracing
        
        Usage:
            with trace_manager.trace_context('vqa', query, 'single') as trace:
                # Do work
                step = trace.create_step('Processing')
                # ...
        """
        return _TraceContext(self, task, query, image_type, metadata, tags)
    
    # ============================================
    # Query Methods
    # ============================================
    
    def get_all_traces(self) -> List[ExecutionTrace]:
        """Get all traces"""
        return list(self._traces.values())
    
    def get_recent_traces(self, limit: int = 10) -> List[ExecutionTrace]:
        """Get recent traces"""
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]
    
    def find_traces(
        self,
        task: Optional[str] = None,
        status: Optional[TraceStatus] = None,
        tag: Optional[str] = None,
        query_contains: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExecutionTrace]:
        """
        Find traces with filters
        
        Args:
            task: Filter by task name
            status: Filter by status
            tag: Filter by tag
            query_contains: Filter by query content
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of matching traces
        """
        traces = list(self._traces.values())
        
        if task:
            traces = [t for t in traces if t.task == task]
        
        if status:
            traces = [t for t in traces if t.status == status]
        
        if tag:
            traces = [t for t in traces if tag in t.tags]
        
        if query_contains:
            traces = [t for t in traces if query_contains.lower() in t.query.lower()]
        
        # Sort by time (newest first)
        traces.sort(key=lambda t: t.start_time, reverse=True)
        
        return traces[offset:offset + limit]
    
    def get_traces_by_task(self, task: str) -> List[ExecutionTrace]:
        """Get all traces for a specific task"""
        return self.find_traces(task=task)
    
    def get_traces_by_status(self, status: TraceStatus) -> List[ExecutionTrace]:
        """Get all traces with a specific status"""
        return self.find_traces(status=status)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about traces"""
        traces = list(self._traces.values())
        total = len(traces)
        
        if total == 0:
            return {
                'total_traces': 0,
                'tasks': {},
                'status_counts': {},
                'avg_duration': 0,
                'success_rate': 0
            }
        
        task_counts = defaultdict(int)
        status_counts = defaultdict(int)
        total_duration = 0.0
        success_count = 0
        
        for trace in traces:
            task_counts[trace.task] += 1
            status_counts[trace.status.value] += 1
            if trace.duration:
                total_duration += trace.duration
            if trace.status == TraceStatus.SUCCESS:
                success_count += 1
        
        return {
            'total_traces': total,
            'tasks': dict(task_counts),
            'status_counts': dict(status_counts),
            'avg_duration': total_duration / total,
            'total_duration': total_duration,
            'success_rate': success_count / total if total > 0 else 0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all traces"""
        all_steps = []
        for trace in self._traces.values():
            all_steps.extend(trace.steps)
        
        if not all_steps:
            return {'total_steps': 0}
        
        durations = [s.duration for s in all_steps if s.duration is not None]
        
        return {
            'total_steps': len(all_steps),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'total_duration': sum(durations),
            'steps_by_status': {
                status.value: len([s for s in all_steps if s.status == status])
                for status in TraceStatus
            }
        }
    
    # ============================================
    # Export Methods
    # ============================================
    
    def to_json(self, trace_id: str, pretty: bool = True) -> str:
        """Export a trace to JSON"""
        trace = self.get_trace(trace_id)
        if not trace:
            return json.dumps({'error': 'Trace not found'})
        return trace.to_json(pretty)
    
    def export_to_json(self, trace_id: str, filepath: str) -> bool:
        """Export a trace to a JSON file"""
        json_str = self.to_json(trace_id)
        if not json_str:
            return False
        
        with open(filepath, 'w') as f:
            f.write(json_str)
        logger.info(f"Exported trace to {filepath}")
        return True
    
    def export_all_to_json(self, directory: str) -> int:
        """Export all traces to JSON files"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for trace_id, trace in self._traces.items():
            filepath = dir_path / f"{trace_id}.json"
            if self.export_to_json(trace_id, str(filepath)):
                count += 1
        
        return count
    
    def export_to_csv(self, filepath: str) -> bool:
        """Export traces summary to CSV"""
        traces = list(self._traces.values())
        if not traces:
            return False
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trace_id', 'task', 'query', 'image_type',
                'status', 'start_time', 'end_time', 'duration',
                'total_steps', 'success_steps', 'failed_steps',
                'has_errors', 'tags'
            ])
            
            for trace in traces:
                writer.writerow([
                    trace.trace_id,
                    trace.task,
                    trace.query[:100],
                    trace.image_type,
                    trace.status.value if isinstance(trace.status, TraceStatus) else trace.status,
                    trace.start_time,
                    trace.end_time,
                    trace.duration,
                    trace.summary.get('total_steps', 0),
                    trace.summary.get('success_steps', 0),
                    trace.summary.get('failed_steps', 0),
                    trace.summary.get('has_errors', False),
                    ', '.join(trace.tags)
                ])
        
        logger.info(f"Exported traces to {filepath}")
        return True
    
    def export_report(self, filepath: str) -> bool:
        """Export a comprehensive report"""
        stats = self.get_stats()
        metrics = self.get_performance_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'performance_metrics': metrics,
            'traces': [trace.to_dict() for trace in self._traces.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported report to {filepath}")
        return True
    
    # ============================================
    # Storage Methods
    # ============================================
    
    def _load_existing_traces(self):
        """Load existing traces from storage"""
        if not self._storage_dir:
            return
        
        for filepath in self._storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    trace = ExecutionTrace.from_dict(data)
                    self._traces[trace.trace_id] = trace
            except Exception as e:
                logger.error(f"Failed to load trace from {filepath}: {e}")
        
        logger.info(f"Loaded {len(self._traces)} traces from storage")
    
    def _save_trace(self, trace: ExecutionTrace):
        """Save a trace to storage"""
        if not self._storage_dir:
            return
        
        filepath = self._storage_dir / f"{trace.trace_id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(trace.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trace {trace.trace_id}: {e}")
    
    def _cleanup_oldest(self):
        """Remove oldest traces when limit is exceeded"""
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.start_time)
        oldest = traces[0]
        del self._traces[oldest.trace_id]
        logger.info(f"Removed oldest trace: {oldest.trace_id}")
    
    def cleanup(self, older_than_days: int = 30):
        """Clean up old traces"""
        cutoff = time.time() - (older_than_days * 24 * 60 * 60)
        
        to_delete = []
        for trace_id, trace in self._traces.items():
            if trace.start_time < cutoff:
                to_delete.append(trace_id)
        
        for trace_id in to_delete:
            del self._traces[trace_id]
            if self._storage_dir:
                filepath = self._storage_dir / f"{trace_id}.json"
                if filepath.exists():
                    filepath.unlink()
        
        logger.info(f"🧹 Cleaned up {len(to_delete)} old traces")
        return len(to_delete)
    
    def clear_all(self):
        """Clear all traces"""
        self._traces.clear()
        if self._storage_dir:
            for filepath in self._storage_dir.glob("*.json"):
                filepath.unlink()
        logger.info("🧹 Cleared all traces")


# ============================================
# Context Manager
# ============================================

class _TraceContext:
    """Context manager for tracing"""
    
    def __init__(
        self,
        manager: ExecutionTraceManager,
        task: str,
        query: str,
        image_type: str,
        metadata: Optional[Dict[str, Any]],
        tags: Optional[List[str]]
    ):
        self.manager = manager
        self.task = task
        self.query = query
        self.image_type = image_type
        self.metadata = metadata
        self.tags = tags
        self.trace: Optional[ExecutionTrace] = None
    
    def __enter__(self) -> ExecutionTrace:
        self.trace = self.manager.create_trace(
            task=self.task,
            query=self.query,
            image_type=self.image_type,
            metadata=self.metadata,
            tags=self.tags
        )
        return self.trace
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            error = str(exc_val) if exc_val else "Unknown error"
            self.manager.fail_trace(self.trace.trace_id, error)
        else:
            self.manager.complete_trace(self.trace.trace_id)


# ============================================
# Test Functions
# ============================================

def test_execution_trace():
    """Test the execution trace system"""
    print("🧪 Testing ExecutionTrace...")
    print("=" * 60)
    
    # Create manager
    manager = ExecutionTraceManager()
    
    # Create a trace manually
    print("\n📝 Creating manual trace:")
    trace = manager.create_trace(
        task='vqa',
        query='What is the land-cover in this image?',
        image_type='single_optical',
        tags=['test', 'vqa']
    )
    
    # Add steps
    step1 = trace.create_step('Task Classification', input_data={'query': trace.query})
    step1.complete({'task': 'vqa', 'confidence': 0.87})
    
    step2 = trace.create_step('Input Validation', input_data={'image_type': 'single_optical'})
    step2.complete({'valid': True})
    
    step3 = trace.create_step('Model Execution', input_data={'model': 'vqa_model'})
    step3.complete({'answer': 'The image shows agricultural fields and urban structures.'})
    
    trace.complete()
    manager.update_trace(trace)
    
    print(f"  Trace ID: {trace.trace_id}")
    print(f"  Task: {trace.task}")
    print(f"  Status: {trace.status.value}")
    print(f"  Steps: {len(trace.steps)}")
    print(f"  Duration: {trace.duration:.3f}s")
    
    # Use context manager
    print("\n📝 Using context manager:")
    with manager.trace_context(
        task='change_detection',
        query='What changed between these dates?',
        image_type='bi_temporal',
        tags=['test', 'change']
    ) as ctx_trace:
        step = ctx_trace.create_step('Change Detection')
        step.complete({'change': 'Urban increased by 12.5%'})
        print(f"  Trace ID: {ctx_trace.trace_id}")
        print(f"  Status: {ctx_trace.status.value}")
    
    # Get stats
    print("\n📊 Statistics:")
    stats = manager.get_stats()
    print(f"  Total traces: {stats['total_traces']}")
    print(f"  Tasks: {stats['tasks']}")
    print(f"  Avg duration: {stats['avg_duration']:.3f}s")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    metrics = manager.get_performance_metrics()
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Avg step duration: {metrics['avg_duration']:.3f}s")
    
    # Export to JSON
    print("\n📄 Exporting trace to JSON:")
    json_str = manager.to_json(trace.trace_id, pretty=True)
    print(f"  JSON length: {len(json_str)} characters")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_execution_trace()