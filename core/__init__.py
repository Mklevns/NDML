# core/__init__.py - Core components package
"""
NDML Core Components

Contains the fundamental building blocks of the NDML system:
- Distributed Memory Nodes (DMN)
- Memory Traces with temporal metadata
- Multi-timescale dynamics engine
- BTSP update mechanism
- Memory lifecycle management
"""

from .dmn import EnhancedDistributedMemoryNode
from .memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
from .dynamics import MultiTimescaleDynamicsEngine, TemporalState, TemporalEvent
from .btsp import BTSPUpdateMechanism, BTSPUpdateDecision
from .lifecycle import MemoryLifecycleManager, MemoryLifecycleState, LifecycleConfig

__all__ = [
    'EnhancedDistributedMemoryNode',
    'MemoryTrace',
    'TemporalMetadata',
    'ConsolidationState',
    'MultiTimescaleDynamicsEngine',
    'TemporalState',
    'TemporalEvent',
    'BTSPUpdateMechanism',
    'BTSPUpdateDecision',
    'MemoryLifecycleManager',
    'MemoryLifecycleState',
    'LifecycleConfig',
]
