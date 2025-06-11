# ndml/core/__init__.py
"""
NDML Core Components

Contains the fundamental building blocks of the NDML system:
- Distributed Memory Nodes (DMN)
- Memory Traces with temporal metadata
- Multi-timescale dynamics engine
- BTSP update mechanism
- Memory lifecycle management
"""

from .dmn import GPUAcceleratedDMN as EnhancedDistributedMemoryNode
from .memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
from .dynamics import IntegratedMultiTimescaleDynamics as MultiTimescaleDynamicsEngine, TimescaleSchedule
from .btsp import BTSPUpdateMechanism, BTSPUpdateDecision
from .lifecycle import MemoryLifecycleManager, MemoryLifecycleState, LifecycleConfig

__all__ = [
    'EnhancedDistributedMemoryNode',
    'MemoryTrace',
    'TemporalMetadata',
    'ConsolidationState',
    'MultiTimescaleDynamicsEngine', # Alias for IntegratedMultiTimescaleDynamics
    'TimescaleSchedule', # Replaces TemporalState, removed TemporalEvent
    'BTSPUpdateMechanism',
    'BTSPUpdateDecision',
    'MemoryLifecycleManager',
    'MemoryLifecycleState',
    'LifecycleConfig',
]
