# __init__.py - Main NDML package initialization
"""
NDML - Neuromorphic Distributed Memory Layer

A sophisticated AI memory system inspired by biological neural networks,
featuring multi-timescale dynamics, distributed consensus, and intelligent
memory lifecycle management.
"""

__version__ = "1.0.0"
__author__ = "NDML Team"
__description__ = "Neuromorphic Distributed Memory Layer for AI Systems"

import logging
import warnings
from typing import Dict, Any, Optional

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Core components
try:
    from .core.dmn import EnhancedDistributedMemoryNode
    from .core.memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
    from .core.dynamics import MultiTimescaleDynamicsEngine
    from .core.btsp import BTSPUpdateMechanism
    from .core.lifecycle import MemoryLifecycleManager
except ImportError as e:
    warnings.warn(f"Failed to import core components: {e}")

# Integration components
try:
    from .integration.memory_gateway import MemoryGateway
    from .integration.llm_wrapper import NDMLIntegratedLLM
    from .integration.fusion_network import MemoryFusionNetwork
    from .integration.temporal_bridge import TemporalLLMBridge
except ImportError as e:
    warnings.warn(f"Failed to import integration components: {e}")

# System management
try:
    from .main import NDMLSystemManager
except ImportError as e:
    warnings.warn(f"Failed to import system manager: {e}")

# Define public API
__all__ = [
    # Core components
    'EnhancedDistributedMemoryNode',
    'MemoryTrace',
    'TemporalMetadata', 
    'ConsolidationState',
    'MultiTimescaleDynamicsEngine',
    'BTSPUpdateMechanism',
    'MemoryLifecycleManager',
    
    # Integration components
    'MemoryGateway',
    'NDMLIntegratedLLM',
    'MemoryFusionNetwork',
    'TemporalLLMBridge',
    
    # System management
    'NDMLSystemManager',
    
    # Utilities
    'get_version',
    'get_system_info',
    'configure_logging',
]

def get_version() -> str:
    """Get NDML version string."""
    return __version__

def get_system_info() -> Dict[str, Any]:
    """Get system information and component availability."""
    import torch
    import sys
    import platform
    
    info = {
        'ndml_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'components_available': {},
    }
    
    # Check component availability
    components = [
        'EnhancedDistributedMemoryNode',
        'MultiTimescaleDynamicsEngine', 
        'MemoryGateway',
        'NDMLIntegratedLLM',
        'MemoryFusionNetwork',
        'NDMLSystemManager',
    ]
    
    for component in components:
        info['components_available'][component] = component in globals()
    
    return info

def configure_logging(level: str = "INFO", 
                     format_str: Optional[str] = None,
                     filename: Optional[str] = None) -> None:
    """Configure NDML logging."""
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if filename:
        handlers.append(logging.FileHandler(filename))
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=handlers
    )

# ============================================================================
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

# ============================================================================
# integration/__init__.py - Integration components package
"""
NDML Integration Components

Components for integrating NDML with external systems:
- Memory Gateway for coordinating distributed memory
- LLM wrapper for language model integration  
- Memory fusion networks
- Temporal bridge for dynamics integration
"""

from .memory_gateway import MemoryGateway
from .llm_wrapper import NDMLIntegratedLLM, BaseLLMAdapter, TransformerAdapter, MambaAdapter
from .fusion_network import MemoryFusionNetwork
from .temporal_bridge import TemporalLLMBridge

__all__ = [
    'MemoryGateway',
    'NDMLIntegratedLLM',
    'BaseLLMAdapter',
    'TransformerAdapter', 
    'MambaAdapter',
    'MemoryFusionNetwork',
    'TemporalLLMBridge',
]

# ============================================================================
# consensus/__init__.py - Consensus components package  
"""
NDML Consensus Components

Distributed consensus mechanisms for coordinating memory updates
across multiple nodes using neuromorphic principles.
"""

# Note: consensus/neuromorphic.py is referenced but not yet implemented
# This would contain the NeuromorphicConsensusLayer class

__all__ = [
    # Will be populated when consensus components are implemented
]

# ============================================================================
# deployment/__init__.py - Deployment components package
"""
NDML Deployment Components

Tools and configurations for deploying NDML in production environments,
including Kubernetes manifests and consensus nodes.
"""

from .consensus_node import ConsensusNode, create_consensus_node

__all__ = [
    'ConsensusNode',
    'create_consensus_node',
]

# ============================================================================
# utils/__init__.py - Utility functions package
"""
NDML Utility Functions

Common utilities for data processing, monitoring, and system management.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def normalize_embeddings(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize embeddings to unit length."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=dim)

def compute_similarity_matrix(embeddings1: torch.Tensor, 
                            embeddings2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of embeddings."""
    norm1 = normalize_embeddings(embeddings1)
    norm2 = normalize_embeddings(embeddings2)
    return torch.mm(norm1, norm2.t())

def create_attention_mask(lengths: List[int], max_length: Optional[int] = None) -> torch.Tensor:
    """Create attention mask from sequence lengths."""
    if max_length is None:
        max_length = max(lengths)
    
    mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask

def batch_embeddings(embeddings: List[torch.Tensor], 
                    padding_value: float = 0.0) -> torch.Tensor:
    """Batch variable-length embeddings with padding."""
    max_length = max(emb.shape[0] for emb in embeddings)
    embed_dim = embeddings[0].shape[-1]
    
    batched = torch.full((len(embeddings), max_length, embed_dim), 
                        padding_value, dtype=embeddings[0].dtype)
    
    for i, emb in enumerate(embeddings):
        batched[i, :emb.shape[0]] = emb
    
    return batched

def exponential_moving_average(new_value: float, 
                             current_avg: float, 
                             alpha: float = 0.1) -> float:
    """Compute exponential moving average."""
    return alpha * new_value + (1 - alpha) * current_avg

def create_decay_schedule(initial_value: float,
                         decay_rate: float,
                         num_steps: int) -> np.ndarray:
    """Create exponential decay schedule."""
    steps = np.arange(num_steps)
    return initial_value * np.exp(-decay_rate * steps)

class PerformanceMonitor:
    """Simple performance monitoring utility."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.timestamps = {}
    
    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.timestamps[metric_name] = []
        
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(time.time())
        
        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]
            self.timestamps[metric_name] = self.timestamps[metric_name][-self.window_size:]
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'latest': values[-1],
        }

def save_embeddings(embeddings: torch.Tensor, 
                   filepath: str,
                   metadata: Optional[Dict[str, Any]] = None):
    """Save embeddings to file with optional metadata."""
    data = {'embeddings': embeddings.cpu()}
    if metadata:
        data['metadata'] = metadata
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, filepath)
    logger.info(f"Saved embeddings to {filepath}")

def load_embeddings(filepath: str) -> Dict[str, Any]:
    """Load embeddings from file."""
    data = torch.load(filepath, map_location='cpu')
    logger.info(f"Loaded embeddings from {filepath}")
    return data

__all__ = [
    'normalize_embeddings',
    'compute_similarity_matrix', 
    'create_attention_mask',
    'batch_embeddings',
    'exponential_moving_average',
    'create_decay_schedule',
    'PerformanceMonitor',
    'save_embeddings',
    'load_embeddings',
]
