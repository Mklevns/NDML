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
import sys as _sys

# Alias lowercase package name for safe imports
_sys.modules.setdefault('ndml', _sys.modules[__name__])

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
