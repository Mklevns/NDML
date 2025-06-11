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
