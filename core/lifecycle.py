import torch # For AdaptiveForgetting.compute_interference & .update_decay_rate (via trace.embedding)
import numpy as np # For AdaptiveForgetting.compute_interference (via trace.timestamp)
import asyncio # For HierarchicalConsolidator (asyncio.PriorityQueue)
from collections import deque # For HierarchicalConsolidator
# Assuming ImportanceSignals is in core.memory_trace as per previous step
from core.memory_trace import ImportanceSignals, EnhancedMemoryTrace
from typing import Dict, List, Any # For type hints

class AdaptiveForgetting:
    """Implements active forgetting for memory optimization"""

    def __init__(self, base_decay_rate=0.001):
        self.base_decay_rate = base_decay_rate
        self.interference_matrix = {}  # Track memory interference

    def compute_interference(self, trace_a: EnhancedMemoryTrace, trace_b: EnhancedMemoryTrace) -> float:
        """Calculate interference between memories"""
        # Semantic overlap causes interference
        # Ensure embeddings are on the same device and are 1D
        embedding_a = trace_a.embedding.cpu().squeeze()
        embedding_b = trace_b.embedding.cpu().squeeze()
        if embedding_a.ndim == 0 or embedding_b.ndim == 0: # Should not happen if embeddings are vectors
            overlap = 0.0
        else:
            overlap = torch.cosine_similarity(
                embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), dim=-1
            ).item()


        # Recent memories interfere more
        # Assuming timestamp is float (seconds)
        recency_factor = np.exp(-(abs(trace_a.timestamp - trace_b.timestamp)) / 86400.0) # 1 day in seconds

        return overlap * recency_factor

    def update_decay_rate(self, trace: EnhancedMemoryTrace, all_traces: List[EnhancedMemoryTrace]) -> float:
        """Adapt decay rate based on interference and importance"""
        if not all_traces:
            return self.base_decay_rate

        # Calculate total interference
        total_interference = sum(
            self.compute_interference(trace, other)
            for other in all_traces if other.id != trace.id
        )
        
        # Average interference, avoid division by zero if all_traces contains only 'trace'
        num_other_traces = len(all_traces) - 1 if any(other.id != trace.id for other in all_traces) else 0
        avg_interference = total_interference / num_other_traces if num_other_traces > 0 else 0.0


        # High interference = faster decay
        interference_factor = 1 + avg_interference # Changed from total_interference / len(all_traces)

        # Importance resists decay
        importance_factor = 1 / (1 + trace.importance_signals.decay_resistance)

        return self.base_decay_rate * interference_factor * importance_factor

class MemoryLifecycleManager: # Renamed from HierarchicalConsolidator
    """Multi-level consolidation mimicking sleep stages"""

    def __init__(self):
        self.stages = {
            'rapid': deque(maxlen=1000),      # SWS-like rapid consolidation
            'slow': deque(maxlen=5000),       # REM-like integration
            'schema': deque(maxlen=10000)     # Cortical schema formation
        }
        self.consolidation_queue = asyncio.PriorityQueue()

    async def stage_memory(self, trace: EnhancedMemoryTrace, importance: ImportanceSignals): # Added type hints
        """Route memory to appropriate consolidation stage"""
        # Ensure importance object is used as intended by the proposal
        composite_score = importance.compute_composite_importance({
            'recent': 2.0, 'calcium': 3.0, 'centrality': 1.5
        })

        # Priority queue: higher score = higher priority (negative for min-heap)
        await self.consolidation_queue.put((-composite_score, trace))

        # Assign to stages based on characteristics
        if importance.burst_detection > 0.7:
            self.stages['rapid'].append(trace)
        elif importance.semantic_centrality > 0.6:
            self.stages['schema'].append(trace)
        else:
            self.stages['slow'].append(trace)

    async def consolidate_batch(self, relational_encoder: Any, batch_size=32): # Added type hint for relational_encoder
        """Process consolidation queue in batches"""
        batch: List[EnhancedMemoryTrace] = [] # Added type hint

        while len(batch) < batch_size and not self.consolidation_queue.empty():
            _, trace_item = await self.consolidation_queue.get() # Renamed to avoid clash
            batch.append(trace_item)

        if batch:
            # Group by consolidation type for efficient processing
            rapid_batch = [t for t in batch if t in self.stages['rapid']]
            slow_batch = [t for t in batch if t in self.stages['slow']]
            schema_batch = [t for t in batch if t in self.stages['schema']]

            # Apply stage-specific consolidation
            # These methods are not defined in the proposal for HierarchicalConsolidator
            # Adding placeholder calls or assuming they exist on relational_encoder
            if hasattr(relational_encoder, 'process_rapid_consolidation'):
                await relational_encoder.process_rapid_consolidation(rapid_batch)
            if hasattr(relational_encoder, 'process_slow_wave_consolidation'):
                await relational_encoder.process_slow_wave_consolidation(slow_batch)
            if hasattr(relational_encoder, 'process_schema_consolidation'):
                await relational_encoder.process_schema_consolidation(schema_batch)

# Placeholder for Relational Encoder if it's to be defined here or imported
# class AdaptiveRelationalEncoder(torch.nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
#         super().__init__()
#         # Simplified definition
#         self.fc = torch.nn.Linear(input_dim, output_dim)
#     def forward(self, x):
#         return self.fc(x)
#     async def process_rapid_consolidation(self, batch): pass
#     async def process_slow_wave_consolidation(self, batch): pass
#     async def process_schema_consolidation(self, batch): pass
