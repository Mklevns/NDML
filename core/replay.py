import numpy as np
from collections import deque
from typing import List, Dict, Any # Added Dict, Any for potential future use or if trace has complex attributes
# Assuming EnhancedMemoryTrace will be used by the replay buffer, and it has an 'id' attribute.
# Also assuming AdaptiveRelationalEncoder might be passed or used.
from core.memory_trace import EnhancedMemoryTrace
# from core.lifecycle import AdaptiveRelationalEncoder # If AdaptiveRelationalEncoder is defined there

class MemoryReplayBuffer:
    """Implements hippocampal replay for offline consolidation"""

    def __init__(self, capacity=10000):
        self.buffer: deque[EnhancedMemoryTrace] = deque(maxlen=capacity) # Type hint for buffer
        self.priority_weights: Dict[str, float] = {} # Type hint for priority_weights

    def add_experience(self, trace: EnhancedMemoryTrace, importance_score: float):
        """Add memory with replay probability proportional to importance"""
        self.buffer.append(trace)
        # Assuming trace has an 'id' attribute which is a string
        self.priority_weights[trace.id] = importance_score

    def sample_replay_batch(self, batch_size=32, temperature=1.0) -> List[EnhancedMemoryTrace]:
        """Sample memories for replay with importance-based probability"""
        if not self.buffer: # Check if buffer is empty
            return []

        current_buffer_size = len(self.buffer)
        actual_batch_size = min(batch_size, current_buffer_size)

        if actual_batch_size == 0: # Should be caught by `if not self.buffer` but as a safeguard
            return []

        # Compute sampling probabilities
        traces = list(self.buffer) # Convert deque to list for consistent indexing

        # Ensure all traces in the buffer have an ID in priority_weights, default to 1.0 if missing
        weights = np.array([
            self.priority_weights.get(t.id, 1.0) for t in traces
        ])

        if np.sum(weights) == 0: # Handle case where all weights are zero
            probs = np.ones(len(traces)) / len(traces) # Uniform probability
        else:
            # Temperature-scaled softmax for exploration/exploitation
            exp_weights = np.exp(weights / temperature)
            probs = exp_weights / np.sum(exp_weights)

        # Ensure probabilities sum to 1, can have small floating point issues
        probs = probs / np.sum(probs)


        # Sample without replacement
        try:
            indices = np.random.choice(
                len(traces), size=actual_batch_size, replace=False, p=probs
            )
            return [traces[i] for i in indices]
        except ValueError as e:
            # This can happen if probs don't sum to 1 due to float precision, or if buffer is smaller than batch size after filtering
            # Fallback to simpler sampling or error handling
            # For now, if there's an issue with probabilities, sample uniformly without replacement
            # Or if buffer is smaller than actual_batch_size (should not happen due to min calculation)
            # Log the error for debugging
            print(f"Warning: Error during np.random.choice in sample_replay_batch: {e}. Using uniform sampling.")
            indices = np.random.choice(len(traces), size=actual_batch_size, replace=False)
            return [traces[i] for i in indices]


    async def replay_consolidation(self, encoder: Any, num_replays=5): # Added Type hint for encoder
        """Perform multiple replay cycles"""
        # Assuming encoder is some form of relational encoder, e.g., AdaptiveRelationalEncoder
        # The proposal does not specify what the encoder does in this context.
        # The original proposal's replay_consolidation modifies trace.associated_traces and priority_weights.

        for _ in range(num_replays): # Renamed cycle to _ as it's not used
            if not self.buffer: # Don't attempt to sample if buffer is empty
                continue

            batch = self.sample_replay_batch()
            if not batch: # sample_replay_batch can return empty list
                continue

            # Strengthen connections between replayed memories
            for i, trace_a in enumerate(batch[:-1]):
                trace_b = batch[i + 1]

                # Temporal binding: memories replayed together bind together
                # Assuming trace_a and trace_b are EnhancedMemoryTrace objects with associated_traces and id
                trace_a.associated_traces.add(trace_b.id)
                trace_b.associated_traces.add(trace_a.id)

                # Update importance based on co-replay
                # Ensure IDs exist in priority_weights, otherwise initialize them
                self.priority_weights[trace_a.id] = self.priority_weights.get(trace_a.id, 1.0) * 1.1
                self.priority_weights[trace_b.id] = self.priority_weights.get(trace_b.id, 1.0) * 1.1
