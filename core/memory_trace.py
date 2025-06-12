import torch
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

@dataclass
class ImportanceSignals:
    """Enhanced importance tracking with multiple timescales"""
    # Short-term signals (minutes to hours)
    recent_access_rate: float = 0.0  # Access frequency in last hour
    burst_detection: float = 0.0      # Sudden spike in access

    # Medium-term signals (hours to days)
    circadian_relevance: float = 0.0  # Time-of-day patterns
    weekly_pattern_score: float = 0.0  # Day-of-week patterns

    # Long-term signals (days to months)
    decay_resistance: float = 0.0     # How well it resists forgetting
    semantic_centrality: float = 0.0  # Hub-ness in memory graph

    # Biological signals
    btsp_calcium: float = 0.0
    replay_count: int = 0
    consolidation_cycles: int = 0

    def compute_composite_importance(self, weights: Dict[str, float]) -> float:
        """Weighted combination of all signals"""
        scores = {
            'recent': self.recent_access_rate,
            'burst': self.burst_detection,
            'circadian': self.circadian_relevance,
            'weekly': self.weekly_pattern_score,
            'decay': self.decay_resistance,
            'centrality': self.semantic_centrality,
            'calcium': self.btsp_calcium,
            'replay': min(self.replay_count / 10.0, 1.0)
        }
        return sum(scores[k] * weights.get(k, 1.0) for k in scores)

@dataclass
class EnhancedMemoryTrace:
    """Extended memory trace with new features"""
    id: str
    content: str # Assuming this was intended to be part of the trace; the original issue implies MemoryTrace (now EnhancedMemoryTrace) is used by DMN which stores embeddings.
    embedding: torch.Tensor
    timestamp: float

    # Original features
    access_count: int = 0
    successful_retrievals: int = 0
    associated_traces: set = field(default_factory=set)

    # New features
    importance_signals: ImportanceSignals = field(default_factory=ImportanceSignals)
    consolidation_level: int = 0  # 0=content, 1=relational, 2=schematic
    context_vector: Optional[torch.Tensor] = None
    replay_history: List[float] = field(default_factory=list)
    interference_score: float = 0.0

    # Hierarchical organization
    parent_schema: Optional[str] = None
    child_memories: set = field(default_factory=set)

    def update_importance(self, btsp_calcium, access_pattern): # access_pattern seems unused in the proposal's version
        """Update all importance signals"""
        self.importance_signals.btsp_calcium = btsp_calcium
        # Ensure access_count is updated before this call for recent_access_rate to be meaningful
        self.importance_signals.recent_access_rate = (
            self.access_count / (time.time() - self.timestamp + 1e-6) # Added epsilon to avoid division by zero
        )

        # Detect access bursts
        if len(self.replay_history) > 1:
            # Ensure replay_history contains timestamps of replays
            # Convert replay_history to numpy array for diff if it's not already
            replay_history_np = np.array(self.replay_history)
            recent_interval = replay_history_np[-1] - replay_history_np[-2]
            # Calculate average interval only if there are enough points for np.diff
            avg_interval = np.mean(np.diff(replay_history_np)) if len(replay_history_np) > 2 else recent_interval
            if recent_interval < 1e-6: # Avoid division by zero or extremely small numbers
                self.importance_signals.burst_detection = 1.0 # Max burst if interval is tiny
            else:
                self.importance_signals.burst_detection = avg_interval / recent_interval
        else:
            self.importance_signals.burst_detection = 0.0 # No burst if not enough history
