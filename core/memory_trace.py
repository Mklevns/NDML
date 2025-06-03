# core/memory_trace.py
import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class MemoryTrace:
    """Enhanced memory trace with comprehensive metadata"""

    # Core content
    content: torch.Tensor  # Embedding vector
    context: Dict[str, Any]  # Rich contextual metadata

    # Temporal information
    timestamp: float  # Creation time
    last_access: float  # Last retrieval time
    access_pattern: List[float] = field(default_factory=list)  # Access history

    # Salience and importance
    salience: float  # Initial salience score
    current_salience: float = None  # Decayed salience
    importance_votes: Dict[str, float] = field(default_factory=dict)  # User votes

    # Access statistics
    access_count: int = 0  # Total access count
    successful_retrievals: int = 0  # Successful retrievals
    context_matches: int = 0  # Context-relevant retrievals

    # Memory lifecycle
    trace_id: str = ""  # Unique identifier
    creation_node: str = ""  # Originating node ID
    consolidation_level: int = 0  # 0=episodic, 1=working, 2=semantic
    eviction_protection: bool = False  # Protect from eviction

    # Relationships
    associated_traces: Set[str] = field(default_factory=set)  # Related memories
    causal_predecessors: Set[str] = field(default_factory=set)  # Causal history
    interference_traces: Set[str] = field(default_factory=set)  # Conflicting memories

    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp
        if self.current_salience is None:
            self.current_salience = self.salience
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID based on content and timestamp"""
        content_hash = hashlib.md5(self.content.cpu().numpy().tobytes()).hexdigest()[:8]
        time_hash = hashlib.md5(str(self.timestamp).encode()).hexdigest()[:8]
        return f"trace_{content_hash}_{time_hash}"

    def update_access_stats(self, current_time: float, context_relevant: bool = True):
        """Update access statistics and patterns"""
        self.access_count += 1
        self.last_access = current_time
        self.access_pattern.append(current_time)

        if context_relevant:
            self.context_matches += 1
            self.successful_retrievals += 1

        # Trim access pattern to last 100 entries
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]

    def compute_decay_score(self, current_time: float, decay_params: Dict[str, float]) -> float:
        """Compute comprehensive decay score for eviction priority"""

        # Time-based decay
        age = current_time - self.timestamp
        recency = current_time - self.last_access

        time_factor = np.exp(-age / decay_params.get('age_scale', 86400))  # 1 day
        recency_factor = np.exp(-recency / decay_params.get('recency_scale', 3600))  # 1 hour

        # Usage-based scoring
        access_factor = np.log1p(self.access_count) / 10.0
        success_rate = self.successful_retrievals / max(1, self.access_count)
        context_relevance = self.context_matches / max(1, self.access_count)

        # Salience contribution
        salience_factor = self.current_salience

        # Relationship importance
        relationship_factor = len(self.associated_traces) * 0.1

        # Protection factor
        protection_factor = 10.0 if self.eviction_protection else 1.0

        # Consolidation level bonus (higher levels are more important)
        consolidation_bonus = self.consolidation_level * 2.0

        # Combined score (higher = less likely to be evicted)
        score = (
                        salience_factor * 3.0 +
                        access_factor * 2.0 +
                        success_rate * 2.0 +
                        context_relevance * 1.5 +
                        time_factor * 1.0 +
                        recency_factor * 1.0 +
                        relationship_factor +
                        consolidation_bonus
                ) * protection_factor

        return max(0.0, score)

    def update_salience_decay(self, current_time: float, decay_rate: float = 0.1):
        """Update current salience with temporal decay"""
        time_elapsed = current_time - self.timestamp
        self.current_salience = self.salience * np.exp(-decay_rate * time_elapsed / 86400)

    def add_association(self, other_trace_id: str, association_type: str = "semantic"):
        """Add association with another trace"""
        self.associated_traces.add(other_trace_id)

        # Store association metadata in context
        if 'associations' not in self.context:
            self.context['associations'] = {}
        self.context['associations'][other_trace_id] = {
            'type': association_type,
            'created_at': time.time()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission"""
        return {
            'trace_id': self.trace_id,
            'content': self.content.cpu().numpy().tolist(),
            'context': self.context,
            'timestamp': self.timestamp,
            'last_access': self.last_access,
            'access_pattern': self.access_pattern,
            'salience': self.salience,
            'current_salience': self.current_salience,
            'importance_votes': self.importance_votes,
            'access_count': self.access_count,
            'successful_retrievals': self.successful_retrievals,
            'context_matches': self.context_matches,
            'creation_node': self.creation_node,
            'consolidation_level': self.consolidation_level,
            'eviction_protection': self.eviction_protection,
            'associated_traces': list(self.associated_traces),
            'causal_predecessors': list(self.causal_predecessors),
            'interference_traces': list(self.interference_traces)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> 'MemoryTrace':
        """Deserialize from dictionary"""
        content = torch.tensor(data['content'], device=device, dtype=torch.float32)

        trace = cls(
            content=content,
            context=data['context'],
            timestamp=data['timestamp'],
            last_access=data['last_access'],
            salience=data['salience'],
            trace_id=data['trace_id']
        )

        # Restore all fields
        trace.access_pattern = data.get('access_pattern', [])
        trace.current_salience = data.get('current_salience', data['salience'])
        trace.importance_votes = data.get('importance_votes', {})
        trace.access_count = data.get('access_count', 0)
        trace.successful_retrievals = data.get('successful_retrievals', 0)
        trace.context_matches = data.get('context_matches', 0)
        trace.creation_node = data.get('creation_node', "")
        trace.consolidation_level = data.get('consolidation_level', 0)
        trace.eviction_protection = data.get('eviction_protection', False)
        trace.associated_traces = set(data.get('associated_traces', []))
        trace.causal_predecessors = set(data.get('causal_predecessors', []))
        trace.interference_traces = set(data.get('interference_traces', []))

        return trace