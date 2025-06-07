# core/memory_trace.py - Enhanced with temporal metadata while preserving existing functionality
import torch
import time
import numpy as np
import hashlib
from typing import Dict, Any, Optional, Union, List, Set
from dataclasses import dataclass, field
from enum import Enum

# NEW: Temporal enums and metadata
class ConsolidationState(Enum):
    INITIAL = "initial"
    CONSOLIDATING = "consolidating" 
    CONSOLIDATED = "consolidated"
    STABLE = "stable"

@dataclass
class TemporalMetadata:
    """Temporal metadata for memory traces."""
    creation_timescale: str = "fast_synaptic"
    consolidation_state: ConsolidationState = ConsolidationState.INITIAL
    temporal_coherence: float = 1.0
    last_temporal_update: float = field(default_factory=time.time)
    cross_timescale_links: Dict[str, Any] = field(default_factory=dict)
    consolidation_strength: float = 0.0
    stability_metric: float = 0.5

@dataclass
class MemoryTrace:
    """Enhanced memory trace with comprehensive metadata AND temporal dynamics"""

    # Core content (PRESERVED from existing)
    content: torch.Tensor  # Embedding vector
    context: Dict[str, Any]  # Rich contextual metadata

    # Temporal information (PRESERVED from existing)
    timestamp: float  # Creation time
    last_access: float  # Last retrieval time
    access_pattern: List[float] = field(default_factory=list)  # Access history

    # Salience and importance (PRESERVED from existing)
    salience: float  # Initial salience score
    current_salience: float = None  # Decayed salience
    importance_votes: Dict[str, float] = field(default_factory=dict)  # User votes

    # Access statistics (PRESERVED from existing)
    access_count: int = 0  # Total access count
    successful_retrievals: int = 0  # Successful retrievals
    context_matches: int = 0  # Context-relevant retrievals

    # Memory lifecycle (PRESERVED from existing)
    trace_id: str = ""  # Unique identifier
    creation_node: str = ""  # Originating node ID
    consolidation_level: int = 0  # 0=episodic, 1=working, 2=semantic
    eviction_protection: bool = False  # Protect from eviction

    # Relationships (PRESERVED from existing)
    associated_traces: Set[str] = field(default_factory=set)  # Related memories
    causal_predecessors: Set[str] = field(default_factory=set)  # Causal history
    interference_traces: Set[str] = field(default_factory=set)  # Conflicting memories

    # NEW: Temporal dynamics
    temporal_metadata: TemporalMetadata = field(default_factory=TemporalMetadata)

    def __post_init__(self):
        # PRESERVED existing post-init logic
        if self.last_access == 0.0:
            self.last_access = self.timestamp
        if self.current_salience is None:
            self.current_salience = self.salience
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID based on content and timestamp (PRESERVED)"""
        content_hash = hashlib.md5(self.content.cpu().numpy().tobytes()).hexdigest()[:8]
        time_hash = hashlib.md5(str(self.timestamp).encode()).hexdigest()[:8]
        return f"trace_{content_hash}_{time_hash}"

    # PRESERVED existing methods
    def update_access_stats(self, current_time: float, context_relevant: bool = True):
        """Update access statistics and patterns (PRESERVED)"""
        self.access_count += 1
        self.last_access = current_time
        self.access_pattern.append(current_time)

        if context_relevant:
            self.context_matches += 1
            self.successful_retrievals += 1

        # Trim access pattern to last 100 entries
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]

        # NEW: Update temporal metadata on access
        self.temporal_metadata.last_temporal_update = current_time

    def compute_decay_score(self, current_time: float, decay_params: Dict[str, float]) -> float:
        """Compute comprehensive decay score for eviction priority (ENHANCED)"""

        # PRESERVED existing decay computation
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

        # NEW: Temporal factors
        temporal_coherence_factor = self.temporal_metadata.temporal_coherence
        consolidation_state_bonus = {
            ConsolidationState.INITIAL: 0.0,
            ConsolidationState.CONSOLIDATING: 1.0,
            ConsolidationState.CONSOLIDATED: 2.0,
            ConsolidationState.STABLE: 3.0
        }.get(self.temporal_metadata.consolidation_state, 0.0)

        # Combined score (higher = less likely to be evicted)
        score = (
                        salience_factor * 3.0 +
                        access_factor * 2.0 +
                        success_rate * 2.0 +
                        context_relevance * 1.5 +
                        time_factor * 1.0 +
                        recency_factor * 1.0 +
                        relationship_factor +
                        consolidation_bonus +
                        temporal_coherence_factor * 0.5 +  # NEW
                        consolidation_state_bonus  # NEW
                ) * protection_factor

        return max(0.0, score)

    def update_salience_decay(self, current_time: float, decay_rate: float = 0.1):
        """Update current salience with temporal decay (ENHANCED)"""
        time_elapsed = current_time - self.timestamp
        self.current_salience = self.salience * np.exp(-decay_rate * time_elapsed / 86400)
        
        # NEW: Update temporal metadata
        self.temporal_metadata.last_temporal_update = current_time

    def add_association(self, other_trace_id: str, association_type: str = "semantic"):
        """Add association with another trace (PRESERVED)"""
        self.associated_traces.add(other_trace_id)

        # Store association metadata in context
        if 'associations' not in self.context:
            self.context['associations'] = {}
        self.context['associations'][other_trace_id] = {
            'type': association_type,
            'created_at': time.time()
        }

    # NEW: Temporal methods
    def get_temporal_age_category(self) -> str:
        """Determine which timescale should handle this trace."""
        age = time.time() - self.timestamp
        
        if age < 1.0:                    # < 1 second
            return "fast_synaptic"
        elif age < 60.0:                 # < 1 minute  
            return "calcium_plasticity"
        elif age < 3600.0:               # < 1 hour
            return "protein_synthesis"
        elif age < 86400.0:              # < 1 day
            return "homeostatic_scaling"
        else:                            # > 1 day
            return "systems_consolidation"
            
    def update_temporal_state(self, temporal_context: Dict[str, Any]) -> None:
        """Update temporal metadata based on current system state."""
        self.temporal_metadata.last_temporal_update = time.time()
        
        # Update consolidation state based on age and context
        age_category = self.get_temporal_age_category()
        
        if age_category in ["protein_synthesis", "homeostatic_scaling"]:
            if self.temporal_metadata.consolidation_state == ConsolidationState.INITIAL:
                self.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATING
                
        # Update temporal coherence
        system_coherence = temporal_context.get("temporal_coherence", 1.0)
        self.temporal_metadata.temporal_coherence = system_coherence
        
        # Update consolidation strength based on access patterns and age
        if self.access_count > 5 and age_category in ["protein_synthesis", "homeostatic_scaling"]:
            self.temporal_metadata.consolidation_strength = min(1.0, 
                self.temporal_metadata.consolidation_strength + 0.1)
        
    def should_consolidate(self, consolidation_threshold: float = 0.7) -> bool:
        """Determine if trace is ready for consolidation."""
        age_category = self.get_temporal_age_category()
        
        return (
            age_category in ["protein_synthesis", "homeostatic_scaling"] and
            self.temporal_metadata.consolidation_strength > consolidation_threshold and
            self.salience > 0.6 and
            self.temporal_metadata.consolidation_state == ConsolidationState.INITIAL
        )
        
    def get_temporal_priority(self) -> int:
        """Get priority for temporal processing."""
        base_priority = int(self.salience * 10)
        
        # Boost priority for traces ready for consolidation
        if self.should_consolidate():
            base_priority += 5
            
        # Boost priority for recent high-salience traces
        if self.get_temporal_age_category() == "fast_synaptic" and self.salience > 0.8:
            base_priority += 3
            
        # Boost priority based on consolidation state
        state_bonus = {
            ConsolidationState.INITIAL: 0,
            ConsolidationState.CONSOLIDATING: 2,
            ConsolidationState.CONSOLIDATED: 1,
            ConsolidationState.STABLE: 0
        }.get(self.temporal_metadata.consolidation_state, 0)
        
        base_priority += state_bonus
            
        return min(10, base_priority)  # Cap at 10

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission (ENHANCED)"""
        # PRESERVED existing serialization
        base_dict = {
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
        
        # NEW: Add temporal metadata
        base_dict['temporal_metadata'] = {
            'creation_timescale': self.temporal_metadata.creation_timescale,
            'consolidation_state': self.temporal_metadata.consolidation_state.value,
            'temporal_coherence': self.temporal_metadata.temporal_coherence,
            'last_temporal_update': self.temporal_metadata.last_temporal_update,
            'cross_timescale_links': self.temporal_metadata.cross_timescale_links,
            'consolidation_strength': self.temporal_metadata.consolidation_strength,
            'stability_metric': self.temporal_metadata.stability_metric
        }
        
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> 'MemoryTrace':
        """Deserialize from dictionary (ENHANCED)"""
        content = torch.tensor(data['content'], device=device, dtype=torch.float32)

        trace = cls(
            content=content,
            context=data['context'],
            timestamp=data['timestamp'],
            last_access=data['last_access'],
            salience=data['salience'],
            trace_id=data['trace_id']
        )

        # PRESERVED existing field restoration
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

        # NEW: Restore temporal metadata
        temporal_data = data.get('temporal_metadata', {})
        if temporal_data:
            trace.temporal_metadata = TemporalMetadata(
                creation_timescale=temporal_data.get('creation_timescale', 'fast_synaptic'),
                consolidation_state=ConsolidationState(temporal_data.get('consolidation_state', 'initial')),
                temporal_coherence=temporal_data.get('temporal_coherence', 1.0),
                last_temporal_update=temporal_data.get('last_temporal_update', time.time()),
                cross_timescale_links=temporal_data.get('cross_timescale_links', {}),
                consolidation_strength=temporal_data.get('consolidation_strength', 0.0),
                stability_metric=temporal_data.get('stability_metric', 0.5)
            )

        return trace

    # NEW: Temporal utility methods
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal statistics for this trace."""
        age = time.time() - self.timestamp
        time_since_last_access = time.time() - self.last_access
        time_since_temporal_update = time.time() - self.temporal_metadata.last_temporal_update
        
        return {
            'age': age,
            'age_category': self.get_temporal_age_category(),
            'time_since_last_access': time_since_last_access,
            'time_since_temporal_update': time_since_temporal_update,
            'consolidation_state': self.temporal_metadata.consolidation_state.value,
            'consolidation_strength': self.temporal_metadata.consolidation_strength,
            'temporal_coherence': self.temporal_metadata.temporal_coherence,
            'temporal_priority': self.get_temporal_priority(),
            'should_consolidate': self.should_consolidate()
        }

    def is_temporally_ready_for(self, operation: str) -> bool:
        """Check if trace is ready for specific temporal operations."""
        age_category = self.get_temporal_age_category()
        
        if operation == "fast_synaptic_processing":
            return age_category == "fast_synaptic" and self.salience > 0.5
            
        elif operation == "calcium_plasticity":
            return age_category in ["fast_synaptic", "calcium_plasticity"] and self.access_count > 1
            
        elif operation == "protein_synthesis":
            return (age_category in ["calcium_plasticity", "protein_synthesis"] and 
                   self.temporal_metadata.consolidation_strength > 0.3)
            
        elif operation == "consolidation":
            return self.should_consolidate()
            
        elif operation == "systems_consolidation":
            return (age_category in ["homeostatic_scaling", "systems_consolidation"] and
                   self.temporal_metadata.consolidation_state == ConsolidationState.CONSOLIDATED)
        
        return False
