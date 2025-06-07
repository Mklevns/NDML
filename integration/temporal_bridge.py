# integration/temporal_bridge.py - Bridge between temporal engine and LLM integration

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from ..core.dynamics import MultiTimescaleDynamicsEngine, TemporalState
from ..core.memory_trace import MemoryTrace

class TemporalLLMBridge:
    """Bridge between temporal dynamics and LLM integration."""
    
    def __init__(self, temporal_engine: MultiTimescaleDynamicsEngine):
        self.temporal_engine = temporal_engine
        self.attention_modulation_enabled = True
        self.memory_formation_modulation_enabled = True
        
    async def process_llm_token_sequence(self, token_embeddings: List[np.ndarray],
                                       context_length: int) -> Dict[str, Any]:
        """Process LLM token sequence through temporal dynamics."""
        
        temporal_outputs = {}
        
        # Process each token
        for i, embedding in enumerate(token_embeddings):
            token_output = await self._process_single_token(
                embedding, context_length + i
            )
            temporal_outputs[f"token_{i}"] = token_output
            
        # Get system-wide temporal context
        temporal_state = self.temporal_engine.get_temporal_state()
        
        # Compute attention modulation weights
        attention_weights = await self._compute_attention_modulation(temporal_state)
        
        # Compute memory formation strength
        formation_strength = await self._compute_memory_formation_strength(temporal_state)
        
        return {
            "token_outputs": temporal_outputs,
            "attention_modulation": attention_weights,
            "memory_formation_strength": formation_strength,
            "temporal_state": temporal_state,
            "system_coherence": temporal_state.coherence_metric
        }
        
    async def _process_single_token(self, embedding: np.ndarray, 
                                  position: int) -> Dict[str, Any]:
        """Process a single token through temporal dynamics."""
        
        # Compute token salience
        salience = self._compute_token_salience(embedding)
        
        # Create temporary memory trace for this token
        temp_trace = MemoryTrace(
            content=embedding,
            context={"position": position, "timestamp": time.time()},
            salience=salience,
            timestamp=time.time(),
            trace_id=f"token_{position}_{time.time()}"
        )
        
        # Inject into temporal dynamics if high salience
        if salience > 0.7:
            await self.temporal_engine.inject_event(
                "memory_trace_activation",
                "fast_synaptic",
                {
                    "trace_data": {
                        "content": embedding,
                        "trace_id": temp_trace.trace_id,
                        "salience": salience
                    },
                    "source": "llm_token"
                }
            )
            
        return {
            "salience": salience,
            "temporal_injection": salience > 0.7,
            "trace_id": temp_trace.trace_id
        }
        
    def _compute_token_salience(self, embedding: np.ndarray) -> float:
        """Compute salience of a token based on its embedding."""
        
        # Simple salience based on embedding magnitude and variance
        magnitude = np.linalg.norm(embedding)
        variance = np.var(embedding)
        
        # Normalize to [0, 1]
        salience = (magnitude * variance) / (1.0 + magnitude * variance)
        
        return float(salience)
        
    async def _compute_attention_modulation(self, temporal_state: TemporalState) -> Dict[str, float]:
        """Compute attention modulation weights from temporal state."""
        
        modulation_weights = {}
        
        # Fast synaptic activity modulates recent token attention
        fast_activity = temporal_state.fast_synaptic.get("activity_level", 0.5)
        modulation_weights["recent_context_weight"] = 0.5 + fast_activity * 0.5
        
        # Systems consolidation modulates long-term context attention
        # (This would be implemented when systems consolidation process is added)
        modulation_weights["longterm_context_weight"] = 0.3
        
        # Calcium plasticity modulates context integration
        calcium_activity = temporal_state.calcium_plasticity.get("calcium_activity", 0.5)
        modulation_weights["context_integration_weight"] = 0.4 + calcium_activity * 0.4
        
        # Protein synthesis modulates memory formation bias
        protein_activity = temporal_state.protein_synthesis.get("consolidation_activity", 0.5) 
        modulation_weights["memory_formation_bias"] = 0.2 + protein_activity * 0.6
        
        return modulation_weights
        
    async def _compute_memory_formation_strength(self, temporal_state: TemporalState) -> float:
        """Compute memory formation strength from temporal state."""
        
        # Base formation strength
        base_strength = 0.5
        
        # Calcium plasticity contributes to formation strength
        calcium_activity = temporal_state.calcium_plasticity.get("calcium_activity", 0.0)
        calcium_contribution = calcium_activity * 0.3
        
        # Fast synaptic activity contributes
        fast_activity = temporal_state.fast_synaptic.get("activity_level", 0.0)
        fast_contribution = fast_activity * 0.2
        
        # System coherence modulates overall strength
        coherence_factor = temporal_state.coherence_metric
        
        formation_strength = (base_strength + calcium_contribution + fast_contribution) * coherence_factor
        
        return min(1.0, formation_strength)
        
    async def modulate_attention_matrix(self, attention_matrix: np.ndarray,
                                      sequence_length: int) -> np.ndarray:
        """Apply temporal modulation to attention matrix."""
        
        if not self.attention_modulation_enabled:
            return attention_matrix
            
        temporal_state = self.temporal_engine.get_temporal_state()
        modulation_weights = await self._compute_attention_modulation(temporal_state)
        
        modulated_matrix = attention_matrix.copy()
        
        # Apply recent context bias (last 20% of sequence)
        recent_cutoff = int(sequence_length * 0.8)
        recent_weight = modulation_weights.get("recent_context_weight", 1.0)
        modulated_matrix[:, recent_cutoff:] *= recent_weight
        
        # Apply long-term context bias (first 20% of sequence)
        longterm_cutoff = int(sequence_length * 0.2)
        longterm_weight = modulation_weights.get("longterm_context_weight", 1.0)
        modulated_matrix[:, :longterm_cutoff] *= longterm_weight
        
        # Normalize to maintain attention properties
        row_sums = modulated_matrix.sum(axis=1, keepdims=True)
        modulated_matrix = modulated_matrix / (row_sums + 1e-8)
        
        return modulated_matrix
        
    async def should_form_memory(self, content: np.ndarray, context: Dict[str, Any]) -> bool:
        """Determine if content should be formed into long-term memory."""
        
        if not self.memory_formation_modulation_enabled:
            return True  # Default behavior
            
        temporal_state = self.temporal_engine.get_temporal_state()
        formation_strength = await self._compute_memory_formation_strength(temporal_state)
        
        # Content-based factors
        content_salience = self._compute_token_salience(content)
        
        # Context-based factors
        context_importance = context.get("importance", 0.5)
        
        # Combined decision
        formation_probability = formation_strength * content_salience * context_importance
        
        # Add some randomness for exploration
        random_factor = np.random.random()
        
        return formation_probability > random_factor
