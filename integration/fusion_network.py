# integration/fusion_network.py - Memory Fusion Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class MemoryFusionNetwork(nn.Module):
    """
    Neural network for fusing retrieved memories with LLM processing.
    
    Implements sophisticated attention-based fusion mechanisms to integrate
    retrieved memory traces with current LLM hidden states.
    """
    
    def __init__(self,
                 model_dimension: int,
                 memory_dimension: int,
                 num_attention_heads: int = 8,
                 fusion_layers: int = 2,
                 dropout_rate: float = 0.1,
                 fusion_strategy: str = "attention"):
        """
        Initialize Memory Fusion Network.
        
        Args:
            model_dimension: Dimension of LLM hidden states
            memory_dimension: Dimension of memory embeddings
            num_attention_heads: Number of attention heads
            fusion_layers: Number of fusion layers
            dropout_rate: Dropout rate for regularization
            fusion_strategy: Strategy for fusion ("attention", "gated", "residual")
        """
        super().__init__()
        
        self.model_dimension = model_dimension
        self.memory_dimension = memory_dimension
        self.num_attention_heads = num_attention_heads
        self.fusion_layers = fusion_layers
        self.dropout_rate = dropout_rate
        self.fusion_strategy = fusion_strategy
        
        # Initialize fusion components
        self._init_projection_layers()
        self._init_fusion_layers()
        self._init_output_layers()
        
        # Performance tracking
        self.fusion_stats = {
            'total_fusions': 0,
            'average_memory_count': 0.0,
            'attention_entropy': 0.0,
            'fusion_time': 0.0,
        }
        
        logger.info(f"MemoryFusionNetwork initialized: {model_dimension}D model, "
                   f"{memory_dimension}D memory, strategy={fusion_strategy}")

    def _init_projection_layers(self):
        """Initialize projection layers for dimension alignment."""
        
        # Project memory to model dimension
        self.memory_projection = nn.Sequential(
            nn.Linear(self.memory_dimension, self.model_dimension),
            nn.LayerNorm(self.model_dimension),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Query projection for attention
        self.query_projection = nn.Linear(self.model_dimension, self.model_dimension)
        
        # Key and value projections for memories
        self.key_projection = nn.Linear(self.model_dimension, self.model_dimension)
        self.value_projection = nn.Linear(self.model_dimension, self.model_dimension)

    def _init_fusion_layers(self):
        """Initialize fusion-specific layers based on strategy."""
        
        if self.fusion_strategy == "attention":
            self.fusion_layers_list = nn.ModuleList([
                MultiHeadMemoryAttention(
                    self.model_dimension,
                    self.num_attention_heads,
                    self.dropout_rate
                ) for _ in range(self.fusion_layers)
            ])
            
        elif self.fusion_strategy == "gated":
            self.fusion_layers_list = nn.ModuleList([
                GatedMemoryFusion(
                    self.model_dimension,
                    self.dropout_rate
                ) for _ in range(self.fusion_layers)
            ])
            
        elif self.fusion_strategy == "residual":
            self.fusion_layers_list = nn.ModuleList([
                ResidualMemoryFusion(
                    self.model_dimension,
                    self.dropout_rate
                ) for _ in range(self.fusion_layers)
            ])
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _init_output_layers(self):
        """Initialize output projection layers."""
        
        self.output_projection = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.LayerNorm(self.model_dimension),
            nn.Dropout(self.dropout_rate)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension // 2),
            nn.ReLU(),
            nn.Linear(self.model_dimension // 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                query_states: torch.Tensor,
                memory_embeddings: torch.Tensor,
                memory_metadata: Optional[List[Dict[str, Any]]] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for memory fusion.
        
        Args:
            query_states: [batch_size, model_dim] - Current LLM states
            memory_embeddings: [batch_size, num_memories, memory_dim] - Retrieved memories
            memory_metadata: Optional metadata for each memory
            attention_mask: Optional mask for memory attention
            
        Returns:
            Dictionary containing fused representations and attention weights
        """
        try:
            import time
            start_time = time.time()
            
            batch_size, num_memories, _ = memory_embeddings.shape
            
            # Project memories to model dimension
            projected_memories = self.memory_projection(memory_embeddings)
            # [batch_size, num_memories, model_dim]
            
            # Prepare query, key, value projections
            queries = self.query_projection(query_states.unsqueeze(1))  # [batch_size, 1, model_dim]
            keys = self.key_projection(projected_memories)  # [batch_size, num_memories, model_dim]
            values = self.value_projection(projected_memories)  # [batch_size, num_memories, model_dim]
            
            # Apply memory metadata modulation if available
            if memory_metadata is not None:
                keys, values = self._apply_metadata_modulation(keys, values, memory_metadata)
            
            # Initialize fusion state
            fusion_state = query_states
            attention_weights_history = []
            
            # Apply fusion layers
            for layer in self.fusion_layers_list:
                if self.fusion_strategy == "attention":
                    fusion_state, attention_weights = layer(
                        fusion_state, keys, values, attention_mask
                    )
                    attention_weights_history.append(attention_weights)
                else:
                    fusion_state = layer(fusion_state, projected_memories, attention_mask)
            
            # Final output projection
            fused_output = self.output_projection(fusion_state)
            
            # Compute confidence
            confidence = self.confidence_estimator(fused_output)
            
            # Aggregate attention weights
            if attention_weights_history:
                final_attention_weights = torch.stack(attention_weights_history).mean(dim=0)
            else:
                final_attention_weights = None
            
            # Update statistics
            self._update_fusion_stats(
                num_memories, final_attention_weights, time.time() - start_time
            )
            
            return {
                'fused_states': fused_output,
                'attention_weights': final_attention_weights,
                'confidence': confidence,
                'fusion_metadata': {
                    'num_memories': num_memories,
                    'fusion_strategy': self.fusion_strategy,
                    'processing_time': time.time() - start_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error in memory fusion forward pass: {e}")
            # Return input states as fallback
            return {
                'fused_states': query_states,
                'attention_weights': None,
                'confidence': torch.ones(query_states.shape[0], 1) * 0.5,
                'fusion_metadata': {'error': str(e)}
            }

    def _apply_metadata_modulation(self,
                                   keys: torch.Tensor,
                                   values: torch.Tensor,
                                   memory_metadata: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply metadata-based modulation to keys and values."""
        try:
            batch_size, num_memories, model_dim = keys.shape
            
            # Extract metadata features
            metadata_features = []
            for batch_idx in range(batch_size):
                batch_features = []
                for mem_idx in range(num_memories):
                    if batch_idx < len(memory_metadata) and mem_idx < len(memory_metadata[batch_idx]):
                        metadata = memory_metadata[batch_idx][mem_idx]
                        features = self._extract_metadata_features(metadata)
                    else:
                        features = torch.zeros(16)  # Default feature vector
                    batch_features.append(features)
                metadata_features.append(torch.stack(batch_features))
            
            metadata_tensor = torch.stack(metadata_features).to(keys.device)
            # [batch_size, num_memories, 16]
            
            # Create modulation factors
            modulation_network = nn.Sequential(
                nn.Linear(16, model_dim // 4),
                nn.ReLU(),
                nn.Linear(model_dim // 4, model_dim),
                nn.Sigmoid()
            ).to(keys.device)
            
            modulation_factors = modulation_network(metadata_tensor)
            # [batch_size, num_memories, model_dim]
            
            # Apply modulation
            modulated_keys = keys * modulation_factors
            modulated_values = values * modulation_factors
            
            return modulated_keys, modulated_values
            
        except Exception as e:
            logger.error(f"Error applying metadata modulation: {e}")
            return keys, values

    def _extract_metadata_features(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Extract numerical features from memory metadata."""
        features = torch.zeros(16)
        
        try:
            # Salience and importance
            features[0] = float(metadata.get('salience', 0.5))
            features[1] = float(metadata.get('importance', 0.5))
            
            # Temporal features
            features[2] = float(metadata.get('age_hours', 0.0)) / 24.0  # Normalize to days
            features[3] = float(metadata.get('recency_hours', 0.0)) / 24.0
            
            # Access patterns
            features[4] = float(metadata.get('access_count', 0)) / 100.0  # Normalize
            features[5] = float(metadata.get('success_rate', 0.5))
            
            # Context similarity
            features[6] = float(metadata.get('context_similarity', 0.5))
            features[7] = float(metadata.get('domain_relevance', 0.5))
            
            # Consolidation state (one-hot)
            consolidation_states = ['initial', 'consolidating', 'consolidated', 'stable']
            consolidation_state = metadata.get('consolidation_state', 'initial')
            if consolidation_state in consolidation_states:
                features[8 + consolidation_states.index(consolidation_state)] = 1.0
            
            # Additional features
            features[12] = float(metadata.get('retrieval_confidence', 0.5))
            features[13] = float(metadata.get('novelty_score', 0.5))
            features[14] = float(metadata.get('error_score', 0.5))
            features[15] = float(metadata.get('temporal_coherence', 1.0))
            
        except Exception as e:
            logger.error(f"Error extracting metadata features: {e}")
        
        return features

    def _update_fusion_stats(self,
                             num_memories: int,
                             attention_weights: Optional[torch.Tensor],
                             processing_time: float):
        """Update fusion statistics."""
        try:
            self.fusion_stats['total_fusions'] += 1
            
            # Update rolling averages
            alpha = 0.1  # Exponential moving average factor
            
            self.fusion_stats['average_memory_count'] = (
                (1 - alpha) * self.fusion_stats['average_memory_count'] +
                alpha * num_memories
            )
            
            self.fusion_stats['fusion_time'] = (
                (1 - alpha) * self.fusion_stats['fusion_time'] +
                alpha * processing_time
            )
            
            # Compute attention entropy if available
            if attention_weights is not None:
                entropy = self._compute_attention_entropy(attention_weights)
                self.fusion_stats['attention_entropy'] = (
                    (1 - alpha) * self.fusion_stats['attention_entropy'] +
                    alpha * entropy
                )
                
        except Exception as e:
            logger.error(f"Error updating fusion stats: {e}")

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        try:
            # Flatten and normalize attention weights
            flat_weights = attention_weights.flatten()
            probabilities = F.softmax(flat_weights, dim=0)
            
            # Compute entropy
            log_probs = torch.log(probabilities + 1e-10)
            entropy = -torch.sum(probabilities * log_probs).item()
            
            return entropy
            
        except Exception as e:
            logger.error(f"Error computing attention entropy: {e}")
            return 0.0

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion network statistics."""
        return self.fusion_stats.copy()


class MultiHeadMemoryAttention(nn.Module):
    """Multi-head attention for memory fusion."""
    
    def __init__(self, model_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                query: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention for memory fusion.
        
        Args:
            query: [batch_size, model_dim]
            keys: [batch_size, num_memories, model_dim]
            values: [batch_size, num_memories, model_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Expand query to match attention interface
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, model_dim]
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            query_expanded, keys, values, key_padding_mask=attention_mask
        )
        
        # Remove sequence dimension and apply residual connection
        attended_output = attended_output.squeeze(1)  # [batch_size, model_dim]
        output = self.layer_norm(query + self.dropout(attended_output))
        
        # Ensure attention weights are properly normalized
        attention_weights_squeezed = attention_weights.squeeze(1)
        if attention_weights_squeezed.dim() > 1:
            attention_weights_normalized = F.softmax(attention_weights_squeezed, dim=-1)
        else:
            attention_weights_normalized = attention_weights_squeezed
        
        return output, attention_weights_normalized


class GatedMemoryFusion(nn.Module):
    """Gated fusion mechanism for memory integration."""
    
    def __init__(self, model_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid()
        )
        
        # Memory aggregation
        self.memory_aggregator = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                query: torch.Tensor,
                memories: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply gated fusion.
        
        Args:
            query: [batch_size, model_dim]
            memories: [batch_size, num_memories, model_dim]
            attention_mask: Optional mask
            
        Returns:
            Fused representation
        """
        # Aggregate memories (simple mean for now)
        if attention_mask is not None:
            masked_memories = memories * attention_mask.unsqueeze(-1)
            aggregated_memory = masked_memories.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            aggregated_memory = memories.mean(dim=1)
        
        # Process aggregated memory
        processed_memory = self.memory_aggregator(aggregated_memory)
        
        # Compute gate
        gate_input = torch.cat([query, processed_memory], dim=1)
        gate = self.gate_network(gate_input)
        
        # Apply gated fusion
        fused = gate * processed_memory + (1 - gate) * query
        
        # Apply layer norm and residual connection
        output = self.layer_norm(query + self.dropout(fused))
        
        return output


class ResidualMemoryFusion(nn.Module):
    """Residual fusion mechanism for memory integration."""
    
    def __init__(self, model_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Memory processing
        self.memory_processor = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim * 2, model_dim)
        )
        
        # Query processing
        self.query_processor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                query: torch.Tensor,
                memories: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply residual fusion.
        
        Args:
            query: [batch_size, model_dim]
            memories: [batch_size, num_memories, model_dim]
            attention_mask: Optional mask
            
        Returns:
            Fused representation
        """
        # Process query
        processed_query = self.query_processor(query)
        
        # Process and aggregate memories
        processed_memories = self.memory_processor(memories)
        
        if attention_mask is not None:
            masked_memories = processed_memories * attention_mask.unsqueeze(-1)
            aggregated_memory = masked_memories.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            aggregated_memory = processed_memories.mean(dim=1)
        
        # Residual connection
        fused = processed_query + aggregated_memory
        
        # Layer norm and final residual
        output = self.layer_norm(query + self.dropout(fused))
        
        return output
