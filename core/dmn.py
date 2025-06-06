# core/dmn.py
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .memory_trace import MemoryTrace
from .btsp import BTSPUpdateMechanism
from .lifecycle import MemoryLifecycleManager
from .dynamics import MultiTimescaleDynamicsEngine

logger = logging.getLogger(__name__)


class EnhancedDistributedMemoryNode(nn.Module):
    """Production-ready DMN with all optimizations and biological features"""

    def __init__(self,
                 node_id: str,
                 dimension: int,
                 capacity: int = 10000,
                 specialization: str = "general",
                 device: str = "cuda",
                 config: Optional[Dict[str, Any]] = None):

        super().__init__()

        self.node_id = node_id
        self.dimension = dimension
        self.capacity = capacity
        self.specialization = specialization
        self.device = device
        self.config = config or self._default_config()

        # Initialize core components
        self._init_memory_storage()
        self._init_weight_matrices()
        self._init_biological_mechanisms()
        self._init_indexing_system()
        self._init_performance_tracking()

        # Thread-safe operations
        self.lock = threading.RLock()
        self.async_executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Enhanced DMN {node_id} initialized: {dimension}D, capacity={capacity}, specialization={specialization}")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration parameters"""
        return {
            'btsp': {
                'calcium_threshold': 0.7,
                'decay_rate': 0.95,
                'novelty_weight': 0.4,
                'importance_weight': 0.3,
                'error_weight': 0.3,
                'learning_rate': 0.1
            },
            'consolidation': {
                'threshold': 0.8,
                'interval_seconds': 3600,  # 1 hour
                'max_traces_per_cycle': 100
            },
            'lifecycle': {
                'eviction_batch_size': 50,
                'decay_params': {
                    'age_scale': 86400,  # 1 day
                    'recency_scale': 3600,  # 1 hour
                    'salience_decay': 0.1
                }
            },
            'indexing': {
                'index_type': 'HNSW',  # or 'Flat'
                'hnsw_m': 16,
                'hnsw_ef_construction': 200,
                'similarity_threshold': 0.5
            },
            'dynamics': {
                'calcium_decay_ms': 200,
                'protein_decay_ms': 30000,
                'eligibility_decay_ms': 5000,
                'competition_strength': 0.1
            }
        }

    def _init_memory_storage(self):
        """Initialize memory storage structures"""
        self.memory_traces: List[MemoryTrace] = []
        self.trace_index: Dict[str, int] = {}  # trace_id -> list index
        self.specialization_counts: Dict[str, int] = defaultdict(int)
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def _init_weight_matrices(self):
        """Initialize fast and slow weight matrices"""
        # Fast weights for rapid adaptation
        self.fast_weights = nn.Parameter(
            torch.randn(self.capacity, self.dimension, device=self.device) * 0.01
        )

        # Slow weights for consolidated knowledge
        self.slow_weights = nn.Parameter(
            torch.randn(self.capacity, self.dimension, device=self.device) * 0.01
        )

        # Usage tracking for weight slots
        self.weight_usage = torch.zeros(self.capacity, device=self.device)
        self.last_weight_update = torch.zeros(self.capacity, device=self.device)

    def _init_biological_mechanisms(self):
        """Initialize biological plasticity mechanisms"""
        self.btsp_mechanism = BTSPUpdateMechanism(
            **self.config['btsp']
        )

        self.dynamics_engine = MultiTimescaleDynamicsEngine(
            dimension=self.dimension,
            capacity=self.capacity,
            device=self.device,
            **self.config['dynamics']
        )

        self.lifecycle_manager = MemoryLifecycleManager(
            node_id=self.node_id,
            **self.config['lifecycle']
        )

    def _init_indexing_system(self):
        """Initialize FAISS indexing with optimizations"""
        index_config = self.config['indexing']

        if index_config['index_type'] == 'HNSW':
            # HNSW index for better performance
            base_index = faiss.IndexHNSWFlat(self.dimension, index_config['hnsw_m'])
            base_index.hnsw.efConstruction = index_config['hnsw_ef_construction']
            self.index = faiss.IndexIDMap(base_index)
        else:
            # Flat index for exact search
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)

        # GPU acceleration if available
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                logger.info(f"DMN {self.node_id}: Using GPU-accelerated FAISS")
            except Exception as e:
                logger.warning(f"DMN {self.node_id}: GPU FAISS failed: {e}")
                self.gpu_resources = None
        else:
            self.gpu_resources = None

        self.next_faiss_id = 0
        self.faiss_id_to_trace_id: Dict[int, str] = {}

    def _init_performance_tracking(self):
        """Initialize performance and statistics tracking"""
        self.stats = {
            'total_traces': 0,
            'total_retrievals': 0,
            'total_updates': 0,
            'consolidations': 0,
            'evictions': 0,
            'cache_hits': 0,
            'index_rebuilds': 0,
            'specialization_distribution': defaultdict(int),
            'average_access_count': 0.0,
            'memory_utilization': 0.0
        }

        self.performance_history = deque(maxlen=1000)
        self.last_stats_update = time.time()

    async def add_memory_trace_async(self,
                                     content: torch.Tensor,
                                     context: Dict[str, Any],
                                     salience: float,
                                     user_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """Add memory trace with full biological processing"""

        current_time = time.time()

        # Create memory trace
        trace = MemoryTrace(
            content=content.detach().cpu(),
            context=context.copy(),
            timestamp=current_time,
            salience=salience,
            creation_node=self.node_id
        )

        # Compute update decision using BTSP mechanism
        update_decision = await self.btsp_mechanism.should_update_async(
            input_state=content,
            existing_traces=self.memory_traces[-50:],  # Check recent traces
            context=context,
            user_feedback=user_feedback
        )

        if not update_decision.should_update:
            logger.debug(f"DMN {self.node_id}: Update rejected for trace {trace.trace_id}")
            return False

        # Check capacity and evict if necessary
        if len(self.memory_traces) >= self.capacity:
            await self._intelligent_eviction_async()

        # Add trace to storage
        with self.lock:
            self.memory_traces.append(trace)
            self.trace_index[trace.trace_id] = len(self.memory_traces) - 1

            # Update specialization tracking
            specialization = context.get('domain', 'general')
            self.specialization_counts[specialization] += 1

        # Add to FAISS index
        await self._add_to_index_async(trace)

        # Update biological dynamics
        await self.dynamics_engine.process_update_async(
            content, salience, update_decision.calcium_level
        )

        # Update fast weights
        await self._update_fast_weights_async(content, salience, update_decision.learning_rate)

        # Update statistics
        self.stats['total_traces'] += 1
        self.stats['total_updates'] += 1

        logger.debug(f"DMN {self.node_id}: Added trace {trace.trace_id}")
        return True

    async def retrieve_memories_async(self,
                                      query: torch.Tensor,
                                      k: int = 10,
                                      context_filter: Optional[Dict[str, Any]] = None,
                                      similarity_threshold: float = None) -> List[Tuple[MemoryTrace, float]]:
        """Advanced memory retrieval with context filtering"""

        if self.index.ntotal == 0:
            return []

        # Use configured threshold if not specified
        if similarity_threshold is None:
            similarity_threshold = self.config['indexing']['similarity_threshold']

        # Prepare normalized query
        query_normalized = F.normalize(query.to(self.device), dim=0)
        query_np = query_normalized.detach().cpu().numpy().astype('float32').reshape(1, -1)

        # Perform FAISS search
        try:
            # Search more than k to allow for filtering
            search_k = min(k * 3, self.index.ntotal)
            similarities, faiss_ids = self.index.search(query_np, search_k)

            # Process results
            results = []
            current_time = time.time()

            for sim, faiss_id in zip(similarities[0], faiss_ids[0]):
                if sim < similarity_threshold or faiss_id not in self.faiss_id_to_trace_id:
                    continue

                trace_id = self.faiss_id_to_trace_id[faiss_id]
                if trace_id not in self.trace_index:
                    continue

                trace_idx = self.trace_index[trace_id]
                if trace_idx >= len(self.memory_traces):
                    continue

                trace = self.memory_traces[trace_idx]

                # Apply context filtering
                if context_filter and not self._matches_context_filter(trace, context_filter):
                    continue

                # Update access statistics
                trace.update_access_stats(current_time, context_relevant=True)

                # Record access pattern for dynamics
                self.access_patterns[trace_id].append(current_time)

                results.append((trace, float(sim)))

                if len(results) >= k:
                    break

            # Update dynamics engine with retrieval pattern
            await self.dynamics_engine.process_retrieval_async(
                query, [trace for trace, _ in results]
            )

            self.stats['total_retrievals'] += 1
            return results

        except Exception as e:
            logger.error(f"DMN {self.node_id}: Retrieval error: {e}")
            return []

    def _matches_context_filter(self, trace: MemoryTrace, context_filter: Dict[str, Any]) -> bool:
        """Check if trace matches context filter criteria"""
        for key, expected_value in context_filter.items():
            if key not in trace.context:
                return False

            trace_value = trace.context[key]

            # Handle different comparison types
            if isinstance(expected_value, dict):
                if '$in' in expected_value:
                    if trace_value not in expected_value['$in']:
                        return False
                elif '$gte' in expected_value:
                    if trace_value < expected_value['$gte']:
                        return False
                elif '$lte' in expected_value:
                    if trace_value > expected_value['$lte']:
                        return False
            elif trace_value != expected_value:
                return False

        return True

    async def _add_to_index_async(self, trace: MemoryTrace):
        """Add trace to FAISS index asynchronously"""
        content_normalized = F.normalize(trace.content.to(self.device), dim=0)
        content_np = content_normalized.detach().cpu().numpy().astype('float32').reshape(1, -1)

        # Assign FAISS ID
        faiss_id = self.next_faiss_id
        self.next_faiss_id += 1

        # Add to index
        self.index.add_with_ids(content_np, np.array([faiss_id], dtype=np.int64))

        # Update mapping
        self.faiss_id_to_trace_id[faiss_id] = trace.trace_id

    async def _intelligent_eviction_async(self):
        """Intelligent memory eviction using multiple criteria"""
        if not self.memory_traces:
            return

        current_time = time.time()
        eviction_batch_size = self.config['lifecycle']['eviction_batch_size']

        # Compute eviction scores for all traces
        scored_traces = []
        for i, trace in enumerate(self.memory_traces):
            score = trace.compute_decay_score(current_time, self.config['lifecycle']['decay_params'])
            scored_traces.append((score, i, trace))

        # Sort by score (lowest first = highest eviction priority)
        scored_traces.sort(key=lambda x: x[0])

        # Select traces for eviction (lowest scores)
        traces_to_evict = scored_traces[:eviction_batch_size]

        # Remove from FAISS index
        faiss_ids_to_remove = []
        for _, _, trace in traces_to_evict:
            for faiss_id, trace_id in self.faiss_id_to_trace_id.items():
                if trace_id == trace.trace_id:
                    faiss_ids_to_remove.append(faiss_id)

        if faiss_ids_to_remove:
            self.index.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64))

            # Clean up mappings
            for faiss_id in faiss_ids_to_remove:
                if faiss_id in self.faiss_id_to_trace_id:
                    del self.faiss_id_to_trace_id[faiss_id]

        # Remove from memory traces list (in reverse order to maintain indices)
        eviction_indices = sorted([idx for _, idx, _ in traces_to_evict], reverse=True)

        with self.lock:
            for idx in eviction_indices:
                if idx < len(self.memory_traces):
                    evicted_trace = self.memory_traces.pop(idx)

                    # Update specialization counts
                    specialization = evicted_trace.context.get('domain', 'general')
                    self.specialization_counts[specialization] = max(0,
                                                                     self.specialization_counts[specialization] - 1)

                    # Remove from trace index and update indices
                    if evicted_trace.trace_id in self.trace_index:
                        del self.trace_index[evicted_trace.trace_id]

                    # Clean up access patterns
                    if evicted_trace.trace_id in self.access_patterns:
                        del self.access_patterns[evicted_trace.trace_id]

            # Rebuild trace index to fix indices after removals
            self.trace_index = {trace.trace_id: i for i, trace in enumerate(self.memory_traces)}

        self.stats['evictions'] += len(traces_to_evict)
        logger.info(f"DMN {self.node_id}: Evicted {len(traces_to_evict)} traces")

    async def _update_fast_weights_async(self, content: torch.Tensor, salience: float, learning_rate: float):
        """Update fast weights using biological-inspired plasticity"""

        with torch.no_grad():
            content_normalized = F.normalize(content.to(self.device), dim=0)

            # Find best matching weight slot
            similarities = F.cosine_similarity(
                content_normalized.unsqueeze(0),
                self.fast_weights,
                dim=1
            )

            best_slot = torch.argmax(similarities).item()

            # Update with learning rate modulated by salience
            effective_lr = learning_rate * salience

            # Hebbian-style update with decay
            self.fast_weights[best_slot] = (
                    (1 - effective_lr) * self.fast_weights[best_slot] +
                    effective_lr * content_normalized
            )

            # Normalize to prevent unbounded growth
            self.fast_weights[best_slot] = F.normalize(self.fast_weights[best_slot], dim=0)

            # Update usage tracking
            self.weight_usage[best_slot] += 1
            self.last_weight_update[best_slot] = time.time()

    async def consolidate_memories_async(self):
        """Consolidate important memories from fast to slow weights"""

        current_time = time.time()
        consolidation_config = self.config['consolidation']

        # Find traces eligible for consolidation
        eligible_traces = []
        for trace in self.memory_traces:
            if (trace.current_salience >= consolidation_config['threshold'] or
                    trace.access_count >= 5):
                eligible_traces.append(trace)

        if not eligible_traces:
            return

        # Limit consolidation batch size
        max_traces = consolidation_config['max_traces_per_cycle']
        if len(eligible_traces) > max_traces:
            # Sort by importance and take top traces
            eligible_traces.sort(
                key=lambda t: t.current_salience * np.log1p(t.access_count),
                reverse=True
            )
            eligible_traces = eligible_traces[:max_traces]

        # Compute consolidated representation
        consolidated_vectors = []
        weights = []

        for trace in eligible_traces:
            content = trace.content.to(self.device)
            weight = trace.current_salience * np.log1p(trace.access_count)
            consolidated_vectors.append(content)
            weights.append(weight)

        if consolidated_vectors:
            # Weighted average of important traces
            stacked_vectors = torch.stack(consolidated_vectors)
            weight_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum()

            consolidated = torch.sum(stacked_vectors * weight_tensor.unsqueeze(1), dim=0)
            consolidated = F.normalize(consolidated, dim=0)

            # Find slot in slow weights (least recently used)
            slot_usage_times = self.last_weight_update.cpu().numpy()
            oldest_slot = np.argmin(slot_usage_times)

            # Update slow weights with EMA
            alpha = 0.1
            with torch.no_grad():
                self.slow_weights[oldest_slot] = (
                        (1 - alpha) * self.slow_weights[oldest_slot] +
                        alpha * consolidated
                )

                self.slow_weights[oldest_slot] = F.normalize(
                    self.slow_weights[oldest_slot], dim=0
                )

            # Update consolidation level of traces
            for trace in eligible_traces:
                trace.consolidation_level = min(2, trace.consolidation_level + 1)

            self.stats['consolidations'] += 1
            logger.info(f"DMN {self.node_id}: Consolidated {len(eligible_traces)} traces to slot {oldest_slot}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive node statistics"""
        current_time = time.time()

        # Update derived statistics
        if self.memory_traces:
            avg_access = sum(trace.access_count for trace in self.memory_traces) / len(self.memory_traces)
            self.stats['average_access_count'] = avg_access

        self.stats['memory_utilization'] = len(self.memory_traces) / self.capacity
        self.stats['specialization_distribution'] = dict(self.specialization_counts)

        # Add dynamics engine stats
        dynamics_stats = self.dynamics_engine.get_stats()

        # Add biological mechanism stats
        btsp_stats = self.btsp_mechanism.get_stats()

        return {
            'node_id': self.node_id,
            'specialization': self.specialization,
            'timestamp': current_time,
            'basic_stats': self.stats,
            'dynamics_stats': dynamics_stats,
            'btsp_stats': btsp_stats,
            'capacity_info': {
                'current_traces': len(self.memory_traces),
                'max_capacity': self.capacity,
                'utilization_percent': self.stats['memory_utilization'] * 100
            },
            'index_info': {
                'total_indexed': self.index.ntotal,
                'index_type': self.config['indexing']['index_type'],
                'gpu_accelerated': self.gpu_resources is not None
            }
        }

    def save_checkpoint(self, filepath: str):
        """Save complete node state"""
        checkpoint = {
            'node_id': self.node_id,
            'dimension': self.dimension,
            'capacity': self.capacity,
            'specialization': self.specialization,
            'config': self.config,
            'fast_weights': self.fast_weights.detach().cpu(),
            'slow_weights': self.slow_weights.detach().cpu(),
            'weight_usage': self.weight_usage.detach().cpu(),
            'last_weight_update': self.last_weight_update.detach().cpu(),
            'memory_traces': [trace.to_dict() for trace in self.memory_traces],
            'trace_index': self.trace_index,
            'specialization_counts': dict(self.specialization_counts),
            'stats': self.stats,
            'faiss_id_to_trace_id': self.faiss_id_to_trace_id,
            'next_faiss_id': self.next_faiss_id
        }

        torch.save(checkpoint, filepath)
        logger.info(f"DMN {self.node_id}: Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load complete node state"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore parameters
        self.fast_weights.data = checkpoint['fast_weights'].to(self.device)
        self.slow_weights.data = checkpoint['slow_weights'].to(self.device)
        self.weight_usage = checkpoint['weight_usage'].to(self.device)
        self.last_weight_update = checkpoint['last_weight_update'].to(self.device)

        # Restore memory traces
        self.memory_traces = []
        for trace_data in checkpoint['memory_traces']:
            trace = MemoryTrace.from_dict(trace_data, self.device)
            self.memory_traces.append(trace)

        # Restore other state
        self.trace_index = checkpoint['trace_index']
        self.specialization_counts = defaultdict(int, checkpoint['specialization_counts'])
        self.stats = checkpoint['stats']
        self.faiss_id_to_trace_id = checkpoint['faiss_id_to_trace_id']
        self.next_faiss_id = checkpoint['next_faiss_id']

        # Rebuild FAISS index
        self._rebuild_faiss_index()

        logger.info(f"DMN {self.node_id}: Loaded checkpoint from {filepath}")

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from memory traces"""
        # Clear existing index
        self.index.reset()

        if self.memory_traces:
            # Prepare all vectors and IDs
            vectors = []
            faiss_ids = []

            # Rebuild mapping
            new_faiss_id_to_trace_id = {}
            new_faiss_id = 0

            for trace in self.memory_traces:
                content_normalized = F.normalize(trace.content.to(self.device), dim=0)
                vectors.append(content_normalized.detach().cpu().numpy())

                faiss_ids.append(new_faiss_id)
                new_faiss_id_to_trace_id[new_faiss_id] = trace.trace_id
                new_faiss_id += 1

            # Add all vectors at once
            vectors_np = np.vstack(vectors).astype('float32')
            faiss_ids_np = np.array(faiss_ids, dtype=np.int64)

            self.index.add_with_ids(vectors_np, faiss_ids_np)

            # Update mappings
            self.faiss_id_to_trace_id = new_faiss_id_to_trace_id
            self.next_faiss_id = new_faiss_id

            logger.info(f"DMN {self.node_id}: Rebuilt FAISS index with {len(vectors)} traces")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'async_executor'):
            self.async_executor.shutdown(wait=True)

        if hasattr(self, 'gpu_resources') and self.gpu_resources:
            self.gpu_resources = None

        logger.info(f"DMN {self.node_id}: Cleaned up resources")