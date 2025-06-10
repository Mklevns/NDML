# core/dmn.py - Enhanced with comprehensive temporal integration
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

# Fixed imports - handle both relative and absolute import cases
try:
    # Try relative imports first (when run as module)
    from .memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
    from .btsp import BTSPUpdateMechanism
    from .lifecycle import MemoryLifecycleManager
    from .dynamics import MultiTimescaleDynamicsEngine
except ImportError:
    # Fall back to absolute imports (when run directly)
    try:
        from core.memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
        from core.btsp import BTSPUpdateMechanism
        from core.lifecycle import MemoryLifecycleManager
        from core.dynamics import MultiTimescaleDynamicsEngine
    except ImportError:
        # Final fallback - assume they're in the same directory
        from memory_trace import MemoryTrace, TemporalMetadata, ConsolidationState
        from btsp import BTSPUpdateMechanism
        from lifecycle import MemoryLifecycleManager
        from dynamics import MultiTimescaleDynamicsEngine

logger = logging.getLogger(__name__)


class EnhancedDistributedMemoryNode(nn.Module):
    """Production-ready DMN with all optimizations, biological features, and temporal dynamics integration"""

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

        # Initialize core components (PRESERVED)
        self._init_memory_storage()
        self._init_weight_matrices()
        self._init_biological_mechanisms()
        self._init_indexing_system()
        self._init_performance_tracking()

        # NEW: Initialize temporal integration
        self._init_temporal_integration()

        # Thread-safe operations (PRESERVED)
        self.lock = threading.RLock()
        self.async_executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Enhanced DMN {node_id} initialized: {dimension}D, capacity={capacity}, "
            f"specialization={specialization}, temporal_enabled={self.temporal_enabled}")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration parameters (ENHANCED with temporal settings)"""
        base_config = {
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
        
        # NEW: Temporal configuration
        base_config['temporal'] = {
            'enabled': True,
            'injection_rate': 0.1,
            'consolidation_threshold': 0.7,
            'temporal_context_weight': 0.3,
            'age_category_weights': {
                'fast_synaptic': 1.2,
                'calcium_plasticity': 1.0,
                'protein_synthesis': 0.8,
                'homeostatic_scaling': 0.6,
                'systems_consolidation': 0.4
            },
            'consolidation_state_weights': {
                'INITIAL': 1.0,
                'CONSOLIDATING': 1.1,
                'CONSOLIDATED': 1.2,
                'STABLE': 1.3
            },
            'event_injection_thresholds': {
                'high_salience_threshold': 0.8,
                'consolidation_ready_threshold': 0.7,
                'temporal_activity_threshold': 0.6
            }
        }
        
        return base_config

    def _init_memory_storage(self):
        """Initialize memory storage structures (PRESERVED)"""
        self.memory_traces: List[MemoryTrace] = []
        self.trace_index: Dict[str, int] = {}  # trace_id -> list index
        self.specialization_counts: Dict[str, int] = defaultdict(int)
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def _init_weight_matrices(self):
        """Initialize fast and slow weight matrices (PRESERVED)"""
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
        """Initialize biological mechanisms with proper error handling"""
        try:
            # Initialize BTSP mechanism
            self.btsp_mechanism = BTSPUpdateMechanism(
                **self.config['btsp']
            )
            
            # Initialize dynamics engine - only accepts config parameter
            dynamics_config = self.config.get('dynamics', {})
            self.dynamics_engine = MultiTimescaleDynamicsEngine(config=dynamics_config)
            
            # Initialize lifecycle manager with proper imports
            try:
                # Try to import LifecycleConfig
                try:
                    from .lifecycle import LifecycleConfig
                except ImportError:
                    try:
                        from core.lifecycle import LifecycleConfig
                    except ImportError:
                        from lifecycle import LifecycleConfig
                        
                lifecycle_config_dict = self.config.get('lifecycle', {})
                lifecycle_config = LifecycleConfig(**lifecycle_config_dict)
                
            except ImportError:
                # If LifecycleConfig is not available, create a simple config object
                logger.warning(f"DMN {self.node_id}: LifecycleConfig not found, using simple config")
                class SimpleLifecycleConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                            
                lifecycle_config_dict = self.config.get('lifecycle', {})
                lifecycle_config = SimpleLifecycleConfig(**lifecycle_config_dict)
            
            # Create lifecycle manager
            self.lifecycle_manager = MemoryLifecycleManager(
                node_id=self.node_id,
                config=lifecycle_config,
                dynamics_engine=self.dynamics_engine
            )
            
            logger.info(f"DMN {self.node_id}: Biological mechanisms initialized successfully")
            
        except Exception as e:
            logger.error(f"DMN {self.node_id}: Error initializing biological mechanisms: {e}")
            # Create fallback mechanisms
            self._init_fallback_biological_mechanisms()

    def _init_fallback_biological_mechanisms(self):
        """Initialize fallback biological mechanisms if main initialization fails"""
        logger.warning(f"DMN {self.node_id}: Using fallback biological mechanisms")
        
        # Simple BTSP fallback
        class SimpleBTSP:
            def __init__(self):
                pass
                
            async def should_update_async(self, input_state, existing_traces, context, user_feedback=None):
                class SimpleDecision:
                    def __init__(self):
                        self.should_update = True
                        self.calcium_level = 0.8
                        self.learning_rate = 0.1
                        self.novelty_score = 0.5
                        self.importance_score = 0.5
                        self.error_score = 0.5
                        self.confidence = 0.8
                return SimpleDecision()
                
            def get_stats(self):
                return {"fallback_btsp": True}
        
        # Simple dynamics engine fallback
        class SimpleDynamics:
            def __init__(self):
                pass
                
            async def process_update_async(self, content, salience, calcium_level):
                return True
                
            async def process_retrieval_async(self, query, traces):
                return True
                
            def get_stats(self):
                return {"fallback_dynamics": True}
        
        # Simple lifecycle manager fallback
        class SimpleLifecycle:
            def __init__(self, node_id, config=None, dynamics_engine=None):
                self.node_id = node_id
                
            async def evaluate_trace_lifecycle(self, trace, current_time):
                from enum import Enum
                class LifecycleState(Enum):
                    ACTIVE = "active"
                return LifecycleState.ACTIVE
                
            async def select_eviction_candidates(self, traces, num_to_evict=5):
                return traces[:min(num_to_evict, len(traces))]
                
            async def perform_maintenance_cycle(self, traces):
                return {"fallback_maintenance": True}
                
            def get_lifecycle_statistics(self):
                return {"fallback_lifecycle": True}
        
        self.btsp_mechanism = SimpleBTSP()
        self.dynamics_engine = SimpleDynamics()
        self.lifecycle_manager = SimpleLifecycle(self.node_id)

    def _init_indexing_system(self):
        """Initialize FAISS indexing with optimizations (PRESERVED)"""
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
        """Initialize performance and statistics tracking (ENHANCED with temporal metrics)"""
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

        # NEW: Temporal statistics
        self.temporal_stats = {
            'temporal_events_injected': 0,
            'temporal_consolidations': 0,
            'temporal_modulations_applied': 0,
            'age_category_distribution': defaultdict(int),
            'consolidation_state_distribution': defaultdict(int),
            'temporal_coherence_history': deque(maxlen=100),
            'temporal_priority_distribution': defaultdict(int),
            'cross_timescale_activations': 0
        }

        self.performance_history = deque(maxlen=1000)
        self.last_stats_update = time.time()

    def _init_temporal_integration(self):
        """Initialize temporal dynamics integration (NEW)"""
        temporal_config = self.config.get('temporal', {})
        
        self.temporal_enabled = temporal_config.get('enabled', True)
        self.temporal_injection_rate = temporal_config.get('injection_rate', 0.1)
        self.temporal_consolidation_threshold = temporal_config.get('consolidation_threshold', 0.7)
        self.temporal_context_weight = temporal_config.get('temporal_context_weight', 0.3)
        
        # Temporal weighting schemes
        self.age_category_weights = temporal_config.get('age_category_weights', {})
        self.consolidation_state_weights = temporal_config.get('consolidation_state_weights', {})
        self.event_injection_thresholds = temporal_config.get('event_injection_thresholds', {})
        
        # Temporal integration state
        self.external_temporal_engine: Optional['MultiTimescaleDynamicsEngine'] = None
        self.temporal_context_cache: Dict[str, Any] = {}
        self.last_temporal_update = time.time()
        
        # Event injection tracking
        self.temporal_event_queue = asyncio.Queue() if self.temporal_enabled else None
        self.temporal_injection_history = deque(maxlen=1000) if self.temporal_enabled else None

    # Rest of the methods would go here...

    # NEW: Temporal integration methods
    async def integrate_temporal_dynamics(self, temporal_engine: 'MultiTimescaleDynamicsEngine') -> None:
        """Integrate with external multi-timescale dynamics engine."""
        if not self.temporal_enabled:
            logger.warning(f"DMN {self.node_id}: Temporal integration disabled")
            return
            
        self.external_temporal_engine = temporal_engine
        
        # Register this DMN with the temporal engine
        if hasattr(temporal_engine, 'integrate_with_dmn'):
            temporal_engine.integrate_with_dmn(self.node_id, self)
            
        logger.info(f"DMN {self.node_id} integrated with temporal dynamics engine")

    async def update_temporal_context(self, temporal_context: Dict[str, Any]) -> None:
        """Update temporal context from external temporal engine."""
        if not self.temporal_enabled:
            return
            
        self.temporal_context_cache = temporal_context.copy()
        self.last_temporal_update = time.time()
        
        # Update temporal coherence history
        coherence = temporal_context.get('temporal_coherence', 1.0)
        self.temporal_stats['temporal_coherence_history'].append(coherence)
        
        # Update memory traces with new temporal context
        await self._update_traces_temporal_state(temporal_context)

    async def _update_traces_temporal_state(self, temporal_context: Dict[str, Any]) -> None:
        """Update temporal state for all memory traces."""
        if not self.temporal_enabled:
            return
            
        current_time = time.time()
        updated_count = 0
        
        for trace in self.memory_traces:
            # Update temporal metadata
            trace.update_temporal_state(temporal_context)
            
            # Update age category distribution
            age_category = trace.get_temporal_age_category()
            self.temporal_stats['age_category_distribution'][age_category] += 1
            
            # Update consolidation state distribution
            consolidation_state = trace.temporal_metadata.consolidation_state.value
            self.temporal_stats['consolidation_state_distribution'][consolidation_state] += 1
            
            updated_count += 1
            
        logger.debug(f"DMN {self.node_id}: Updated temporal state for {updated_count} traces")

    async def inject_temporal_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Inject temporal events into external temporal engine."""
        if not self.temporal_enabled or not self.external_temporal_engine:
            return False
            
        try:
            # Map DMN events to temporal engine events
            if event_type == "high_salience_storage":
                success = await self.external_temporal_engine.inject_event(
                    "memory_trace_activation",
                    "fast_synaptic",
                    {
                        **event_data,
                        "source_dmn": self.node_id,
                        "injection_time": time.time()
                    }
                )
                
            elif event_type == "consolidation_ready":
                success = await self.external_temporal_engine.inject_event(
                    "consolidation_request",
                    "protein_synthesis", 
                    {
                        **event_data,
                        "source_dmn": self.node_id,
                        "injection_time": time.time()
                    }
                )
                
            elif event_type == "retrieval_pattern":
                success = await self.external_temporal_engine.inject_event(
                    "retrieval_activity",
                    "calcium_plasticity",
                    {
                        **event_data,
                        "source_dmn": self.node_id,
                        "injection_time": time.time()
                    }
                )
            else:
                logger.warning(f"DMN {self.node_id}: Unknown temporal event type: {event_type}")
                return False
                
            if success:
                self.temporal_stats['temporal_events_injected'] += 1
                self.temporal_injection_history.append({
                    'event_type': event_type,
                    'timestamp': time.time(),
                    'data': event_data
                })
                
            return success
            
        except Exception as e:
            logger.error(f"DMN {self.node_id}: Temporal event injection failed: {e}")
            return False

    async def add_memory_trace_async(self,
                                 content: torch.Tensor,
                                 context: Dict[str, Any],
                                 salience: float,
                                 user_feedback: Optional[Dict[str, Any]] = None) -> bool:
    
    
        current_time = time.time()
        
        # DEVICE FIX: Store content on CPU for FAISS compatibility but keep original device reference
        original_device = content.device
        content_cpu = content.detach().cpu()

        # Create memory trace with CPU content
        trace = MemoryTrace(
            content=content_cpu,
            context=context.copy(),
            timestamp=current_time,
            salience=salience,
            creation_node=self.node_id
        )

        if self.temporal_enabled and self.temporal_context_cache:
            trace.update_temporal_state(self.temporal_context_cache)

        # DEVICE FIX: Use content on the device expected by BTSP
        try:
            # For BTSP, we need the content on the processing device (usually GPU)
            content_for_btsp = content.to(self.device) if self.device != 'cpu' else content_cpu
            update_decision = await self.btsp_mechanism.should_update_async(
                input_state=content_for_btsp,
                existing_traces=self.memory_traces[-50:],
                context=context,
                user_feedback=user_feedback
            )
        except Exception as e:
            logger.error(f"DMN {self.node_id}: BTSP evaluation failed: {e}")
            class SimpleDecision:
                def __init__(self):
                    self.should_update = True
                    self.calcium_level = 0.8
                    self.learning_rate = 0.1
            update_decision = SimpleDecision()

        if not update_decision.should_update:
            logger.debug(f"DMN {self.node_id}: Update rejected for trace {trace.trace_id}")
            return False

        if len(self.memory_traces) >= self.capacity:
            await self._intelligent_eviction_async()

        with self.lock:
            self.memory_traces.append(trace)
            self.trace_index[trace.trace_id] = len(self.memory_traces) - 1
            specialization = context.get('domain', 'general')
            self.specialization_counts[specialization] += 1

        # DEVICE FIX: Add to FAISS using CPU content
        await self._add_to_index_async(trace)

        # DEVICE FIX: Update dynamics with CPU content
        try:
            await self.dynamics_engine.process_update_async(
                content_cpu, salience, update_decision.calcium_level
            )
        except Exception as e:
            logger.debug(f"DMN {self.node_id}: Dynamics engine update failed: {e}")

        # DEVICE FIX: Update fast weights with device content
        try:
            content_for_weights = content.to(self.device) if self.device != 'cpu' else content_cpu
            await self._update_fast_weights_async(
                content_for_weights, salience, update_decision.learning_rate
            )
        except Exception as e:
            logger.debug(f"DMN {self.node_id}: Fast weights update failed: {e}")

        if self.temporal_enabled:
            try:
                await self._handle_temporal_events_on_storage(trace, update_decision)
            except Exception as e:
                logger.debug(f"DMN {self.node_id}: Temporal event injection failed: {e}")

        # Update statistics
        self.stats['total_traces'] += 1
        self.stats['total_updates'] += 1
        
        if self.temporal_enabled:
            age_category = trace.get_temporal_age_category()
            self.temporal_stats['age_category_distribution'][age_category] += 1

        logger.debug(f"DMN {self.node_id}: Added trace {trace.trace_id}")
        return True
        class SimpleDecision:
                def __init__(self):
                    self.should_update = True
                    self.calcium_level = 0.8
                    self.learning_rate = 0.1
                    update_decision = SimpleDecision()

        if not update_decision.should_update:
            logger.debug(f"DMN {self.node_id}: Update rejected for trace {trace.trace_id}")
            return False

        # Check capacity and evict if necessary (PRESERVED)
        if len(self.memory_traces) >= self.capacity:
            await self._intelligent_eviction_async()

        # Add trace to storage (PRESERVED)
        with self.lock:
            self.memory_traces.append(trace)
            self.trace_index[trace.trace_id] = len(self.memory_traces) - 1

            # Update specialization tracking
            specialization = context.get('domain', 'general')
            self.specialization_counts[specialization] += 1

        # Add to FAISS index (PRESERVED) - USE CPU VERSION
        await self._add_to_index_async(trace)

        # Update biological dynamics (PRESERVED) - DEVICE CONSISTENCY
        try:
            await self.dynamics_engine.process_update_async(
                content_cpu, salience, update_decision.calcium_level
            )
        except Exception as e:
            logger.debug(f"DMN {self.node_id}: Dynamics engine update failed: {e}")

        # Update fast weights (PRESERVED) - DEVICE CONSISTENCY
        try:
            await self._update_fast_weights_async(
                content.to(self.device), salience, update_decision.learning_rate
            )
        except Exception as e:
            logger.debug(f"DMN {self.node_id}: Fast weights update failed: {e}")

        # NEW: Temporal event injection
        if self.temporal_enabled:
            try:
                await self._handle_temporal_events_on_storage(trace, update_decision)
            except Exception as e:
                logger.debug(f"DMN {self.node_id}: Temporal event injection failed: {e}")

        # Update statistics (ENHANCED)
        self.stats['total_traces'] += 1
        self.stats['total_updates'] += 1
        
        if self.temporal_enabled:
            age_category = trace.get_temporal_age_category()
            self.temporal_stats['age_category_distribution'][age_category] += 1

        logger.debug(f"DMN {self.node_id}: Added trace {trace.trace_id}")
        return True
    async def _handle_temporal_events_on_storage(self, trace: MemoryTrace, update_decision) -> None:
        """Handle temporal events during memory storage (NEW)"""
        
        # High salience traces trigger fast synaptic events
        if trace.salience >= self.event_injection_thresholds.get('high_salience_threshold', 0.8):
            await self.inject_temporal_event(
                "high_salience_storage",
                {
                    "trace_data": {
                        "content": trace.content.numpy().tolist(),
                        "trace_id": trace.trace_id,
                        "salience": trace.salience
                    },
                    "update_decision": {
                        "calcium_level": update_decision.calcium_level,
                        "learning_rate": update_decision.learning_rate
                    }
                }
            )
            
        # Traces ready for consolidation
        if trace.should_consolidate(self.temporal_consolidation_threshold):
            await self.inject_temporal_event(
                "consolidation_ready",
                {
                    "memory_data": {
                        "trace_id": trace.trace_id,
                        "content": trace.content.numpy().tolist(),
                        "salience": trace.salience,
                        "context": trace.context,
                        "consolidation_strength": trace.temporal_metadata.consolidation_strength
                    },
                    "priority": trace.get_temporal_priority()
                }
            )

    async def retrieve_memories_async(self,
                                  query: torch.Tensor,
                                  k: int = 10,
                                  context_filter: Optional[Dict[str, Any]] = None,
                                  similarity_threshold: float = None,
                                  temporal_context: Optional[Dict[str, Any]] = None) -> List[Tuple[MemoryTrace, float]]:
    
    
        logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Starting retrieval. Index total: {self.index.ntotal}")
        
        if self.index.ntotal == 0:
            logger.warning(f"⚠️ DMN {self.node_id}: Index is empty, returning no results")
            return []

        if similarity_threshold is None:
            similarity_threshold = self.config['indexing']['similarity_threshold']
        
        logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Using similarity threshold: {similarity_threshold}")

        if temporal_context is None and self.temporal_enabled:
            temporal_context = self.temporal_context_cache

        # Prepare normalized query - always work with CPU for FAISS
        original_device = query.device
        query_cpu = query.detach().cpu()
        query_normalized = F.normalize(query_cpu, dim=0)
        query_np = query_normalized.numpy().astype('float32').reshape(1, -1)
        
        logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Query shape: {query_np.shape}")

        try:
            # Search more than k to allow for filtering
            search_k = min(k * 3, self.index.ntotal)
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Searching for top {search_k} results")
            
            similarities, faiss_ids = self.index.search(query_np, search_k)
            
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: FAISS returned {len(similarities[0])} results")
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Top similarities: {similarities[0][:3]}")
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Top FAISS IDs: {faiss_ids[0][:3]}")

            # Process results
            results = []
            current_time = time.time()

            for i, (sim, faiss_id) in enumerate(zip(similarities[0], faiss_ids[0])):
                logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Processing result {i}: sim={sim:.4f}, faiss_id={faiss_id}")
                
                if sim < similarity_threshold:
                    logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Similarity {sim:.4f} below threshold {similarity_threshold}")
                    continue
                    
                if faiss_id not in self.faiss_id_to_trace_id:
                    logger.warning(f"⚠️ DMN {self.node_id}: FAISS ID {faiss_id} not found in mapping")
                    continue

                trace_id = self.faiss_id_to_trace_id[faiss_id]
                if trace_id not in self.trace_index:
                    logger.warning(f"⚠️ DMN {self.node_id}: Trace ID {trace_id} not found in trace index")
                    continue

                trace_idx = self.trace_index[trace_id]
                if trace_idx >= len(self.memory_traces):
                    logger.warning(f"⚠️ DMN {self.node_id}: Trace index {trace_idx} out of bounds")
                    continue

                trace = self.memory_traces[trace_idx]

                # Apply context filtering
                if context_filter and not self._matches_context_filter(trace, context_filter):
                    logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Trace {trace_id} filtered out by context")
                    continue

                # Apply temporal modulation
                if self.temporal_enabled and temporal_context:
                    modulated_sim = self._apply_temporal_modulation(sim, trace, temporal_context)
                else:
                    modulated_sim = sim

                # Update access statistics
                trace.update_access_stats(current_time, context_relevant=True)
                self.access_patterns[trace_id].append(current_time)

                results.append((trace, float(modulated_sim)))
                logger.debug(f"✅ DMN {self.node_id}: Added result: trace_id={trace_id}, similarity={modulated_sim:.4f}")

                if len(results) >= k:
                    break

            # Sort by modulated similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Returning {len(results)} results")

            # Update dynamics engine
            try:
                await self.dynamics_engine.process_retrieval_async(
                    query_cpu, [trace for trace, _ in results]
                )
            except Exception as e:
                logger.debug(f"DMN {self.node_id}: Dynamics engine retrieval processing failed: {e}")

            # Inject temporal events
            if self.temporal_enabled and results:
                try:
                    await self._handle_temporal_events_on_retrieval(query_cpu, results, temporal_context)
                except Exception as e:
                    logger.debug(f"DMN {self.node_id}: Temporal event injection failed: {e}")

            self.stats['total_retrievals'] += 1
            return results

        except Exception as e:
            logger.error(f"❌ DMN {self.node_id}: Retrieval error: {e}")
            import traceback
            logger.error(f"❌ DMN {self.node_id}: Full traceback:\n{traceback.format_exc()}")
            return []

    def _apply_temporal_modulation(self, base_similarity: float, 
                                 trace: MemoryTrace, 
                                 temporal_context: Dict[str, Any]) -> float:
        """Apply temporal modulation to similarity scores (NEW)"""
        
        modulated_similarity = base_similarity
        
        # Age category modulation
        age_category = trace.get_temporal_age_category()
        age_weight = self.age_category_weights.get(age_category, 1.0)
        modulated_similarity *= age_weight
        
        # Consolidation state modulation
        consolidation_state = trace.temporal_metadata.consolidation_state.value
        consolidation_weight = self.consolidation_state_weights.get(consolidation_state, 1.0)
        modulated_similarity *= consolidation_weight
        
        # Fast synaptic activity bias
        if "fast_synaptic_activity" in temporal_context:
            fast_activity = temporal_context["fast_synaptic_activity"]
            trace_age = time.time() - trace.timestamp
            
            # Boost recent traces when fast synaptic activity is high
            if trace_age < 60.0 and fast_activity > 0.7:  # Last minute, high activity
                modulated_similarity *= 1.2
                
        # Consolidation activity bias
        if "consolidation_activity" in temporal_context:
            consol_activity = temporal_context["consolidation_activity"]
            
            # Boost consolidated traces when consolidation activity is high
            if (trace.temporal_metadata.consolidation_state == ConsolidationState.CONSOLIDATED and
                consol_activity > 0.6):
                modulated_similarity *= 1.1
                
        # Temporal coherence modulation
        temporal_coherence = temporal_context.get("temporal_coherence", 1.0)
        trace_coherence = trace.temporal_metadata.temporal_coherence
        
        # Boost traces with coherence matching system state
        coherence_match = 1.0 - abs(temporal_coherence - trace_coherence)
        coherence_factor = 0.9 + 0.2 * coherence_match  # 0.9-1.1 range
        modulated_similarity *= coherence_factor
        
        # Apply temporal context weight
        final_similarity = (
            base_similarity * (1 - self.temporal_context_weight) +
            modulated_similarity * self.temporal_context_weight
        )
        
        # Track temporal modulation application
        if abs(final_similarity - base_similarity) > 0.01:
            self.temporal_stats['temporal_modulations_applied'] += 1
        
        return final_similarity

    async def _handle_temporal_events_on_retrieval(self, query: torch.Tensor,
                                                  results: List[Tuple[MemoryTrace, float]],
                                                  temporal_context: Optional[Dict[str, Any]]) -> None:
        """Handle temporal events during memory retrieval (NEW)"""
        
        if not results:
            return
            
        # Inject retrieval pattern event
        retrieval_strength = np.mean([similarity for _, similarity in results])
        
        if retrieval_strength >= self.event_injection_thresholds.get('temporal_activity_threshold', 0.6):
            await self.inject_temporal_event(
                "retrieval_pattern",
                {
                    "query_info": {
                        "query_norm": float(torch.norm(query)),
                        "retrieval_strength": float(retrieval_strength),
                        "num_results": len(results)
                    },
                    "retrieved_traces": [
                        {
                            "trace_id": trace.trace_id,
                            "similarity": float(similarity),
                            "age_category": trace.get_temporal_age_category(),
                            "consolidation_state": trace.temporal_metadata.consolidation_state.value
                        }
                        for trace, similarity in results[:5]  # Top 5 traces
                    ],
                    "temporal_context": temporal_context
                }
            )

    def _matches_context_filter(self, trace: MemoryTrace, context_filter: Dict[str, Any]) -> bool:
        """Check if trace matches context filter criteria (PRESERVED)"""
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
    
    
        try:
            # Always use CPU for FAISS operations
            content_cpu = trace.content.detach().cpu()
            content_normalized = F.normalize(content_cpu, dim=0)
            content_np = content_normalized.numpy().astype('float32').reshape(1, -1)

            # Assign FAISS ID
            faiss_id = self.next_faiss_id
            self.next_faiss_id += 1

            # Add to index
            self.index.add_with_ids(content_np, np.array([faiss_id], dtype=np.int64))

            # Update mapping
            self.faiss_id_to_trace_id[faiss_id] = trace.trace_id
            
            # Verify the addition
            current_total = self.index.ntotal
            logger.debug(f"🔍 DEBUG: DMN {self.node_id}: Added trace to index. Total indexed: {current_total}")
            
            # Verify we can search for what we just added
            if current_total > 0:
                similarities, found_ids = self.index.search(content_np, 1)
                if len(found_ids[0]) > 0 and found_ids[0][0] == faiss_id:
                    logger.debug(f"✅ DMN {self.node_id}: Successfully verified trace {trace.trace_id} in index")
                else:
                    logger.warning(f"⚠️ DMN {self.node_id}: Could not verify trace {trace.trace_id} in index immediately after adding")
                    
        except Exception as e:
            logger.error(f"❌ DMN {self.node_id}: Failed to add trace to FAISS index: {e}")
            # Don't fail the entire operation if indexing fails, but log it prominently
            logger.error(f"❌ DMN {self.node_id}: FAISS indexing failed")

    async def _intelligent_eviction_async(self):
        
        if not self.memory_traces:
            return

        current_time = time.time()
        eviction_batch_size = self.config['lifecycle']['eviction_batch_size']

        # Compute eviction scores for all traces (ENHANCED)
        scored_traces = []
        for i, trace in enumerate(self.memory_traces):
            # Base decay score (PRESERVED)
            base_score = trace.compute_decay_score(current_time, self.config['lifecycle']['decay_params'])
            
            # NEW: Apply temporal adjustments to eviction score
            if self.temporal_enabled:
                temporal_score_adjustment = self._compute_temporal_eviction_adjustment(trace)
                final_score = base_score * temporal_score_adjustment
            else:
                final_score = base_score
                
            scored_traces.append((final_score, i, trace))

        # Sort by score (lowest first = highest eviction priority)
        scored_traces.sort(key=lambda x: x[0])

        # Select traces for eviction (lowest scores)
        traces_to_evict = scored_traces[:eviction_batch_size]

        # Remove from FAISS index (PRESERVED)
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

        # Remove from memory traces list (in reverse order to maintain indices) (PRESERVED)
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

    def _compute_temporal_eviction_adjustment(self, trace: MemoryTrace) -> float:
        """Compute temporal adjustment factor for eviction scoring (NEW)"""
        
        adjustment_factor = 1.0
        
        # Consolidation state protection
        consolidation_state = trace.temporal_metadata.consolidation_state
        if consolidation_state == ConsolidationState.CONSOLIDATED:
            adjustment_factor *= 1.5  # Less likely to be evicted
        elif consolidation_state == ConsolidationState.STABLE:
            adjustment_factor *= 2.0  # Much less likely to be evicted
        elif consolidation_state == ConsolidationState.CONSOLIDATING:
            adjustment_factor *= 1.2  # Slightly protected
            
        # Temporal coherence protection
        temporal_coherence = trace.temporal_metadata.temporal_coherence
        if temporal_coherence > 0.8:
            adjustment_factor *= 1.3  # High coherence traces are protected
            
        # Age category consideration
        age_category = trace.get_temporal_age_category()
        if age_category == "systems_consolidation":
            adjustment_factor *= 1.4  # Long-term memories are protected
        elif age_category == "fast_synaptic":
            adjustment_factor *= 0.9  # Recent memories slightly more evictable
            
        # Consolidation strength protection
        consolidation_strength = trace.temporal_metadata.consolidation_strength
        if consolidation_strength > 0.7:
            adjustment_factor *= 1.2
            
        return adjustment_factor

    async def _update_fast_weights_async(self, content: torch.Tensor, salience: float, learning_rate: float):
        """Update fast weights using biological-inspired plasticity (PRESERVED)"""

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
        """Consolidate important memories from fast to slow weights (ENHANCED with temporal awareness)"""

        current_time = time.time()
        consolidation_config = self.config['consolidation']

        # Find traces eligible for consolidation (ENHANCED)
        eligible_traces = []
        for trace in self.memory_traces:
            # Original criteria (PRESERVED)
            basic_eligibility = (trace.current_salience >= consolidation_config['threshold'] or
                               trace.access_count >= 5)
            
            # NEW: Temporal criteria
            temporal_eligibility = False
            if self.temporal_enabled:
                temporal_eligibility = (
                    trace.should_consolidate(self.temporal_consolidation_threshold) or
                    trace.temporal_metadata.consolidation_state in [ConsolidationState.CONSOLIDATING, ConsolidationState.CONSOLIDATED]
                )
            
            if basic_eligibility or temporal_eligibility:
                eligible_traces.append(trace)

        if not eligible_traces:
            return

        # Limit consolidation batch size
        max_traces = consolidation_config['max_traces_per_cycle']
        if len(eligible_traces) > max_traces:
            # Sort by importance and temporal priority (ENHANCED)
            def consolidation_priority(trace):
                base_priority = trace.current_salience * np.log1p(trace.access_count)
                if self.temporal_enabled:
                    temporal_priority = trace.get_temporal_priority() / 10.0  # Normalize to [0,1]
                    return base_priority + temporal_priority
                return base_priority
                
            eligible_traces.sort(key=consolidation_priority, reverse=True)
            eligible_traces = eligible_traces[:max_traces]

        # Compute consolidated representation (PRESERVED)
        consolidated_vectors = []
        weights = []

        for trace in eligible_traces:
            content = trace.content.to(self.device)
            weight = trace.current_salience * np.log1p(trace.access_count)
            
            # NEW: Apply temporal weighting
            if self.temporal_enabled:
                temporal_weight = trace.get_temporal_priority() / 10.0
                weight = weight * (0.7 + 0.3 * temporal_weight)
                
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

            # Update consolidation level of traces (ENHANCED)
            for trace in eligible_traces:
                trace.consolidation_level = min(2, trace.consolidation_level + 1)
                
                # NEW: Update temporal consolidation state
                if self.temporal_enabled:
                    if trace.temporal_metadata.consolidation_state == ConsolidationState.CONSOLIDATING:
                        trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATED

            self.stats['consolidations'] += 1
            
            # NEW: Update temporal consolidation stats
            if self.temporal_enabled:
                self.temporal_stats['temporal_consolidations'] += 1
                
            logger.info(f"DMN {self.node_id}: Consolidated {len(eligible_traces)} traces to slot {oldest_slot}")

    # NEW: Temporal-specific methods
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive temporal statistics (NEW)"""
        if not self.temporal_enabled:
            return {"temporal_enabled": False}
            
        current_time = time.time()
        
        # Temporal coherence statistics
        coherence_history = list(self.temporal_stats['temporal_coherence_history'])
        
        # Age category distribution
        total_traces = sum(self.temporal_stats['age_category_distribution'].values())
        age_category_percentages = {
            category: (count / max(1, total_traces)) * 100
            for category, count in self.temporal_stats['age_category_distribution'].items()
        }
        
        # Consolidation state distribution
        consolidation_percentages = {
            state: (count / max(1, total_traces)) * 100
            for state, count in self.temporal_stats['consolidation_state_distribution'].items()
        }
        
        return {
            "temporal_enabled": True,
            "temporal_engine_integrated": self.external_temporal_engine is not None,
            "last_temporal_update": self.last_temporal_update,
            "time_since_temporal_update": current_time - self.last_temporal_update,
            "temporal_events_injected": self.temporal_stats['temporal_events_injected'],
            "temporal_consolidations": self.temporal_stats['temporal_consolidations'],
            "temporal_modulations_applied": self.temporal_stats['temporal_modulations_applied'],
            "cross_timescale_activations": self.temporal_stats['cross_timescale_activations'],
            "age_category_distribution": age_category_percentages,
            "consolidation_state_distribution": consolidation_percentages,
            "temporal_coherence_stats": {
                "current": coherence_history[-1] if coherence_history else 1.0,
                "average": np.mean(coherence_history) if coherence_history else 1.0,
                "std": np.std(coherence_history) if coherence_history else 0.0,
                "min": np.min(coherence_history) if coherence_history else 1.0,
                "max": np.max(coherence_history) if coherence_history else 1.0
            },
            "temporal_priority_distribution": dict(self.temporal_stats['temporal_priority_distribution']),
            "recent_injection_events": list(self.temporal_injection_history)[-10:] if self.temporal_injection_history else []
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive node statistics (ENHANCED with temporal data)"""
        current_time = time.time()

        # Update derived statistics (PRESERVED)
        if self.memory_traces:
            avg_access = sum(trace.access_count for trace in self.memory_traces) / len(self.memory_traces)
            self.stats['average_access_count'] = avg_access

        self.stats['memory_utilization'] = len(self.memory_traces) / self.capacity
        self.stats['specialization_distribution'] = dict(self.specialization_counts)

        # Add dynamics engine stats (PRESERVED)
        dynamics_stats = self.dynamics_engine.get_stats()

        # Add biological mechanism stats (PRESERVED)
        btsp_stats = self.btsp_mechanism.get_stats()

        # NEW: Add temporal statistics
        temporal_stats = self.get_temporal_statistics()

        return {
            'node_id': self.node_id,
            'specialization': self.specialization,
            'timestamp': current_time,
            'basic_stats': self.stats,
            'temporal_stats': temporal_stats,  # NEW
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
        """Save complete node state (ENHANCED with temporal data)"""
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
        
        # NEW: Add temporal state
        if self.temporal_enabled:
            checkpoint['temporal_stats'] = self.temporal_stats
            checkpoint['temporal_context_cache'] = self.temporal_context_cache
            checkpoint['last_temporal_update'] = self.last_temporal_update
            checkpoint['temporal_injection_history'] = list(self.temporal_injection_history) if self.temporal_injection_history else []

        torch.save(checkpoint, filepath)
        logger.info(f"DMN {self.node_id}: Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load complete node state (ENHANCED with temporal data)"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore parameters (PRESERVED)
        self.fast_weights.data = checkpoint['fast_weights'].to(self.device)
        self.slow_weights.data = checkpoint['slow_weights'].to(self.device)
        self.weight_usage = checkpoint['weight_usage'].to(self.device)
        self.last_weight_update = checkpoint['last_weight_update'].to(self.device)

        # Restore memory traces (PRESERVED)
        self.memory_traces = []
        for trace_data in checkpoint['memory_traces']:
            trace = MemoryTrace.from_dict(trace_data, self.device)
            self.memory_traces.append(trace)

        # Restore other state (PRESERVED)
        self.trace_index = checkpoint['trace_index']
        self.specialization_counts = defaultdict(int, checkpoint['specialization_counts'])
        self.stats = checkpoint['stats']
        self.faiss_id_to_trace_id = checkpoint['faiss_id_to_trace_id']
        self.next_faiss_id = checkpoint['next_faiss_id']

        # NEW: Restore temporal state
        if self.temporal_enabled and 'temporal_stats' in checkpoint:
            self.temporal_stats = checkpoint['temporal_stats']
            self.temporal_context_cache = checkpoint.get('temporal_context_cache', {})
            self.last_temporal_update = checkpoint.get('last_temporal_update', time.time())
            
            temporal_injection_data = checkpoint.get('temporal_injection_history', [])
            if self.temporal_injection_history is not None:
                self.temporal_injection_history.extend(temporal_injection_data)

        # Rebuild FAISS index (PRESERVED)
        self._rebuild_faiss_index()

        logger.info(f"DMN {self.node_id}: Loaded checkpoint from {filepath}")

    def _rebuild_faiss_index(self):
    
        # Clear existing index
        self.index.reset()

        if self.memory_traces:
            # Prepare all vectors and IDs - USE CPU
            vectors = []
            faiss_ids = []

            # Rebuild mapping
            new_faiss_id_to_trace_id = {}
            new_faiss_id = 0

            for trace in self.memory_traces:
                try:
                    content_cpu = trace.content.detach().cpu()
                    content_normalized = F.normalize(content_cpu, dim=0)
                    vectors.append(content_normalized.detach().cpu().numpy())

                    faiss_ids.append(new_faiss_id)
                    new_faiss_id_to_trace_id[new_faiss_id] = trace.trace_id
                    new_faiss_id += 1
                except Exception as e:
                    logger.warning(f"DMN {self.node_id}: Failed to process trace {trace.trace_id} for index rebuild: {e}")
                    continue

            if vectors:
                try:
                    # Add all vectors at once
                    vectors_np = np.vstack(vectors).astype('float32')
                    faiss_ids_np = np.array(faiss_ids, dtype=np.int64)

                    self.index.add_with_ids(vectors_np, faiss_ids_np)

                    # Update mappings
                    self.faiss_id_to_trace_id = new_faiss_id_to_trace_id
                    self.next_faiss_id = new_faiss_id

                    logger.info(f"DMN {self.node_id}: Rebuilt FAISS index with {len(vectors)} traces")
                except Exception as e:
                    logger.error(f"DMN {self.node_id}: Failed to rebuild FAISS index: {e}")
            else:
                logger.warning(f"DMN {self.node_id}: No valid vectors for index rebuild")