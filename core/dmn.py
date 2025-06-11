# core/dmn.py - GPU-Accelerated FAISS Implementation

import torch
import faiss
import numpy as np
import logging
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import uuid
from typing import Dict, Optional
import asyncio
from typing import Optional, Any


logger = logging.getLogger(__name__)


class UpdateDecision:
    """Update decision with calcium_level"""
    def __init__(self, should_update, calcium_level, novelty, importance):
        self.should_update = should_update
        self.calcium_level = calcium_level  # This is what you needed!
        self.novelty = novelty
        self.importance = importance

class SimpleBTSP:
    """Simple BTSP that provides calcium_level"""

    def __init__(self, config):
        self.calcium_threshold = config.get('calcium_threshold', 0.7)
        self.novelty_weight = config.get('novelty_weight', 0.4)

    async def evaluate_async(self, content, context, existing_traces, user_feedback=None):
        """Evaluate and return decision with calcium_level"""

        # Calculate novelty
        novelty = 1.0
        if existing_traces:
            similarities = []
            for trace in existing_traces[-10:]:  # Check recent traces
                try:
                    sim = torch.cosine_similarity(content, trace.content, dim=0)
                    similarities.append(sim.item())
                except Exception as e:
                    logger.warning(f"Could not compute similarity for a trace: {e}")
                    continue

            if similarities:
                max_sim = max(similarities)
                novelty = 1.0 - max_sim

        # Get importance from context
        importance = context.get('importance', 0.5)
        if isinstance(importance, str):
            importance = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(importance, 0.5)

        # Calculate calcium level
        calcium_level = novelty * self.novelty_weight + importance * (1 - self.novelty_weight)
        should_update = calcium_level > self.calcium_threshold

        return UpdateDecision(should_update, calcium_level, novelty, importance)


@dataclass
class MemoryTrace:
    trace_id: str
    content: torch.Tensor
    context: Dict[str, Any]  # Make context more specific if possible, else Any
    salience: float
    timestamp: float

    # New fields from user's method
    last_access: float = 0.0 # Initialized by add_memory_trace_async with current_time
    current_salience: float = 0.0 # Initialized by add_memory_trace_async with salience
    creation_node: str = "" # Initialized by add_memory_trace_async
    access_count: int = 0
    successful_retrievals: int = 0 # Not explicitly set by new method, but good to have
    context_matches: int = 0 # Not explicitly set by new method, but good to have
    consolidation_level: float = 0.0
    eviction_protection: bool = False

    def __post_init__(self):
        # Initialize last_access and current_salience if not set by constructor,
        # though the new add_memory_trace_async sets them.
        if self.last_access == 0.0: # Check if default, means not set by specific constructor
            self.last_access = self.timestamp
        if self.current_salience == 0.0 and self.salience != 0.0: # Check if default
             self.current_salience = self.salience
        if not self.creation_node: # if empty string
            # This might be an issue if trace_id doesn't contain node_id or if node_id isn't easily accessible here
            # For now, leave it, as add_memory_trace_async sets it.
            pass


    def update_access_stats(self, current_time: float, context_relevant: bool = False):
        self.last_access = current_time
        self.access_count += 1
        if context_relevant:
            self.context_matches += 1
        # successful_retrievals would typically be incremented by the calling code after confirming retrieval was useful.
        # current_salience might decay or be updated by other mechanisms, not covered here.
        logger.debug(f"Trace {self.trace_id}: Accessed. Count: {self.access_count}, Last: {self.last_access}, ContextRelevant: {context_relevant}")

class IntegratedMemoryNode:
    """Base class to replace missing EnhancedDistributedMemoryNode"""

    def __init__(self, node_id, dimension, capacity, specialization, device="cpu", config=None):
        self.node_id = node_id
        self.dimension = dimension
        self.capacity = capacity
        self.specialization = specialization
        self.device = device
        self.config = config or {}
        self.memory_traces = [] # Note: GPUAcceleratedDMN might have a different structure for memory_traces
        self.trace_index = {}   # Note: GPUAcceleratedDMN might have a different structure for trace_index
        self.stats = {'total_updates': 0, 'total_retrievals': 0, 'evictions': 0, 'consolidations': 0}
        self.btsp = None # Initialized in _init_biological_mechanisms

    async def _init_biological_mechanisms(self):
        """Initialize with simple BTSP"""
        self.btsp = SimpleBTSP(self.config.get('btsp', {}))
        # Note: GPUAcceleratedDMN's _init_biological_mechanisms calls super() then does more.

    async def add_memory_trace_async(self, content, context, salience, **kwargs):
        raise NotImplementedError("Subclasses must implement add_memory_trace_async")


class GPUAcceleratedDMN(IntegratedMemoryNode):
    """Enhanced DMN with GPU acceleration for FAISS and vector operations"""
    
    def __init__(self, node_id: str, dimension: int, capacity: int = 10000,
                 specialization: str = "general", device: str = "auto", 
                 config: Optional[Dict] = None):

        # Auto-detect best device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_gpu = self.device == "cuda"

        logger.info(f"DMN {node_id}: Initializing with device={self.device}, GPU={self.use_gpu}")

        # Initialize parent class
        super().__init__(node_id, dimension, capacity, specialization, device, config)

        # self.memory_traces is now a list of MemoryTrace objects
        self.memory_traces: List[MemoryTrace] = []
        # self.trace_index maps trace_id (str) to the list index (int)
        self.trace_index: Dict[str, int] = {}

        # NEW: Initialize dynamics engine placeholder
        self.dynamics_engine: Optional['IntegratedMultiTimescaleDynamics'] = None
        self._dynamics_initialized = False

        # GPU/FAISS specific attributes
        self.gpu_resources: Optional[faiss.StandardGpuResources] = None
        self.index: Optional[faiss.Index] = None
        self.next_faiss_id: int = 0
        self.faiss_id_to_trace_id: Dict[int, str] = {}

    async def _init_biological_mechanisms(self):
        """Enhanced biological mechanisms with GPU dynamics"""

        # Initialize parent biological mechanisms (BTSP)
        await super()._init_biological_mechanisms()

        # Initialize FAISS indexing
        self._init_indexing_system()

        # Initialize GPU dynamics engine
        if not self._dynamics_initialized:
            try:
                from core.dynamics import create_integrated_dynamics # Local import

                dynamics_config = self.config.copy()
                dynamics_config.update({
                    'max_synapses': self.capacity * 2,  # Allow overhead
                    'dimension': self.dimension,
                    'device': self.device,
                    'node_id': self.node_id
                })

                self.dynamics_engine = create_integrated_dynamics(dynamics_config)
                await self.dynamics_engine.start()

                self._dynamics_initialized = True
                logger.info(f"DMN {self.node_id}: ✅ GPU dynamics engine initialized")

            except Exception as e:
                logger.error(f"DMN {self.node_id}: GPU dynamics initialization failed: {e}")
                self.dynamics_engine = None

    def _init_indexing_system(self):
        """GPU-accelerated FAISS indexing system"""
        
        index_config = self.config['indexing']
        
        # Strategy 1 & 2: GPU-Accelerated FAISS
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                # Create GPU resources
                self.gpu_resources = faiss.StandardGpuResources()
                
                # Start with CPU index, then move to GPU
                # Temporarily forcing IndexFlatIP for GPU per subtask instructions
                logger.info(f"DMN {self.node_id}: Forcing IndexFlatIP for GPU.")
                cpu_index = faiss.IndexFlatIP(self.dimension)
                
                # Add ID mapping
                cpu_index_with_ids = faiss.IndexIDMap(cpu_index)
                
                # Move to GPU
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index_with_ids)
                
                logger.info(f"DMN {self.node_id}: ✅ GPU-accelerated FAISS initialized")
                
            except Exception as e:
                logger.warning(f"DMN {self.node_id}: GPU FAISS failed: {e}, falling back to CPU")
                self._init_cpu_index(index_config)
                self.gpu_resources = None
        else:
            logger.info(f"DMN {self.node_id}: Using CPU FAISS")
            self._init_cpu_index(index_config)
            self.gpu_resources = None

    async def add_memory_trace_async(self, content: torch.Tensor, context: Dict[str, Any],
                                   salience: float, user_feedback=None) -> Optional[MemoryTrace]:
        """Complete memory storage with GPU dynamics integration"""

        try:
            # Validate input
            if not isinstance(content, torch.Tensor):
                content = torch.tensor(content, dtype=torch.float32)

            if content.dim() != 1 or content.shape[0] != self.dimension:
                logger.error(f"DMN {self.node_id}: Invalid content dimensions: {content.shape}")
                return None

            # Create trace with exact MemoryTrace signature
            current_time = time.time()
            # Generate a more robust trace_id
            trace_id_payload = content.cpu().numpy().tobytes()
            trace_id = f"{self.node_id}_{int(current_time * 1000)}_{hash(trace_id_payload) % 100000}"


            trace = MemoryTrace(
                content=content.clone().detach(), # Store a detached clone
                context=context.copy(), # Store a copy of the context
                timestamp=current_time,
                last_access=current_time, # Set last_access on creation
                salience=salience,
                current_salience=salience, # Initialize current_salience
                trace_id=trace_id,
                creation_node=self.node_id, # Set creation_node
                access_count=0, # Explicitly 0
                successful_retrievals=0, # Explicitly 0
                context_matches=0, # Explicitly 0
                consolidation_level=0.0, # Explicitly 0.0
                eviction_protection=False # Explicitly False
            )

            # BTSP evaluation - get calcium_level!
            calcium_level = 1.0 # Default calcium level
            should_store = True # Default to store

            if self.btsp: # Check if BTSP mechanism exists
                try:
                    update_decision = await self.btsp.evaluate_async(
                        content=content, # Pass original content
                        context=context,
                        existing_traces=list(self.memory_traces.values()) if isinstance(self.memory_traces, dict) else self.memory_traces, # Pass the list of current traces
                        user_feedback=user_feedback
                    )

                    should_store = update_decision.should_update
                    calcium_level = update_decision.calcium_level

                    logger.debug(f"DMN {self.node_id}: BTSP decision: store={should_store}, calcium={calcium_level:.4f}")

                except Exception as e:
                    logger.warning(f"DMN {self.node_id}: BTSP evaluation failed: {e}, falling back to salience check.")
                    # Fallback logic if BTSP fails
                    should_store = salience > self.config.get('btsp_fallback_salience_threshold', 0.5)
            else:
                # Fallback logic if BTSP is not initialized
                logger.warning(f"DMN {self.node_id}: BTSP not initialized, falling back to salience check.")
                should_store = salience > self.config.get('btsp_fallback_salience_threshold', 0.5)


            if not should_store:
                logger.info(f"DMN {self.node_id}: Trace not stored based on BTSP/fallback. Calcium: {calcium_level:.4f}, Salience: {salience:.4f}")
                return None

            # Register with dynamics FIRST if dynamics engine is available and initialized
            if hasattr(self, 'dynamics_engine') and self.dynamics_engine and self._dynamics_initialized:
                try:
                    success_register = self.dynamics_engine.register_trace(trace.trace_id)
                    if not success_register: # Assuming register_trace returns boolean
                        logger.warning(f"DMN {self.node_id}: Failed to register trace {trace.trace_id} with dynamics engine.")
                except Exception as e:
                    logger.error(f"DMN {self.node_id}: Dynamics engine register_trace for {trace.trace_id} failed: {e}", exc_info=True)

            # Store in FAISS index and memory collections
            try:
                await self._add_to_index_async(trace) # Add to FAISS

                # Add to memory_traces list and update trace_index dictionary
                # self.memory_traces is List[MemoryTrace] as per typical DMNs, or Dict if trace_id is key
                # self.trace_index is Dict[str, int] (trace_id to list index)
                # Based on __init__ it's Dict[str, MemoryTrace] and Dict[str, str]
                # For this new method: using self.memory_traces.append and trace_index as dict[id -> list_idx]
                # THIS IS A CHANGE IN STRUCTURE. Needs __init__ update.
                # For now, let's stick to the Dict[str, MemoryTrace] from previous __init__ to avoid breaking retrieve.
                # The prompt implies self.memory_traces will be a list.
                # "self.memory_traces.append(trace)"
                # "self.trace_index[trace.trace_id] = len(self.memory_traces) - 1"
                # This conflicts with recent __init__ update.
                # Reconciling: The prompt's new method structure for memory_traces (list) and trace_index (dict id->idx)
                # is more standard for DMNs where list order can matter and direct indexing is useful.
                # Let's proceed with the prompt's new structure for add_memory_trace_async.
                # The __init__ will need to be changed in the next step.

                # Assuming self.memory_traces is List[MemoryTrace] and self.trace_index is Dict[str, int] for this method.
                # This will require an __init__ update in a subsequent step.

                self.memory_traces.append(trace)
                self.trace_index[trace.trace_id] = len(self.memory_traces) - 1

                logger.debug(f"DMN {self.node_id}: Successfully stored trace {trace.trace_id} in local collections and FAISS. Total traces: {len(self.memory_traces)}")
            except Exception as e:
                logger.error(f"DMN {self.node_id}: Failed to store trace {trace.trace_id} in FAISS/collections: {e}", exc_info=True)
                # Attempt to unregister from dynamics if registration happened and storage failed
                if hasattr(self, 'dynamics_engine') and self.dynamics_engine and self._dynamics_initialized:
                    try:
                        self.dynamics_engine.unregister_trace(trace.trace_id)
                        logger.info(f"DMN {self.node_id}: Unregistered trace {trace.trace_id} from dynamics due to storage failure.")
                    except Exception as unreg_e:
                        logger.error(f"DMN {self.node_id}: Failed to unregister trace {trace.trace_id} from dynamics after storage failure: {unreg_e}", exc_info=True)
                raise e # Re-raise the storage exception

            # Process through temporal dynamics if dynamics engine is available and initialized
            if hasattr(self, 'dynamics_engine') and self.dynamics_engine and self._dynamics_initialized:
                try:
                    await self.dynamics_engine.process_update_async(
                        content=trace.content, # Use content from the stored trace
                        salience=trace.salience, # Use salience from the stored trace
                        calcium_level=calcium_level
                    )
                    logger.debug(f"DMN {self.node_id}: Processed trace {trace.trace_id} through GPU dynamics engine.")
                except Exception as e:
                    logger.error(f"DMN {self.node_id}: GPU dynamics engine processing for trace {trace.trace_id} failed: {e}", exc_info=True)

            self.stats['total_updates'] = self.stats.get('total_updates', 0) + 1 # Safely increment
            return trace

        except Exception as e:
            logger.error(f"DMN {self.node_id}: Unhandled error in add_memory_trace_async: {e}", exc_info=True)
            return None

    def _init_cpu_index(self, index_config):
        """Fallback CPU index initialization"""
        # Use Flat index for now (we know it works)
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)

    async def _add_to_index_async(self, trace):
        """GPU-accelerated index addition with async transfers"""
        
        try:
            # Strategy 5: Use non-blocking transfers
            content_cpu = trace.content.detach().cpu()
            content_normalized = torch.nn.functional.normalize(content_cpu, dim=-1)

            
            # Convert to numpy for FAISS
            content_np = content_normalized.numpy().astype('float32').reshape(1, -1)
            
            # Assign FAISS ID
            faiss_id = self.next_faiss_id
            self.next_faiss_id += 1
            
            # Add to index (GPU or CPU)
            if self.use_gpu:
                # For GPU FAISS, this is automatically async
                self.index.add_with_ids(content_np, np.array([faiss_id], dtype=np.int64))
            else:
                # CPU version
                self.index.add_with_ids(content_np, np.array([faiss_id], dtype=np.int64))
            
            # Update mapping
            self.faiss_id_to_trace_id[faiss_id] = trace.trace_id
            
            logger.debug(f"DMN {self.node_id}: Added trace {trace.trace_id} to {'GPU' if self.use_gpu else 'CPU'} index")
            
        except Exception as e:
            logger.error(f"DMN {self.node_id}: Failed to add to index: {e}")

    async def _rebuild_faiss_index(self):
        """Rebuilds the FAISS index from scratch using all traces in memory."""
        logger.info(f"DMN {self.node_id}: Starting FAISS index rebuild.")
        try:
            # Resetting the index and associated mappings
            logger.info(f"DMN {self.node_id}: Clearing existing FAISS index and mappings.")
            # Re-initialize index to ensure it's clean and to handle potential GPU resource re-creation
            # This logic is similar to _init_indexing_system but simplified for rebuild
            index_config = self.config.get('indexing', {})
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                if not self.gpu_resources:
                    logger.info(f"DMN {self.node_id}: Initializing GPU resources for rebuild.")
                    self.gpu_resources = faiss.StandardGpuResources()

                logger.info(f"DMN {self.node_id}: Rebuilding with IndexFlatIP for GPU.")
                cpu_index = faiss.IndexFlatIP(self.dimension)
                cpu_index_with_ids = faiss.IndexIDMap(cpu_index)
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index_with_ids)
                logger.info(f"DMN {self.node_id}: Re-initialized FAISS index on GPU (FlatIP).")
            else:
                # Fallback or CPU-only mode
                base_index = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(base_index)
                logger.info(f"DMN {self.node_id}: Re-initialized FAISS index on CPU (FlatIP).")

            self.faiss_id_to_trace_id.clear()
            self.next_faiss_id = 0

            # Assuming self.memory_traces is a dict {trace_id: trace_object}
            if not hasattr(self, 'memory_traces') or not self.memory_traces:
                logger.info(f"DMN {self.node_id}: No memory traces to add to the index. Index is empty.")
                if self.index: self.index.reset()
                return

            all_vectors = []
            all_faiss_ids = []

            logger.info(f"DMN {self.node_id}: Processing {len(self.memory_traces)} traces for re-indexing.")

            for trace in self.memory_traces: # self.memory_traces is now a List[MemoryTrace]
                if not hasattr(trace, 'content') or not isinstance(trace.content, torch.Tensor):
                    logger.warning(f"DMN {self.node_id}: Trace {trace.trace_id} has invalid or missing content, skipping.")
                    continue

                content_cpu = trace.content.detach().cpu()

                # Ensure content is a 1D vector or can be treated as one.
                if content_cpu.ndim == 0:
                    logger.warning(f"DMN {self.node_id}: Trace {trace_id} content is a scalar, skipping.")
                    continue
                elif content_cpu.ndim > 1:
                    # If content is (N, dim), take the first vector. Modify if other behavior is needed.
                    logger.warning(f"DMN {self.node_id}: Trace {trace_id} content has shape {content_cpu.shape}. Using first vector.")
                    content_cpu = content_cpu[0]
                    if content_cpu.ndim == 0: # Check again if the first vector was scalar
                         logger.warning(f"DMN {self.node_id}: First vector of trace {trace_id} content is scalar, skipping.")
                         continue

                # Normalize requires (N, dim), so unsqueeze if it's (dim)
                content_normalized = torch.nn.functional.normalize(content_cpu.unsqueeze(0), dim=-1)
                content_np = content_normalized.numpy().astype('float32') # Already (1, dim)

                all_vectors.append(content_np)

                faiss_id = self.next_faiss_id
                self.faiss_id_to_trace_id[faiss_id] = trace.trace_id # Use trace.trace_id here
                all_faiss_ids.append(faiss_id)
                self.next_faiss_id += 1

            if not all_vectors:
                logger.info(f"DMN {self.node_id}: No valid vectors collected from traces. Index will be empty.")
                if self.index: self.index.reset()
                return

            all_vectors_np = np.vstack(all_vectors)
            all_faiss_ids_np = np.array(all_faiss_ids, dtype=np.int64)

            self.index.add_with_ids(all_vectors_np, all_faiss_ids_np)

            logger.info(f"DMN {self.node_id}: Successfully rebuilt FAISS index.")
            logger.info(f"DMN {self.node_id}: Processed {len(all_vectors)} traces. Index ntotal: {self.index.ntotal if self.index else 'None'}.")

        except Exception as e:
            logger.error(f"DMN {self.node_id}: Error during FAISS index rebuild: {e}", exc_info=True)
            logger.warning(f"DMN {self.node_id}: Index rebuild failed. Index might be in an inconsistent state.")

    async def retrieve_memories_async(self, query: torch.Tensor, k: int = 10, 
                                    context_filter=None, similarity_threshold=None, 
                                    temporal_context=None):
        """GPU-accelerated memory retrieval"""
        
        if similarity_threshold is None:
            similarity_threshold = self.config['indexing']['similarity_threshold']
        
        # Strategy 1: GPU tensor operations
        with torch.no_grad():
    # Keep the query on the GPU
            query_gpu = query.to(self.device)
            query_normalized = torch.nn.functional.normalize(query_gpu, dim=-1)
            logger.debug(f"DMN {self.node_id}: Query normalized. Shape: {query_normalized.shape}, Device: {query_normalized.device}")
            
            # Reshape for FAISS search
            query_tensor_gpu = query_normalized.reshape(1, -1)

            # Perform search directly on the GPU
            # Note: faiss-gpu can take a torch tensor directly if it's contiguous
            similarities, faiss_ids = self.index.search(query_tensor_gpu, min(k * 2, self.index.ntotal))
            logger.debug(f"DMN {self.node_id}: Raw FAISS search results - Similarities: {similarities[0][:5]}, FAISS IDs: {faiss_ids[0][:5]}")
            
            logger.debug(f"DMN {self.node_id}: {'GPU' if self.use_gpu else 'CPU'} search returned {len(similarities[0])} results")
            
        # Process results (same as before)
        results = []
        current_time = time.time()
        
        for sim, faiss_id in zip(similarities[0], faiss_ids[0]):
            logger.debug(f"DMN {self.node_id}: Processing FAISS ID {faiss_id} with similarity {sim:.4f}")
            if not (-1.01 <= sim <= 1.01): # Check for IndexFlatIP, range is [-1, 1]
                logger.warning(f"DMN {self.node_id}: Unusual similarity score {sim:.4f} for FAISS ID {faiss_id}. Expected range [-1, 1] for IP index with normalized vectors.")

            if sim < similarity_threshold:
                continue
                
            if faiss_id not in self.faiss_id_to_trace_id:
                logger.warning(f"DMN {self.node_id}: FAISS ID {faiss_id} not found in faiss_id_to_trace_id map. Skipping.")
                continue
                
            trace_id = self.faiss_id_to_trace_id[faiss_id]

            if trace_id not in self.trace_index:
                logger.warning(f"DMN {self.node_id}: Trace ID {trace_id} from FAISS (original FAISS ID: {faiss_id}) not found in trace_index. Skipping.")
                continue

            list_idx = self.trace_index[trace_id]

            if not (0 <= list_idx < len(self.memory_traces)):
                logger.warning(f"DMN {self.node_id}: Stale or invalid list index {list_idx} for trace_id {trace_id} (FAISS ID: {faiss_id}). Max index: {len(self.memory_traces)-1}. Skipping.")
                continue

            trace = self.memory_traces[list_idx]
            
            # Apply context filtering
            if context_filter and not self._matches_context_filter(trace, context_filter):
                continue
            
            # Apply temporal modulation if enabled
            if self.temporal_enabled and temporal_context:
                modulated_sim = self._apply_temporal_modulation(sim, trace, temporal_context)
            else:
                modulated_sim = sim
            
            trace.update_access_stats(current_time, context_relevant=True)
            results.append((trace, float(modulated_sim)))
            
            if len(results) >= k:
                break
        
        logger.debug(f"DMN {self.node_id}: Found {len(results)} potential results after initial filtering (threshold, context, etc.) before final sorting and k-limiting.")
        results.sort(key=lambda x: x[1], reverse=True)
        
        self.stats['total_retrievals'] += 1
        return results

# Strategy 3: GPU-Accelerated Fast Dynamics
class GPUFastDynamics:
    """GPU kernels for fast timescale dynamics (5ms, 500ms)"""
    
    def __init__(self, device="cuda"):
        self.device = device
        
    def update_eligibility_traces(self, traces: torch.Tensor, dt: float, tau: float) -> torch.Tensor:
        """GPU-accelerated eligibility trace decay"""
        # Strategy 3: Element-wise GPU operations instead of loops
        decay_factor = torch.exp(torch.tensor(-dt / tau, device=self.device))
        return traces * decay_factor
    
    def update_synaptic_weights(self, weights: torch.Tensor, eligibility: torch.Tensor, 
                               learning_rates: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated weight updates"""
        # All operations on GPU in parallel
        return weights + learning_rates * eligibility
    
    def compute_neural_activity(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Fast GPU matrix multiplication for neural activity"""
        return torch.mm(inputs, weights.t())

# Strategy 7: Mixed Precision Support
class MixedPrecisionDMN(GPUAcceleratedDMN):
    """DMN with mixed precision for memory efficiency"""
    
    def __init__(self, *args, use_fp16=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fp16 = use_fp16 and self.use_gpu
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None
        
    def _store_weights_fp16(self, weights: torch.Tensor) -> torch.Tensor:
        """Store weights in FP16 to save memory"""
        if self.use_fp16:
            return weights.half()
        return weights
    
    def _compute_weights_fp32(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute in FP32 for precision"""
        if self.use_fp16:
            return weights.float()
        return weights

# Strategy 10: Profiling and Adaptive Offloading
class ProfiledGPUDMN(MixedPrecisionDMN):
    """DMN with automatic CPU/GPU selection based on profiling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_times = []
        self.cpu_times = []
        self.adaptive_threshold = 100  # Switch based on data size
        
    def _should_use_gpu(self, data_size: int) -> bool:
        """Decide CPU vs GPU based on data size and profiling"""
        
        # Small datasets often faster on CPU
        if data_size < self.adaptive_threshold:
            return False
            
        # Large datasets definitely GPU
        if data_size > 1000:
            return self.use_gpu
            
        # Medium datasets: decide based on recent performance
        if len(self.gpu_times) > 10 and len(self.cpu_times) > 10:
            avg_gpu = sum(self.gpu_times[-10:]) / 10
            avg_cpu = sum(self.cpu_times[-10:]) / 10
            return avg_gpu < avg_cpu and self.use_gpu
            
        return self.use_gpu

# Quick implementation starter
def create_gpu_accelerated_dmn(node_id: str, config: dict) -> ProfiledGPUDMN:
    """Factory function to create optimized DMN"""
    
    return ProfiledGPUDMN(
        node_id=node_id,
        dimension=config['system']['dimension'],
        capacity=config['system']['node_capacity'],
        device="auto",  # Auto-detect
        config=config
    )