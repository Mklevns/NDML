# core/dmn.py - GPU-Accelerated FAISS Implementation

import torch
import faiss
import numpy as np
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class GPUAcceleratedDMN(EnhancedDistributedMemoryNode):
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

        # NEW: Initialize dynamics engine placeholder
        self.dynamics_engine: Optional['IntegratedMultiTimescaleDynamics'] = None
        self._dynamics_initialized = False

    async def _init_biological_mechanisms(self):
        """Enhanced biological mechanisms with GPU dynamics"""

        # Initialize parent biological mechanisms (BTSP, etc.)
        await super()._init_biological_mechanisms()

        # NEW: Initialize GPU dynamics engine
        if not self._dynamics_initialized:
            try:
                from core.dynamics import create_integrated_dynamics

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
                logger.info(f"DMN {self.node_id}: GPU dynamics engine initialized")

            except Exception as e:
                logger.error(f"DMN {self.node_id}: Failed to initialize GPU dynamics: {e}")
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

        self.next_faiss_id = 0
        self.faiss_id_to_trace_id: Dict[int, str] = {}

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

            for trace_id, trace in self.memory_traces.items():
                if not hasattr(trace, 'content') or not isinstance(trace.content, torch.Tensor):
                    logger.warning(f"DMN {self.node_id}: Trace {trace_id} has invalid or missing content, skipping.")
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
                self.faiss_id_to_trace_id[faiss_id] = trace_id
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
                continue
                
            trace_id = self.faiss_id_to_trace_id[faiss_id]
            if trace_id not in self.trace_index:
                continue
                
            trace = self.memory_traces[self.trace_index[trace_id]]
            
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