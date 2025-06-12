# core/dynamics.py - Complete GPU-Accelerated Temporal Dynamics with Proper Integration

import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Add these class definitions to fix the import errors
@dataclass
class TemporalState:
    coherence_metric: float = 1.0
    system_stability: float = 1.0
    fast_synaptic: Dict[str, Any] = field(default_factory=dict)
    calcium_plasticity: Dict[str, Any] = field(default_factory=dict)
    protein_synthesis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalEvent:
    event_type: str
    timescale: str
    params: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class TimescaleSchedule:
    """Manages when each timescale should update"""
    last_update: float = 0.0
    interval: float = 0.005  # seconds
    next_update: float = 0.0
    
    def should_update(self, current_time: float) -> bool:
        return current_time >= self.next_update
    
    def mark_updated(self, current_time: float):
        self.last_update = current_time
        self.next_update = current_time + self.interval

class SynapseTraceMapping:
    """Maps MemoryTrace objects to GPU synapse indices"""
    
    def __init__(self, max_synapses: int):
        self.max_synapses = max_synapses
        self.trace_to_synapse: Dict[str, int] = {}  # trace_id -> synapse_index
        self.synapse_to_trace: Dict[int, str] = {}  # synapse_index -> trace_id
        self.free_indices: List[int] = list(range(max_synapses))
        self.next_index = 0
    
    def assign_synapse(self, trace_id: str) -> Optional[int]:
        """Assign a synapse index to a memory trace"""
        if trace_id in self.trace_to_synapse:
            return self.trace_to_synapse[trace_id]
        
        if not self.free_indices:
            # No free synapses - could implement LRU eviction here
            logger.warning("No free synapses available for new trace")
            return None
        
        synapse_idx = self.free_indices.pop(0)
        self.trace_to_synapse[trace_id] = synapse_idx
        self.synapse_to_trace[synapse_idx] = trace_id
        
        logger.debug(f"Assigned synapse {synapse_idx} to trace {trace_id}")
        return synapse_idx
    
    def release_synapse(self, trace_id: str) -> bool:
        """Release a synapse when trace is evicted"""
        if trace_id not in self.trace_to_synapse:
            return False
        
        synapse_idx = self.trace_to_synapse[trace_id]
        del self.trace_to_synapse[trace_id]
        del self.synapse_to_trace[synapse_idx]
        self.free_indices.append(synapse_idx)
        
        logger.debug(f"Released synapse {synapse_idx} from trace {trace_id}")
        return True
    
    def get_synapse_indices(self, trace_ids: List[str]) -> List[int]:
        """Get synapse indices for multiple traces"""
        indices = []
        for trace_id in trace_ids:
            if trace_id in self.trace_to_synapse:
                indices.append(self.trace_to_synapse[trace_id])
        return indices

class GPUTemporalDynamics:
    """GPU-accelerated temporal dynamics with complete CPU fallbacks"""
    
    def __init__(self, device="cuda", use_mixed_precision=True, max_synapses=100000):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_gpu = self.device.type == "cuda"
        self.use_fp16 = use_mixed_precision and self.use_gpu
        self.max_synapses = max_synapses
        
        # Initialize state tensors
        self._init_state_tensors()
        
        # CUDA streams for async operations
        if self.use_gpu:
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        
        logger.info(f"GPU Dynamics initialized: device={self.device}, "
                   f"fp16={self.use_fp16}, synapses={max_synapses}")

    def _init_state_tensors(self):
        """Initialize all state tensors on appropriate device"""
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Core state tensors
        self.eligibility_traces = torch.zeros(self.max_synapses, dtype=dtype, device=self.device)
        self.calcium_levels = torch.zeros(self.max_synapses, dtype=dtype, device=self.device)
        self.protein_concentrations = torch.zeros(self.max_synapses, dtype=dtype, device=self.device)
        
        # Synaptic weights (could be large - use memory mapping if needed)
        self.synaptic_weights = torch.randn(
            self.max_synapses, 512, dtype=dtype, device=self.device
        ) * 0.01
        
        # Activity history for homeostatic scaling
        self.activity_history = torch.zeros(
            self.max_synapses, 100, dtype=dtype, device=self.device  # Last 100 time steps
        )
        
        self.activity_ptr = 0  # Circular buffer pointer

    def fast_synaptic_update(self, active_synapses: torch.Tensor, dt: float = 0.005) -> Dict[str, Any]:
        """5ms timescale: GPU-accelerated synaptic transmission with CPU fallback"""
        
        start_time = time.time()
        
        try:
            if self.use_gpu:
                return self._gpu_fast_synaptic_update(active_synapses, dt)
            else:
                return self._cpu_fast_synaptic_update(active_synapses, dt)
        except Exception as e:
            logger.error(f"Fast synaptic update failed: {e}")
            # Fallback to CPU if GPU fails
            if self.use_gpu:
                logger.warning("Falling back to CPU for fast synaptic update")
                return self._cpu_fast_synaptic_update(active_synapses, dt)
            raise
        finally:
            self.operation_times['fast_synaptic'].append(time.time() - start_time)

    def _gpu_fast_synaptic_update(self, active_synapses: torch.Tensor, dt: float) -> Dict[str, Any]:
        """GPU implementation of fast synaptic update"""
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            with torch.cuda.stream(self.compute_stream):
                # Eligibility decay (parallel across all synapses)
                tau_eligibility = 0.02  # 20ms
                decay_factor = torch.exp(torch.tensor(-dt / tau_eligibility, device=self.device))
                self.eligibility_traces *= decay_factor
                
                # Spike-driven eligibility updates
                if active_synapses.numel() > 0:
                    # Ensure indices are valid
                    valid_indices = active_synapses[active_synapses < self.max_synapses]
                    if valid_indices.numel() > 0:
                        self.eligibility_traces[valid_indices] += 1.0
                
                # Activity-dependent calcium influx
                calcium_influx = self.eligibility_traces * 0.1
                self.calcium_levels += calcium_influx
                
                # Calcium decay
                tau_calcium = 0.2  # 200ms
                ca_decay = torch.exp(torch.tensor(-dt / tau_calcium, device=self.device))
                self.calcium_levels *= ca_decay
                
                # Update activity history
                self.activity_history[:, self.activity_ptr] = self.eligibility_traces
                self.activity_ptr = (self.activity_ptr + 1) % self.activity_history.shape[1]
                
                # Compute return values
                active_count = torch.sum(self.eligibility_traces > 0.1).item()
                avg_calcium = torch.mean(self.calcium_levels).item()
        
        return {
            "active_synapses": active_count,
            "avg_calcium": avg_calcium,
            "processing_device": "gpu"
        }

    def _cpu_fast_synaptic_update(self, active_synapses: torch.Tensor, dt: float) -> Dict[str, Any]:
        """CPU implementation of fast synaptic update"""
        # Convert to CPU tensors
        eligibility_cpu = self.eligibility_traces.cpu() if self.use_gpu else self.eligibility_traces
        calcium_cpu = self.calcium_levels.cpu() if self.use_gpu else self.calcium_levels
        active_cpu = active_synapses.cpu()
        
        # Eligibility decay
        tau_eligibility = 0.02
        decay_factor = np.exp(-dt / tau_eligibility)
        eligibility_cpu *= decay_factor
        
        # Spike updates
        if active_cpu.numel() > 0:
            valid_indices = active_cpu[active_cpu < self.max_synapses]
            if valid_indices.numel() > 0:
                eligibility_cpu[valid_indices] += 1.0
        
        # Calcium updates
        calcium_influx = eligibility_cpu * 0.1
        calcium_cpu += calcium_influx
        
        tau_calcium = 0.2
        ca_decay = np.exp(-dt / tau_calcium)
        calcium_cpu *= ca_decay
        
        # Update main tensors
        if self.use_gpu:
            self.eligibility_traces.copy_(eligibility_cpu.to(self.device))
            self.calcium_levels.copy_(calcium_cpu.to(self.device))
        else:
            self.eligibility_traces = eligibility_cpu
            self.calcium_levels = calcium_cpu
        
        return {
            "active_synapses": int(torch.sum(eligibility_cpu > 0.1).item()),
            "avg_calcium": float(torch.mean(calcium_cpu).item()),
            "processing_device": "cpu"
        }

    def calcium_plasticity_update(self, dt: float = 0.5) -> Dict[str, Any]:
        """500ms timescale: GPU-accelerated plasticity with CPU fallback"""
        
        start_time = time.time()
        
        try:
            if self.use_gpu:
                return self._gpu_calcium_plasticity_update(dt)
            else:
                return self._cpu_calcium_plasticity_update(dt)
        except Exception as e:
            logger.error(f"Calcium plasticity update failed: {e}")
            if self.use_gpu:
                logger.warning("Falling back to CPU for calcium plasticity update")
                return self._cpu_calcium_plasticity_update(dt)
            raise
        finally:
            self.operation_times['calcium_plasticity'].append(time.time() - start_time)

    def _gpu_calcium_plasticity_update(self, dt: float) -> Dict[str, Any]:
        """GPU implementation of calcium plasticity"""
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            with torch.cuda.stream(self.compute_stream):
                # LTP/LTD thresholds
                ltp_threshold = 0.8
                ltd_threshold = 0.4
                
                # Parallel threshold comparisons
                ltp_mask = self.calcium_levels > ltp_threshold
                ltd_mask = (self.calcium_levels > ltd_threshold) & (self.calcium_levels <= ltp_threshold)
                
                # Weight updates (parallel across all synapses)
                ltp_strength = (self.calcium_levels - ltp_threshold) * ltp_mask.float()
                ltd_strength = (ltd_threshold - self.calcium_levels) * ltd_mask.float()
                
                # Apply weight changes
                weight_changes = (ltp_strength - ltd_strength).unsqueeze(1) * 0.01
                self.synaptic_weights += weight_changes
                
                # Normalize weights to prevent runaway
                weight_norms = torch.norm(self.synaptic_weights, dim=1, keepdim=True)
                self.synaptic_weights = torch.nn.functional.normalize(
                    self.synaptic_weights, dim=1
                ) * torch.clamp(weight_norms, max=2.0)
                
                ltp_events = torch.sum(ltp_mask).item()
                ltd_events = torch.sum(ltd_mask).item()
        
        return {
            "ltp_events": ltp_events,
            "ltd_events": ltd_events,
            "total_plasticity": ltp_events + ltd_events,
            "processing_device": "gpu"
        }

    def _cpu_calcium_plasticity_update(self, dt: float) -> Dict[str, Any]:
        """CPU implementation of calcium plasticity"""
        calcium_cpu = self.calcium_levels.cpu() if self.use_gpu else self.calcium_levels
        weights_cpu = self.synaptic_weights.cpu() if self.use_gpu else self.synaptic_weights
        
        # Thresholds
        ltp_threshold = 0.8
        ltd_threshold = 0.4
        
        # Masks
        ltp_mask = calcium_cpu > ltp_threshold
        ltd_mask = (calcium_cpu > ltd_threshold) & (calcium_cpu <= ltp_threshold)
        
        # Weight changes
        ltp_strength = (calcium_cpu - ltp_threshold) * ltp_mask.float()
        ltd_strength = (ltd_threshold - calcium_cpu) * ltd_mask.float()
        
        weight_changes = (ltp_strength - ltd_strength).unsqueeze(1) * 0.01
        weights_cpu += weight_changes
        
        # Normalize
        weight_norms = torch.norm(weights_cpu, dim=1, keepdim=True)
        weights_cpu = torch.nn.functional.normalize(weights_cpu, dim=1) * torch.clamp(weight_norms, max=2.0)
        
        # Update main tensors
        if self.use_gpu:
            self.synaptic_weights.copy_(weights_cpu.to(self.device))
        else:
            self.synaptic_weights = weights_cpu
        
        return {
            "ltp_events": int(torch.sum(ltp_mask).item()),
            "ltd_events": int(torch.sum(ltd_mask).item()),
            "total_plasticity": int(torch.sum(ltp_mask | ltd_mask).item()),
            "processing_device": "cpu"
        }

    def protein_synthesis_update(self, dt: float = 60.0) -> Dict[str, Any]:
        """60s timescale: GPU-accelerated consolidation with CPU fallback"""
        
        start_time = time.time()
        
        try:
            if self.use_gpu:
                return self._gpu_protein_synthesis_update(dt)
            else:
                return self._cpu_protein_synthesis_update(dt)
        except Exception as e:
            logger.error(f"Protein synthesis update failed: {e}")
            if self.use_gpu:
                logger.warning("Falling back to CPU for protein synthesis update")
                return self._cpu_protein_synthesis_update(dt)
            raise
        finally:
            self.operation_times['protein_synthesis'].append(time.time() - start_time)

    def _gpu_protein_synthesis_update(self, dt: float) -> Dict[str, Any]:
        """GPU implementation of protein synthesis"""
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            with torch.cuda.stream(self.compute_stream):
                # Protein synthesis driven by calcium history
                synthesis_threshold = 0.6
                synthesis_mask = self.calcium_levels > synthesis_threshold
                
                # Protein concentration updates
                synthesis_rate = self.calcium_levels * synthesis_mask.float() * 0.01
                self.protein_concentrations += synthesis_rate * dt
                
                # Protein decay
                tau_protein = 3600.0  # 1 hour
                protein_decay = torch.exp(torch.tensor(-dt / tau_protein, device=self.device))
                self.protein_concentrations *= protein_decay
                
                consolidating_synapses = torch.sum(synthesis_mask).item()
                avg_protein = torch.mean(self.protein_concentrations).item()
        
        return {
            "consolidating_synapses": consolidating_synapses,
            "avg_protein_concentration": avg_protein,
            "processing_device": "gpu"
        }

    def _cpu_protein_synthesis_update(self, dt: float) -> Dict[str, Any]:
        """CPU implementation of protein synthesis"""
        calcium_cpu = self.calcium_levels.cpu() if self.use_gpu else self.calcium_levels
        protein_cpu = self.protein_concentrations.cpu() if self.use_gpu else self.protein_concentrations
        
        # Synthesis
        synthesis_threshold = 0.6
        synthesis_mask = calcium_cpu > synthesis_threshold
        
        synthesis_rate = calcium_cpu * synthesis_mask.float() * 0.01
        protein_cpu += synthesis_rate * dt
        
        # Decay
        tau_protein = 3600.0
        protein_decay = np.exp(-dt / tau_protein)
        protein_cpu *= protein_decay
        
        # Update main tensor
        if self.use_gpu:
            self.protein_concentrations.copy_(protein_cpu.to(self.device))
        else:
            self.protein_concentrations = protein_cpu
        
        return {
            "consolidating_synapses": int(torch.sum(synthesis_mask).item()),
            "avg_protein_concentration": float(torch.mean(protein_cpu).item()),
            "processing_device": "cpu"
        }

    def homeostatic_scaling_update(self, dt: float = 3600.0) -> Dict[str, Any]:
        """1 hour timescale: Homeostatic scaling"""
        
        start_time = time.time()
        
        try:
            # Compute average activity over recent history
            if self.use_gpu:
                with torch.cuda.stream(self.compute_stream):
                    avg_activity = torch.mean(self.activity_history, dim=1)
                    target_activity = 0.1  # Target firing rate
                    
                    # Scaling factor
                    scaling_factor = target_activity / (avg_activity + 1e-6)
                    scaling_factor = torch.clamp(scaling_factor, 0.5, 2.0)  # Limit scaling
                    
                    # Apply scaling to weights
                    self.synaptic_weights *= scaling_factor.unsqueeze(1)
                    
                    scaled_synapses = torch.sum(torch.abs(scaling_factor - 1.0) > 0.1).item()
            else:
                # CPU version
                activity_cpu = self.activity_history.cpu() if self.use_gpu else self.activity_history
                avg_activity = torch.mean(activity_cpu, dim=1)
                target_activity = 0.1
                
                scaling_factor = target_activity / (avg_activity + 1e-6)
                scaling_factor = torch.clamp(scaling_factor, 0.5, 2.0)
                
                weights_cpu = self.synaptic_weights.cpu() if self.use_gpu else self.synaptic_weights
                weights_cpu *= scaling_factor.unsqueeze(1)
                
                if self.use_gpu:
                    self.synaptic_weights.copy_(weights_cpu.to(self.device))
                else:
                    self.synaptic_weights = weights_cpu
                
                scaled_synapses = int(torch.sum(torch.abs(scaling_factor - 1.0) > 0.1).item())
            
            return {
                "scaled_synapses": scaled_synapses,
                "avg_scaling_factor": float(torch.mean(scaling_factor).item()),
                "processing_device": "gpu" if self.use_gpu else "cpu"
            }
            
        except Exception as e:
            logger.error(f"Homeostatic scaling failed: {e}")
            return {"scaled_synapses": 0, "avg_scaling_factor": 1.0, "error": str(e)}
        finally:
            self.operation_times['homeostatic_scaling'].append(time.time() - start_time)

    async def async_update_batch(self, trace_batch: List[torch.Tensor]) -> Dict[str, Any]:
        """Async batch processing without blocking synchronization"""
        
        if not trace_batch:
            return {}
        
        # Fire and forget - don't synchronize here
        if self.use_gpu:
            with torch.cuda.stream(self.compute_stream):
                batch_tensor = torch.stack([trace.to(self.device, non_blocking=True) for trace in trace_batch])
                
                # Process batch
                if self.synaptic_weights.numel() > 0:
                    similarities = torch.mm(batch_tensor, self.synaptic_weights.t())
                    best_matches = torch.argmax(similarities, dim=1)
                    best_similarities = torch.max(similarities, dim=1)[0]
                else:
                    best_matches = torch.zeros(len(trace_batch), dtype=torch.long, device=self.device)
                    best_similarities = torch.zeros(len(trace_batch), device=self.device)
                
                # Store results for later retrieval - don't synchronize yet
                return {
                    "batch_size": len(trace_batch),
                    "best_matches": best_matches,
                    "similarities": best_similarities,
                    "processing_device": "gpu",
                    "stream": self.compute_stream  # Caller can synchronize if needed
                }
        else:
            # CPU processing
            return self._cpu_batch_process(trace_batch)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {
            "device": str(self.device),
            "using_gpu": self.use_gpu,
            "using_fp16": self.use_fp16,
            "max_synapses": self.max_synapses
        }
        
        # Operation timing statistics
        for op_name, times in self.operation_times.items():
            if times:
                stats[f"{op_name}_avg_time"] = np.mean(times[-100:])  # Last 100 operations
                stats[f"{op_name}_operations"] = len(times)
        
        if self.use_gpu:
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            })
        
        return stats

class IntegratedMultiTimescaleDynamics:
    """Complete integration layer with proper timescale orchestration"""
    
    def __init__(self, config=None, device="auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        # Initialize GPU dynamics engine
        max_synapses = config.get('max_synapses', 100000) if config else 100000
        self.gpu_dynamics = GPUTemporalDynamics(self.device, max_synapses=max_synapses)
        
        # Initialize trace-synapse mapping
        self.synapse_mapping = SynapseTraceMapping(max_synapses)
        
        # Initialize timescale schedules
        self.schedules = {
            'fast_synaptic': TimescaleSchedule(interval=0.005),     # 5ms
            'calcium_plasticity': TimescaleSchedule(interval=0.5),  # 500ms
            'protein_synthesis': TimescaleSchedule(interval=60.0),  # 60s
            'homeostatic_scaling': TimescaleSchedule(interval=3600.0), # 1 hour
        }
        
        self.current_time = 0.0
        self.is_running = False
        
        # Statistics
        self.update_counts = defaultdict(int)
        self.last_stats_time = time.time()
        
        logger.info(f"Integrated Multi-Timescale Dynamics initialized with {max_synapses} synapses")

    async def start(self):
        """Start the dynamics engine"""
        self.is_running = True
        self.current_time = time.time()
        
        # Start background update loop
        asyncio.create_task(self._background_update_loop())
        
        logger.info("Multi-timescale dynamics engine started")

    async def stop(self):
        """Stop the dynamics engine"""
        self.is_running = False
        logger.info("Multi-timescale dynamics engine stopped")

    async def _background_update_loop(self):
        """Background loop that runs timescale updates at appropriate intervals"""
        
        while self.is_running:
            try:
                current_time = time.time()
                dt = current_time - self.current_time
                self.current_time = current_time
                
                # Check each timescale for updates
                results = {}
                
                # Fast synaptic (5ms)
                if self.schedules['fast_synaptic'].should_update(current_time):
                    # Get currently active synapses (those with recent activity)
                    active_synapses = self._get_active_synapses()
                    results['fast_synaptic'] = self.gpu_dynamics.fast_synaptic_update(active_synapses)
                    self.schedules['fast_synaptic'].mark_updated(current_time)
                    self.update_counts['fast_synaptic'] += 1
                
                # Calcium plasticity (500ms)
                if self.schedules['calcium_plasticity'].should_update(current_time):
                    results['calcium_plasticity'] = self.gpu_dynamics.calcium_plasticity_update()
                    self.schedules['calcium_plasticity'].mark_updated(current_time)
                    self.update_counts['calcium_plasticity'] += 1
                
                # Protein synthesis (60s)
                if self.schedules['protein_synthesis'].should_update(current_time):
                    results['protein_synthesis'] = self.gpu_dynamics.protein_synthesis_update()
                    self.schedules['protein_synthesis'].mark_updated(current_time)
                    self.update_counts['protein_synthesis'] += 1
                
                # Homeostatic scaling (1 hour)
                if self.schedules['homeostatic_scaling'].should_update(current_time):
                    results['homeostatic_scaling'] = self.gpu_dynamics.homeostatic_scaling_update()
                    self.schedules['homeostatic_scaling'].mark_updated(current_time)
                    self.update_counts['homeostatic_scaling'] += 1
                
                # Sleep until next potential update
                await asyncio.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                logger.error(f"Background update loop error: {e}")
                await asyncio.sleep(0.1)  # Longer sleep on error

    def _get_active_synapses(self) -> torch.Tensor:
        """Get indices of currently active synapses"""
        # This could be based on recent trace accesses, current eligibility, etc.
        # For now, return synapses with high eligibility
        try:
            eligibility = self.gpu_dynamics.eligibility_traces
            active_mask = eligibility > 0.1
            active_indices = torch.nonzero(active_mask).flatten()
            return active_indices
        except Exception:
            return torch.empty(0, dtype=torch.long, device=self.gpu_dynamics.device)

    async def process_update_async(self, content: torch.Tensor, salience: float, calcium_level: float) -> bool:
        """Process memory update through temporal dynamics"""
        try:
            # Move content to appropriate device
            content_device = content.to(self.gpu_dynamics.device)
            
            # This method gets called when memories are stored/updated
            # We need to trigger appropriate timescale responses
            
            # High salience triggers immediate synaptic activity
            if salience > 0.7:
                # Find or assign synapse for this content
                content_hash = hash(content_device.cpu().numpy().tobytes())
                trace_id = f"content_{content_hash}"
                
                synapse_idx = self.synapse_mapping.assign_synapse(trace_id)
                if synapse_idx is not None:
                    # Trigger activity at this synapse
                    active_synapses = torch.tensor([synapse_idx], device=self.gpu_dynamics.device)
                    self.gpu_dynamics.fast_synaptic_update(active_synapses)
            
            # High calcium level modulates plasticity
            if calcium_level > 0.8:
                # Could trigger immediate plasticity update
                self.gpu_dynamics.calcium_plasticity_update()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing update: {e}")
            return False

    async def process_retrieval_async(self, query: torch.Tensor, retrieved_traces: List[Any]) -> bool:
        """Process memory retrieval through temporal dynamics"""
        try:
            if not retrieved_traces:
                return True
            
            # Retrieval activates synapses associated with retrieved traces
            trace_ids = [getattr(trace, 'trace_id', f'trace_{i}') for i, trace in enumerate(retrieved_traces)]
            synapse_indices = self.synapse_mapping.get_synapse_indices(trace_ids)
            
            if synapse_indices:
                active_synapses = torch.tensor(synapse_indices, device=self.gpu_dynamics.device)
                self.gpu_dynamics.fast_synaptic_update(active_synapses)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing retrieval: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        base_stats = self.gpu_dynamics.get_performance_stats()
        
        base_stats.update({
            'is_running': self.is_running,
            'current_time': self.current_time,
            'update_counts': dict(self.update_counts),
            'temporal_coherence': 1.0,  # Could compute from GPU state
            'system_stability': 1.0,    # Could compute from weight changes
            'synapse_utilization': len(self.synapse_mapping.trace_to_synapse) / self.synapse_mapping.max_synapses
        })
        
        return base_stats

    def register_trace(self, trace_id: str) -> bool:
        """Register a new memory trace with the dynamics system"""
        synapse_idx = self.synapse_mapping.assign_synapse(trace_id)
        return synapse_idx is not None

    def unregister_trace(self, trace_id: str) -> bool:
        """Unregister a memory trace (e.g., during eviction)"""
        return self.synapse_mapping.release_synapse(trace_id)

# Factory function for easy integration
def create_integrated_dynamics(config: Dict[str, Any]) -> IntegratedMultiTimescaleDynamics:
    """Factory function to create properly configured dynamics engine"""
    
    return IntegratedMultiTimescaleDynamics(
        config=config,
        device=config.get('device', 'auto')
    )

# Add this alias for MultiTimescaleDynamicsEngine
MultiTimescaleDynamicsEngine = IntegratedMultiTimescaleDynamics