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


class ConsolidationState(Enum):
    INITIAL = "initial"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    REACTIVATED = "reactivated"
    WEAKENING = "weakening"

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationProgress:
    """Tracks detailed consolidation progress with biological realism"""
    calcium_level: float = 0.0
    protein_synthesis_level: float = 0.0
    synaptic_strength: float = 0.0
    systems_integration: float = 0.0
    homeostatic_scaling: float = 0.0
    last_calcium_spike: float = 0.0
    consolidation_attempts: int = 0
    interference_resistance: float = 0.0


class CalciumDynamicsSimulator:
    """Simulates calcium dynamics for BTSP-based consolidation"""

    def __init__(self, threshold: float = 0.7, decay_tau: float = 10.0):
        """Initialize the simulator.

        Parameters
        ----------
        threshold : float, optional
            Calcium level required to trigger downstream plasticity.
        decay_tau : float, optional
            Time constant (in seconds) for exponential calcium decay.
        """
        self.threshold = threshold
        self.decay_tau = decay_tau  # Time constant in seconds
        self.calcium_level = 0.0
        self.last_update_time = 0.0

    async def update_calcium(self, signal_strength: float, current_time: float) -> float:
        """Update calcium level with temporal decay and signal integration"""
        try:
            # Apply temporal decay
            if self.last_update_time > 0:
                dt = current_time - self.last_update_time
                decay_factor = np.exp(-dt / self.decay_tau)
                self.calcium_level *= decay_factor

            # Signal-driven calcium influx (simulating NMDA/VGCC activation)
            calcium_influx = signal_strength * 0.3  # Scaling factor
            self.calcium_level += calcium_influx

            # Saturation and bounds
            self.calcium_level = np.clip(self.calcium_level, 0.0, 2.0)
            self.last_update_time = current_time

            return self.calcium_level

        except Exception as e:
            logger.error(f"Error updating calcium dynamics: {e}")
            return self.calcium_level


class BTSPConsolidationEngine:
    """BTSP-based consolidation engine with multi-timescale dynamics"""

    def __init__(self, config: Optional[Dict] = None):
        """Create a new consolidation engine.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If ``None`` a default configuration is
            used.  When ``config['use_gpu']`` is ``True`` and CUDA is available
            GPU accelerated operations will be enabled.
        """
        self.config = config or self._default_config()
        self.use_gpu = torch.cuda.is_available() and self.config.get("use_gpu", False)
        self.calcium_simulator = CalciumDynamicsSimulator(
            threshold=self.config['calcium_threshold'],
            decay_tau=self.config['calcium_decay_tau']
        )
        self.consolidation_stats = defaultdict(int)
        self.active_consolidations = {}

    def _default_config(self) -> Dict:
        return {
            'calcium_threshold': 0.7,
            'calcium_decay_tau': 10.0,
            'consolidation_rate': 0.1,
            'interference_threshold': 0.3,
            'max_concurrent_consolidations': 50,
            'protein_synthesis_window': 600,  # 10 minutes
            'systems_consolidation_window': 3600,  # 1 hour
            'homeostatic_window': 86400,  # 24 hours
        }

    async def compute_consolidation_signal(self, trace: Any, current_time: float) -> float:
        """Compute consolidation signal based on multiple factors"""
        try:
            signals = []
            salience_signal = getattr(trace, 'current_salience', 0.0)
            signals.append(salience_signal)
            access_count = getattr(trace, 'access_count', 0)
            access_signal = np.log1p(access_count) / 10.0
            signals.append(access_signal)
            if hasattr(trace, 'temporal_metadata'):
                temporal_coherence = getattr(trace.temporal_metadata, 'temporal_coherence', 0.0)
                signals.append(temporal_coherence)
            interference = await self._compute_interference(trace, current_time)
            signals.append(-interference * self.config['interference_threshold'])
            weights = [0.4, 0.2, 0.2, 0.2]
            total_signal = sum(w * s for w, s in zip(weights, signals))
            return np.clip(total_signal, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Error computing consolidation signal: {e}")
            return 0.0

    async def _compute_interference(self, trace: Any, current_time: float) -> float:
        """Compute interference from other consolidating memories"""
        try:
            if not hasattr(trace, 'content_embedding'):
                return 0.0
            trace_embedding = trace.content_embedding

            other_embeddings = []
            strengths = []
            for oid, otrace in self.active_consolidations.items():
                if oid == getattr(trace, 'trace_id', None):
                    continue
                if hasattr(otrace, 'content_embedding'):
                    other_embeddings.append(otrace.content_embedding)
                    if hasattr(otrace, 'temporal_metadata'):
                        strengths.append(getattr(otrace.temporal_metadata, 'consolidation_strength', 0.0))
                    else:
                        strengths.append(0.0)

            if not other_embeddings:
                return 0.0

            other_tensor = torch.stack(other_embeddings)
            similarities = torch.cosine_similarity(trace_embedding.unsqueeze(0), other_tensor, dim=1)
            strength_tensor = torch.tensor(strengths, device=similarities.device)
            interference_score = torch.sum(similarities * strength_tensor).item()

            return float(np.clip(interference_score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error computing interference: {e}")
            return 0.0

    async def initiate_consolidation(self, trace: Any, current_time: float) -> bool:
        """
        Enhanced consolidation initiation with BTSP mechanisms and multi-timescale dynamics
        """
        try:
            trace_id = getattr(trace, 'trace_id', f'trace_{id(trace)}')
            logger.debug(f"Initiating enhanced consolidation for trace {trace_id}")
            if trace_id in self.active_consolidations:
                logger.debug(f"Trace {trace_id} already consolidating")
                return False
            if len(self.active_consolidations) >= self.config.get('max_concurrent_consolidations', 50):
                logger.warning("Maximum concurrent consolidations reached")
                return False
            if not hasattr(trace, 'temporal_metadata'):
                from core.memory_trace import TemporalMetadata # Assuming TemporalMetadata is in core.memory_trace
                trace.temporal_metadata = TemporalMetadata()
            age_category = self._get_temporal_age_category(trace, current_time)
            consolidation_signal = await self.compute_consolidation_signal(trace, current_time)
            calcium_level = await self.calcium_simulator.update_calcium(
                consolidation_signal, current_time
            )
            if calcium_level < self.config['calcium_threshold']:
                logger.debug(f"Calcium level {calcium_level:.3f} below threshold {self.config['calcium_threshold']}")
                return False
            consolidation_progress = ConsolidationProgress(
                calcium_level=calcium_level,
                last_calcium_spike=current_time,
                consolidation_attempts=1
            )
            trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATING
            trace.temporal_metadata.last_consolidation = current_time
            trace.temporal_metadata.consolidation_strength = 0.1
            if not hasattr(trace, '_consolidation_progress'): # Ensure attribute exists before assignment
                trace._consolidation_progress = consolidation_progress
            else: # if it exists, update it
                trace._consolidation_progress = consolidation_progress

            self.active_consolidations[trace_id] = trace
            self.consolidation_stats['initiated'] += 1
            self.consolidation_stats[f'initiated_{age_category}'] += 1
            logger.info(f"Consolidation initiated for trace {trace_id} (age: {age_category}, calcium: {calcium_level:.3f})")
            return True
        except Exception as e:
            logger.error(f"Error initiating consolidation for trace {getattr(trace, 'trace_id', 'unknown')}: {e}")
            return False

    async def update_consolidation_progress(self, trace: Any, current_time: float,
                                          progress_increment: float = None) -> bool:
        """
        Enhanced consolidation progress update with multi-timescale biological dynamics
        """
        try:
            trace_id = getattr(trace, 'trace_id', f'trace_{id(trace)}')
            if not hasattr(trace, 'temporal_metadata'):
                logger.warning(f"No temporal metadata for trace {trace_id}")
                return False
            if trace.temporal_metadata.consolidation_state != ConsolidationState.CONSOLIDATING:
                return False
            if not hasattr(trace, '_consolidation_progress'):
                trace._consolidation_progress = ConsolidationProgress()
            progress = trace._consolidation_progress
            age_category = self._get_temporal_age_category(trace, current_time)
            if progress_increment is None:
                progress_increment = await self._compute_timescale_progress(
                    trace, current_time, age_category
                )
            consolidation_signal = await self.compute_consolidation_signal(trace, current_time)
            calcium_level = await self.calcium_simulator.update_calcium(
                consolidation_signal, current_time
            )
            progress.calcium_level = calcium_level
            success = await self._update_multiscale_consolidation(
                trace, progress, current_time, age_category, progress_increment
            )
            if not success:
                return False
            old_strength = trace.temporal_metadata.consolidation_strength
            new_strength = self._compute_consolidated_strength(progress)
            trace.temporal_metadata.consolidation_strength = min(1.0, new_strength)
            completion_threshold = self._get_completion_threshold(age_category)
            if new_strength >= completion_threshold:
                await self._complete_consolidation(trace, current_time)
                self.consolidation_stats['completed'] += 1
                logger.info(f"Consolidation completed for trace {trace_id} (strength: {new_strength:.3f})")
            strength_improvement = new_strength - old_strength
            if strength_improvement > 0:
                self.consolidation_stats['progress_updates'] += 1
                self.consolidation_stats[f'progress_{age_category}'] += 1
            logger.debug(f"Consolidation progress for trace {trace_id}: {old_strength:.3f} -> {new_strength:.3f}")
            return True
        except Exception as e:
            logger.error(f"Error updating consolidation progress for trace {getattr(trace, 'trace_id', 'unknown')}: {e}")
            return False

    def _get_temporal_age_category(self, trace: Any, current_time: float) -> str:
        """Determine temporal age category based on biological timescales"""
        try:
            age = current_time - getattr(trace, 'timestamp', current_time)
            if age < 0.005: return "fast_synaptic"
            elif age < 0.5: return "calcium_plasticity"
            elif age < 60: return "protein_synthesis"
            elif age < 3600: return "homeostatic_scaling"
            else: return "systems_consolidation"
        except Exception as e:
            logger.error(f"Error determining age category: {e}")
            return "protein_synthesis"

    async def _compute_timescale_progress(self, trace: Any, current_time: float,
                                        age_category: str) -> float:
        """Compute progress increment based on biological timescale"""
        try:
            base_rates = {
                "fast_synaptic": 0.5, "calcium_plasticity": 0.3,
                "protein_synthesis": 0.1, "homeostatic_scaling": 0.05,
                "systems_consolidation": 0.02
            }
            base_rate = base_rates.get(age_category, 0.1)
            salience_factor = getattr(trace, 'current_salience', 0.5)
            calcium_factor = min(1.0, self.calcium_simulator.calcium_level / self.config['calcium_threshold'])
            interference = await self._compute_interference(trace, current_time)
            interference_factor = 1.0 - (interference * 0.5)
            progress_val = base_rate * salience_factor * calcium_factor * interference_factor # Renamed progress to progress_val
            return np.clip(progress_val, 0.0, 0.5)
        except Exception as e:
            logger.error(f"Error computing timescale progress: {e}")
            return 0.01

    async def _update_multiscale_consolidation(self, trace: Any, progress: ConsolidationProgress,
                                             current_time: float, age_category: str,
                                             progress_increment: float) -> bool:
        """Update consolidation across multiple biological timescales"""
        try:
            if age_category in ["protein_synthesis", "homeostatic_scaling", "systems_consolidation"]:
                protein_factor = min(1.0, progress.calcium_level * 2.0)
                progress.protein_synthesis_level = min(1.0,
                    progress.protein_synthesis_level + progress_increment * protein_factor
                )
            if progress.calcium_level > self.config['calcium_threshold']:
                strength_factor = (progress.calcium_level - self.config['calcium_threshold']) / \
                                   (2.0 - self.config['calcium_threshold'])
                progress.synaptic_strength = min(1.0,
                    progress.synaptic_strength + progress_increment * strength_factor
                )
            if age_category in ["homeostatic_scaling", "systems_consolidation"]:
                systems_factor = progress.protein_synthesis_level * 0.5
                progress.systems_integration = min(1.0,
                    progress.systems_integration + progress_increment * systems_factor
                )
            if age_category == "systems_consolidation":
                homeostatic_factor = progress.systems_integration * 0.3
                progress.homeostatic_scaling = min(1.0,
                    progress.homeostatic_scaling + progress_increment * homeostatic_factor
                )
            resistance_increment = progress_increment * 0.2
            progress.interference_resistance = min(1.0,
                progress.interference_resistance + resistance_increment
            )
            progress.consolidation_attempts += 1
            return True
        except Exception as e:
            logger.error(f"Error updating multiscale consolidation: {e}")
            return False

    def _compute_consolidated_strength(self, progress: ConsolidationProgress) -> float:
        """Compute overall consolidation strength from multi-timescale components"""
        try:
            weights = {'calcium': 0.2, 'protein': 0.25, 'synaptic': 0.25, 'systems': 0.2, 'homeostatic': 0.1}
            components = {
                'calcium': min(1.0, progress.calcium_level / 2.0),
                'protein': progress.protein_synthesis_level,
                'synaptic': progress.synaptic_strength,
                'systems': progress.systems_integration,
                'homeostatic': progress.homeostatic_scaling
            }
            total_strength = sum(weights[k] * components[k] for k in weights.keys())
            resistance_bonus = progress.interference_resistance * 0.1
            return min(1.0, total_strength + resistance_bonus)
        except Exception as e:
            logger.error(f"Error computing consolidated strength: {e}")
            return 0.0

    def _get_completion_threshold(self, age_category: str) -> float:
        """Get consolidation completion threshold based on age category"""
        thresholds = {
            "fast_synaptic": 0.8, "calcium_plasticity": 0.75,
            "protein_synthesis": 0.7, "homeostatic_scaling": 0.65,
            "systems_consolidation": 0.6
        }
        return thresholds.get(age_category, 0.7)

    async def _complete_consolidation(self, trace: Any, current_time: float):
        """Complete consolidation process and cleanup"""
        try:
            trace_id = getattr(trace, 'trace_id', f'trace_{id(trace)}')
            trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATED
            trace.temporal_metadata.last_consolidation = current_time
            if trace_id in self.active_consolidations:
                del self.active_consolidations[trace_id]
            logger.info(f"Consolidation completed for trace {trace_id}")
        except Exception as e:
            logger.error(f"Error completing consolidation: {e}")

    async def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get detailed consolidation statistics"""
        try:
            stats = dict(self.consolidation_stats)
            stats.update({
                'active_consolidations': len(self.active_consolidations),
                'calcium_level': self.calcium_simulator.calcium_level,
                'calcium_threshold': self.config['calcium_threshold'],
                'consolidation_rate': self.config['consolidation_rate'],
                'timestamp': time.time()
            })
            return stats
        except Exception as e:
            logger.error(f"Error getting consolidation statistics: {e}")
            return {}

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

    def __init__(self, device: str = "cuda", use_mixed_precision: bool = True,
                 max_synapses: int = 100000, config: Optional[Dict[str, Any]] = None):
        """Initialize GPU dynamics state.

        Parameters
        ----------
        device : str, optional
            Preferred device identifier (``"cuda"`` or ``"cpu"``).
        use_mixed_precision : bool, optional
            Enable FP16 operations when running on GPU.
        max_synapses : int, optional
            Maximum number of synapses that will be simulated.
        config : dict, optional
            Optional configuration controlling GPU usage (expects ``use_gpu`` key).
        """
        self.config = config or {}
        self.use_gpu = torch.cuda.is_available() and self.config.get("use_gpu", False)
        self.device = torch.device(device if self.use_gpu else "cpu")
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

class IntegratedMultiTimescaleDynamics(BTSPConsolidationEngine):
    """Complete integration layer with proper timescale orchestration, including BTSP consolidation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "auto"):
        """Initialize the integrated dynamics engine.

        Parameters
        ----------
        config : dict, optional
            Combined configuration for consolidation and GPU dynamics.
        device : str, optional
            Preferred compute device. ``"auto"`` selects GPU when available and
            allowed by the configuration.
        """
        # Initialize BTSP consolidation engine first
        consolidation_config = config.get('consolidation', {}) if config else {}
        super().__init__(consolidation_config)

        # Then initialize dynamics-specific components
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device

        # Use config for max_synapses, defaulting if not provided
        max_synapses = config.get('max_synapses', 100000) if config else 100000
        self.gpu_dynamics = GPUTemporalDynamics(
            self.device, max_synapses=max_synapses,
            config=config
        )
        self.synapse_mapping = SynapseTraceMapping(max_synapses)

        # Initialize timescale schedules (ensure TimescaleSchedule is defined or imported)
        self.schedules = {
            'fast_synaptic': TimescaleSchedule(interval=0.005),
            'calcium_plasticity': TimescaleSchedule(interval=0.5),
            'protein_synthesis': TimescaleSchedule(interval=60.0),
            'homeostatic_scaling': TimescaleSchedule(interval=3600.0),
        }

        self.current_time = 0.0 # This might be already set by BTSPConsolidationEngine or managed differently. Review if necessary.
        self.is_running = False
        self.update_counts = defaultdict(int) # This is specific to IntegratedMultiTimescaleDynamics

        logger.info(f"Enhanced IntegratedMultiTimescaleDynamics with inherited BTSP initialized, max_synapses={max_synapses}")

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
        """Get comprehensive system statistics from both dynamics and consolidation engine."""
        base_stats = self.gpu_dynamics.get_performance_stats()

        # Get consolidation stats. Since get_consolidation_statistics is async,
        # and get_stats is synchronous, we need to run the async method.
        # Ensure asyncio is imported.
        try:
            # If an event loop is already running (e.g., in a Jupyter notebook or other async context)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # This is tricky. If called from an async context that's already running,
                # creating a new task might be better, but that makes get_stats async.
                # For simplicity in a sync method, we might just log a warning
                # or attempt asyncio.run in a thread if truly necessary.
                # The simplest approach for now if a loop is running is to not block it with asyncio.run directly.
                # However, the user's example used asyncio.run. Let's try to stick to that.
                # If issues arise, this part might need refinement based on execution context.
                consolidation_stats = asyncio.run(self.get_consolidation_statistics())
            else:
                consolidation_stats = asyncio.run(self.get_consolidation_statistics())
        except RuntimeError as e:
            # RuntimeError is often raised by asyncio.run if it's called from a running loop
            # or if it's re-entrant.
            logger.warning(f"Could not get consolidation stats via asyncio.run: {e}. Falling back to empty dict.")
            consolidation_stats = {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting consolidation_stats: {e}")
            consolidation_stats = {"error": str(e)}


        base_stats.update({
            'is_running': self.is_running,
            'current_time': self.current_time, # This is IntegratedMultiTimescaleDynamics's current_time
            'update_counts': dict(self.update_counts),
            'synapse_utilization': len(self.synapse_mapping.trace_to_synapse) / self.synapse_mapping.max_synapses,
            'consolidation_engine_stats': consolidation_stats # Nest consolidation stats
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