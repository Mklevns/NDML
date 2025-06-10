# core/dynamics.py - Multi-Timescale Dynamics Engine for NDML
"""
Multi-Timescale Dynamics Engine for Neuromorphic Distributed Memory Layer (NDML)

This module implements a comprehensive temporal dynamics system inspired by 
biological neural networks, featuring five distinct timescales that operate
concurrently to provide realistic memory formation, consolidation, and retrieval.

Timescales:
- Fast Synaptic (5ms): Action potential propagation and synaptic transmission
- Calcium Plasticity (500ms): Calcium-dependent synaptic plasticity (STDP/LTP/LTD)
- Protein Synthesis (60s): Long-term memory consolidation via protein synthesis
- Homeostatic Scaling (1h): Network stability and activity regulation
- Systems Consolidation (24h): Cross-system memory reorganization

Author: NDML Team
Version: 1.0.0
"""

import asyncio
import numpy as np
import time
import torch  # ADD THIS LINE
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from enum import Enum

# Add ConsolidationState import
try:
    from .memory_trace import ConsolidationState
except ImportError:
    try:
        from core.memory_trace import ConsolidationState
    except ImportError:
        from memory_trace import ConsolidationState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Configuration
TIMESCALE_CONFIGS = {
    "fast_synaptic": {
        "base_duration": 0.005,      # 5ms
        "min_duration": 0.001,       # 1ms
        "max_duration": 0.02,        # 20ms
        "adaptation_rate": 0.2,
        "stability_threshold": 0.02
    },
    "calcium_plasticity": {
        "base_duration": 0.5,        # 500ms
        "min_duration": 0.1,         # 100ms
        "max_duration": 2.0,         # 2s
        "adaptation_rate": 0.15,
        "stability_threshold": 0.1
    },
    "protein_synthesis": {
        "base_duration": 60.0,       # 1 minute
        "min_duration": 30.0,        # 30s
        "max_duration": 300.0,       # 5 minutes
        "adaptation_rate": 0.1,
        "stability_threshold": 0.5
    },
    "homeostatic_scaling": {
        "base_duration": 3600.0,     # 1 hour
        "min_duration": 1800.0,      # 30 minutes
        "max_duration": 7200.0,      # 2 hours
        "adaptation_rate": 0.05,
        "stability_threshold": 2.0
    },
    "systems_consolidation": {
        "base_duration": 86400.0,    # 24 hours
        "min_duration": 43200.0,     # 12 hours
        "max_duration": 172800.0,    # 48 hours
        "adaptation_rate": 0.02,
        "stability_threshold": 10.0
    }
}

@dataclass
class TemporalEvent:
    """Represents an event in the temporal dynamics system."""
    event_type: str
    timestamp: float
    source_process: str
    data: Dict[str, Any]
    priority: int = 1

@dataclass
class TemporalState:
    """Complete temporal state across all timescales."""
    fast_synaptic: Dict[str, Any] = field(default_factory=dict)
    calcium_plasticity: Dict[str, Any] = field(default_factory=dict)
    protein_synthesis: Dict[str, Any] = field(default_factory=dict)
    homeostatic_scaling: Dict[str, Any] = field(default_factory=dict)
    systems_consolidation: Dict[str, Any] = field(default_factory=dict)
    coherence_metric: float = 1.0
    stability_metric: float = 1.0
    timestamp: float = field(default_factory=time.time)

class ProcessStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ADAPTING = "adapting"
    CONSOLIDATING = "consolidating"

# Base Classes

class TemporalProcess(ABC):
    """Abstract base class for temporal processes."""
    
    def __init__(self, process_name: str, config: Dict[str, Any]):
        self.process_name = process_name
        self.config = config
        self.status = ProcessStatus.INACTIVE
        
        # Timing parameters
        self.current_duration = config["base_duration"]
        self.adaptation_rate = config["adaptation_rate"]
        self.stability_threshold = config["stability_threshold"]
        
        # State management
        self.internal_state = {}
        self.last_update_time = 0.0
        self.performance_history = deque(maxlen=100)
        self.event_queue = asyncio.Queue()
        
        # Performance tracking
        self.performance_metrics = {
            "efficiency": 0.5,
            "stability": 1.0,
            "activity_level": 0.0,
            "adaptation_count": 0
        }
        
    @abstractmethod
    async def process_step(self, current_time: float, delta_time: float) -> Dict[str, Any]:
        """Execute one processing step for this timescale."""
        pass
        
    @abstractmethod
    def _create_spike_pair_filter(self):
        """Create spike pair timing filter for STDP-like dynamics"""
        return {
            'tau_pre': 20.0,  # Pre-synaptic window (ms)
            'tau_post': 20.0,  # Post-synaptic window (ms) 
            'window_size': 50.0,  # Total window size (ms)
            'amplitude': 1.0,  # Filter amplitude
            'enabled': True
        }

    def compute_performance_metric(self) -> float:
        """Compute current performance metric for this process."""
        pass
        
    async def activate(self) -> None:
        """Activate the temporal process."""
        self.status = ProcessStatus.ACTIVE
        logging.info(f"Activated temporal process: {self.process_name}")
        
    async def deactivate(self) -> None:
        """Deactivate the temporal process."""
        self.status = ProcessStatus.INACTIVE
        logging.info(f"Deactivated temporal process: {self.process_name}")
        
    async def inject_event(self, event: TemporalEvent) -> None:
        """Inject an event into this process."""
        await self.event_queue.put(event)
        
    async def _adaptive_duration_adjustment(self) -> None:
        """Adapt the process duration based on performance."""
        if len(self.performance_history) < 10:
            return
            
        recent_performance = np.mean(list(self.performance_history)[-10:])
        performance_variance = np.var(list(self.performance_history)[-10:])
        
        # Adjust duration based on stability
        if performance_variance < self.stability_threshold:
            # Stable performance, can speed up
            target_duration = max(
                self.config["min_duration"],
                self.current_duration * (1 - self.adaptation_rate)
            )
        else:
            # Unstable performance, slow down
            target_duration = min(
                self.config["max_duration"], 
                self.current_duration * (1 + self.adaptation_rate)
            )
            
        self.current_duration = target_duration
        self.performance_metrics["adaptation_count"] += 1

# Specific Temporal Process Implementations

class FastSynapticProcess(TemporalProcess):
    """Fast synaptic transmission process (milliseconds)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("fast_synaptic", config)
        
        # Fast synaptic state
        self.internal_state = {
            "spike_trains": defaultdict(list),      # neuron_id -> spike_times
            "synaptic_weights": {},                 # (pre, post) -> weight
            "membrane_potentials": {},              # neuron_id -> potential
            "refractory_periods": {},               # neuron_id -> remaining_time
            "spike_count": 0,
            "last_activity_burst": 0.0
        }
        
    async def process_step(self, current_time: float, delta_time: float) -> Dict[str, Any]:
        """Process fast synaptic dynamics."""
        
        if self.status != ProcessStatus.ACTIVE:
            return {}
            
        # Process queued events
        events_processed = 0
        while not self.event_queue.empty() and events_processed < 100:  # Limit processing
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.001)
                await self._handle_event(event, current_time)
                events_processed += 1
            except asyncio.TimeoutError:
                break
                
        # Update membrane potentials and detect spikes
        new_spikes = await self._update_membrane_dynamics(current_time, delta_time)
        
        # Update performance metrics
        await self._update_performance_metrics(current_time, new_spikes)
        
        # Adaptive duration adjustment
        await self._adaptive_duration_adjustment()
        
        self.last_update_time = current_time
        
        return {
            "new_spikes": new_spikes,
            "total_spikes": self.internal_state["spike_count"],
            "active_neurons": len([n for n, p in self.internal_state["membrane_potentials"].items() if p > -0.06]),
            "activity_level": self.performance_metrics["activity_level"]
        }
        
    async def _handle_event(self, event: TemporalEvent, current_time: float) -> None:
        """Handle events for fast synaptic process."""
        
        if event.event_type == "external_stimulus":
            # External stimulation of neurons
            neuron_ids = event.data.get("neuron_ids", [])
            stimulus_strength = event.data.get("strength", 0.02)  # 20mV
            
            for neuron_id in neuron_ids:
                current_potential = self.internal_state["membrane_potentials"].get(neuron_id, -0.07)
                self.internal_state["membrane_potentials"][neuron_id] = current_potential + stimulus_strength
                
        elif event.event_type == "memory_trace_activation":
            # Activation from memory trace
            trace_data = event.data.get("trace_data", {})
            salience = event.data.get("salience", 0.5)
            
            # Convert trace activation to spike pattern
            if salience > 0.7:  # High salience traces
                affected_neurons = list(range(min(10, len(trace_data.get("content", [])))))
                for neuron_id in affected_neurons:
                    self.internal_state["spike_trains"][neuron_id].append(current_time)
                    self.internal_state["spike_count"] += 1
                    
    async def _update_membrane_dynamics(self, current_time: float, delta_time: float) -> List[Tuple[int, float]]:
        """Update membrane potentials and detect spikes."""
        
        new_spikes = []
        spike_threshold = -0.055  # -55mV
        reset_potential = -0.08   # -80mV
        resting_potential = -0.07 # -70mV
        
        for neuron_id, potential in list(self.internal_state["membrane_potentials"].items()):
            
            # Check refractory period
            if neuron_id in self.internal_state["refractory_periods"]:
                self.internal_state["refractory_periods"][neuron_id] -= delta_time
                if self.internal_state["refractory_periods"][neuron_id] <= 0:
                    del self.internal_state["refractory_periods"][neuron_id]
                continue
                
            # Passive decay toward resting potential
            decay_factor = np.exp(-delta_time / 0.02)  # 20ms time constant
            new_potential = potential * decay_factor + resting_potential * (1 - decay_factor)
            
            # Check for spike
            if new_potential > spike_threshold:
                # Generate spike
                new_spikes.append((neuron_id, current_time))
                self.internal_state["spike_trains"][neuron_id].append(current_time)
                self.internal_state["spike_count"] += 1
                
                # Reset potential and enter refractory period
                self.internal_state["membrane_potentials"][neuron_id] = reset_potential
                self.internal_state["refractory_periods"][neuron_id] = 0.002  # 2ms refractory
            else:
                self.internal_state["membrane_potentials"][neuron_id] = new_potential
                
        # Clean old spike times (keep only last second)
        for neuron_id in self.internal_state["spike_trains"]:
            self.internal_state["spike_trains"][neuron_id] = [
                t for t in self.internal_state["spike_trains"][neuron_id]
                if current_time - t < 1.0
            ]
            
        return new_spikes
        
    async def _update_performance_metrics(self, current_time: float, new_spikes: List[Tuple[int, float]]) -> None:
        """Update performance metrics for fast synaptic process."""
        
        # Activity level based on recent spikes
        recent_spike_count = len([s for s in new_spikes if current_time - s[1] < 0.1])
        self.performance_metrics["activity_level"] = min(1.0, recent_spike_count / 10.0)
        
        # Efficiency based on spike pattern regularity
        if len(new_spikes) > 1:
            spike_intervals = [new_spikes[i+1][1] - new_spikes[i][1] for i in range(len(new_spikes)-1)]
            interval_variance = np.var(spike_intervals) if spike_intervals else 0
            efficiency = 1.0 / (1.0 + interval_variance * 1000)  # Scale variance
            self.performance_metrics["efficiency"] = efficiency
            
        # Add to performance history
        overall_performance = (self.performance_metrics["activity_level"] + 
                             self.performance_metrics["efficiency"]) / 2.0
        self.performance_history.append(overall_performance)
        
    def compute_performance_metric(self) -> float:
        """Compute current performance metric."""
        if len(self.performance_history) == 0:
            return 0.5
        return float(np.mean(list(self.performance_history)[-10:]))
    def _create_spike_pair_filter(self):
        """Create spike pair timing filter for STDP-like dynamics"""
        return {
            'tau_pre': 20.0,  # Pre-synaptic window (ms)
            'tau_post': 20.0,  # Post-synaptic window (ms) 
            'window_size': 50.0,  # Total window size (ms)
            'amplitude': 1.0,  # Filter amplitude
            'enabled': True
        }


class CalciumPlasticityProcess(TemporalProcess):
    """Calcium-dependent plasticity process (hundreds of milliseconds)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("calcium_plasticity", config)
        
        # Calcium dynamics state
        self.internal_state = {
            "calcium_levels": {},           # (pre, post) -> [Ca2+] concentration
            "plasticity_events": [],        # List of recent plasticity events
            "ltp_events": 0,               # Long-term potentiation count
            "ltd_events": 0,               # Long-term depression count
            "stdp_windows": {},            # STDP timing windows
            "calcium_decay_rate": 0.1      # Calcium decay time constant
        }
        
    async def process_step(self, current_time: float, delta_time: float) -> Dict[str, Any]:
        """Process calcium plasticity dynamics."""
        
        if self.status != ProcessStatus.ACTIVE:
            return {}
            
        # Process events from fast synaptic process
        await self._process_spike_pair_events(current_time)
        
        # Update calcium dynamics
        await self._update_calcium_dynamics(current_time, delta_time)
        
        # Detect plasticity events
        plasticity_events = await self._detect_plasticity_events(current_time)
        
        # Update performance metrics
        await self._update_performance_metrics(current_time, plasticity_events)
        
        await self._adaptive_duration_adjustment()
        
        self.last_update_time = current_time
        
        return {
            "plasticity_events": plasticity_events,
            "active_synapses": len(self.internal_state["calcium_levels"]),
            "ltp_rate": self.internal_state["ltp_events"] / max(1, len(plasticity_events)),
            "calcium_activity": np.mean(list(self.internal_state["calcium_levels"].values())) if self.internal_state["calcium_levels"] else 0
        }
        
    async def _process_spike_pair_events(self, current_time: float) -> None:
        """Process spike pairs from fast synaptic process."""
        
        # In a real implementation, this would receive spike pair data
        # from the fast synaptic process via cross-scale communication
        
        while not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.001)
                
                if event.event_type == "spike_pair":
                    pre_neuron = event.data.get("pre_neuron")
                    post_neuron = event.data.get("post_neuron")
                    timing_diff = event.data.get("timing_difference", 0.0)
                    
                    if pre_neuron is not None and post_neuron is not None:
                        await self._handle_spike_pair(pre_neuron, post_neuron, timing_diff, current_time)
                        
            except asyncio.TimeoutError:
                break
                
    async def _handle_spike_pair(self, pre_neuron: int, post_neuron: int, 
                               timing_diff: float, current_time: float) -> None:
        """Handle a spike pair for STDP."""
        
        synapse_key = (pre_neuron, post_neuron)
        
        # Calculate calcium influx based on STDP timing
        if abs(timing_diff) < 0.1:  # Within 100ms STDP window
            if timing_diff > 0:  # Pre before post (LTP)
                calcium_influx = 0.5 * np.exp(-timing_diff / 0.02)  # 20ms decay
            else:  # Post before pre (LTD)  
                calcium_influx = -0.3 * np.exp(timing_diff / 0.02)
                
            # Update calcium level
            current_calcium = self.internal_state["calcium_levels"].get(synapse_key, 0.0)
            new_calcium = max(0.0, min(2.0, current_calcium + calcium_influx))  # Clamp [0, 2]
            self.internal_state["calcium_levels"][synapse_key] = new_calcium
            
            # Record STDP event
            self.internal_state["stdp_windows"][synapse_key] = {
                "timing_diff": timing_diff,
                "calcium_level": new_calcium,
                "timestamp": current_time
            }
            
    async def _update_calcium_dynamics(self, current_time: float, delta_time: float) -> None:
        """Update calcium concentration dynamics."""
        
        decay_rate = self.internal_state["calcium_decay_rate"]
        decay_factor = np.exp(-delta_time / decay_rate)
        
        # Decay all calcium levels
        synapses_to_remove = []
        for synapse_key, calcium_level in self.internal_state["calcium_levels"].items():
            new_level = calcium_level * decay_factor
            
            if new_level < 0.01:  # Below threshold, remove
                synapses_to_remove.append(synapse_key)
            else:
                self.internal_state["calcium_levels"][synapse_key] = new_level
                
        # Remove inactive synapses
        for synapse_key in synapses_to_remove:
            del self.internal_state["calcium_levels"][synapse_key]
            if synapse_key in self.internal_state["stdp_windows"]:
                del self.internal_state["stdp_windows"][synapse_key]
                
    async def _detect_plasticity_events(self, current_time: float) -> List[Dict[str, Any]]:
        """Detect plasticity events based on calcium levels."""
        
        plasticity_events = []
        ltp_threshold = 0.8
        ltd_threshold = 0.3
        
        for synapse_key, calcium_level in self.internal_state["calcium_levels"].items():
            
            plasticity_type = None
            plasticity_strength = 0.0
            
            if calcium_level > ltp_threshold:
                plasticity_type = "LTP"
                plasticity_strength = (calcium_level - ltp_threshold) / (2.0 - ltp_threshold)
                self.internal_state["ltp_events"] += 1
                
            elif calcium_level > ltd_threshold:
                plasticity_type = "LTD"  
                plasticity_strength = (calcium_level - ltd_threshold) / (ltp_threshold - ltd_threshold)
                self.internal_state["ltd_events"] += 1
                
            if plasticity_type:
                event = {
                    "synapse": synapse_key,
                    "type": plasticity_type,
                    "strength": plasticity_strength,
                    "calcium_level": calcium_level,
                    "timestamp": current_time
                }
                plasticity_events.append(event)
                self.internal_state["plasticity_events"].append(event)
                
        # Clean old plasticity events (keep last 10 seconds)
        self.internal_state["plasticity_events"] = [
            event for event in self.internal_state["plasticity_events"]
            if current_time - event["timestamp"] < 10.0
        ]
        
        return plasticity_events
        
    async def _update_performance_metrics(self, current_time: float, 
                                        plasticity_events: List[Dict[str, Any]]) -> None:
        """Update performance metrics for calcium plasticity."""
        
        # Activity level based on recent plasticity events
        recent_events = len([e for e in plasticity_events if current_time - e["timestamp"] < 1.0])
        self.performance_metrics["activity_level"] = min(1.0, recent_events / 5.0)
        
        # Efficiency based on LTP/LTD balance
        if self.internal_state["ltp_events"] + self.internal_state["ltd_events"] > 0:
            ltp_ratio = self.internal_state["ltp_events"] / (self.internal_state["ltp_events"] + self.internal_state["ltd_events"])
            # Optimal balance around 60% LTP, 40% LTD
            efficiency = 1.0 - abs(ltp_ratio - 0.6) / 0.6
            self.performance_metrics["efficiency"] = efficiency
            
        # Overall performance
        overall_performance = (self.performance_metrics["activity_level"] + 
                             self.performance_metrics["efficiency"]) / 2.0
        self.performance_history.append(overall_performance)
        
    def compute_performance_metric(self) -> float:
        """Compute current performance metric."""
        if len(self.performance_history) == 0:
            return 0.5
        return float(np.mean(list(self.performance_history)[-10:]))
    def _create_spike_pair_filter(self):
        """Create spike pair timing filter for STDP-like dynamics"""
        return {
            'tau_pre': 20.0,  # Pre-synaptic window (ms)
            'tau_post': 20.0,  # Post-synaptic window (ms) 
            'window_size': 50.0,  # Total window size (ms)
            'amplitude': 1.0,  # Filter amplitude
            'enabled': True
        }


class ProteinSynthesisProcess(TemporalProcess):
    """Protein synthesis-dependent memory consolidation (minutes to hours)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("protein_synthesis", config)
        
        # Protein synthesis state
        self.internal_state = {
            "consolidation_queue": [],          # Memories waiting for consolidation
            "active_consolidations": {},        # Currently consolidating memories
            "consolidated_memories": {},        # Successfully consolidated memories
            "consolidation_rate": 0.01,         # Rate of consolidation per step
            "consolidation_threshold": 0.7,     # Threshold for initiating consolidation
            "synthesis_capacity": 10            # Max concurrent consolidations
        }
        
    async def process_step(self, current_time: float, delta_time: float) -> Dict[str, Any]:
        """Process protein synthesis dynamics."""
        
        if self.status != ProcessStatus.ACTIVE:
            return {}
            
        # Process consolidation requests
        await self._process_consolidation_requests(current_time)
        
        # Update active consolidations
        newly_consolidated = await self._update_consolidations(current_time, delta_time)
        
        # Initiate new consolidations if capacity allows
        await self._initiate_new_consolidations(current_time)
        
        # Update performance metrics
        await self._update_performance_metrics(current_time, newly_consolidated)
        
        await self._adaptive_duration_adjustment()
        
        self.last_update_time = current_time
        
        return {
            "newly_consolidated": newly_consolidated,
            "active_consolidations": len(self.internal_state["active_consolidations"]),
            "consolidated_total": len(self.internal_state["consolidated_memories"]),
            "queue_length": len(self.internal_state["consolidation_queue"])
        }
        
    async def _process_consolidation_requests(self, current_time: float) -> None:
        """Process incoming consolidation requests."""
        
        while not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.001)
                
                if event.event_type == "consolidation_request":
                    memory_data = event.data.get("memory_data", {})
                    priority = event.data.get("priority", 1)
                    
                    consolidation_item = {
                        "memory_id": memory_data.get("trace_id", f"mem_{current_time}"),
                        "memory_data": memory_data,
                        "priority": priority,
                        "request_time": current_time,
                        "salience": memory_data.get("salience", 0.5)
                    }
                    
                    self.internal_state["consolidation_queue"].append(consolidation_item)
                    
            except asyncio.TimeoutError:
                break
                
        # Sort queue by priority and salience
        self.internal_state["consolidation_queue"].sort(
            key=lambda x: (x["priority"], x["salience"]), reverse=True
        )
        
    async def _update_consolidations(self, current_time: float, delta_time: float) -> List[str]:
        """Update active consolidations and return newly consolidated memory IDs."""
        
        newly_consolidated = []
        completed_consolidations = []
        
        for memory_id, consolidation_data in self.internal_state["active_consolidations"].items():
            
            # Update consolidation progress
            progress = consolidation_data["progress"]
            consolidation_rate = self.internal_state["consolidation_rate"]
            
            # Progress depends on salience and time
            salience_factor = consolidation_data.get("salience", 0.5)
            time_factor = min(1.0, (current_time - consolidation_data["start_time"]) / 60.0)  # 1 minute baseline
            
            progress_increment = consolidation_rate * salience_factor * time_factor * delta_time
            new_progress = min(1.0, progress + progress_increment)
            
            consolidation_data["progress"] = new_progress
            
            # Check if consolidation is complete
            if new_progress >= 1.0:
                # Move to consolidated memories
                self.internal_state["consolidated_memories"][memory_id] = {
                    "memory_data": consolidation_data["memory_data"],
                    "consolidation_time": current_time,
                    "consolidation_strength": salience_factor,
                    "stability": 1.0
                }
                
                newly_consolidated.append(memory_id)
                completed_consolidations.append(memory_id)
                
        # Remove completed consolidations
        for memory_id in completed_consolidations:
            del self.internal_state["active_consolidations"][memory_id]
            
        return newly_consolidated
        
    async def _initiate_new_consolidations(self, current_time: float) -> None:
        """Initiate new consolidations from the queue."""
        
        available_capacity = (self.internal_state["synthesis_capacity"] - 
                            len(self.internal_state["active_consolidations"]))
        
        items_to_consolidate = min(available_capacity, len(self.internal_state["consolidation_queue"]))
        
        for _ in range(items_to_consolidate):
            if not self.internal_state["consolidation_queue"]:
                break
                
            item = self.internal_state["consolidation_queue"].pop(0)
            
            # Check if meets consolidation threshold
            if item["salience"] >= self.internal_state["consolidation_threshold"]:
                
                memory_id = item["memory_id"]
                self.internal_state["active_consolidations"][memory_id] = {
                    "memory_data": item["memory_data"],
                    "start_time": current_time,
                    "progress": 0.0,
                    "salience": item["salience"],
                    "priority": item["priority"]
                }
                
    async def _update_performance_metrics(self, current_time: float, 
                                        newly_consolidated: List[str]) -> None:
        """Update performance metrics for protein synthesis."""
        
        # Activity level based on consolidation throughput
        consolidation_rate = len(newly_consolidated) / max(1, len(self.internal_state["active_consolidations"]))
        self.performance_metrics["activity_level"] = min(1.0, consolidation_rate)
        
        # Efficiency based on queue management
        queue_efficiency = 1.0 - (len(self.internal_state["consolidation_queue"]) / 
                                 max(1, self.internal_state["synthesis_capacity"]))
        self.performance_metrics["efficiency"] = max(0.0, queue_efficiency)
        
        # Overall performance
        overall_performance = (self.performance_metrics["activity_level"] + 
                             self.performance_metrics["efficiency"]) / 2.0
        self.performance_history.append(overall_performance)
        
    def compute_performance_metric(self) -> float:
        """Compute current performance metric."""
        if len(self.performance_history) == 0:
            return 0.5
        return float(np.mean(list(self.performance_history)[-10:]))

# Cross-Scale Communication System
    def _create_spike_pair_filter(self):
        """Create spike pair timing filter for STDP-like dynamics"""
        return {
            'tau_pre': 20.0,  # Pre-synaptic window (ms)
            'tau_post': 20.0,  # Post-synaptic window (ms) 
            'window_size': 50.0,  # Total window size (ms)
            'amplitude': 1.0,  # Filter amplitude
            'enabled': True
        }


class CrossScaleCommunicator:
    """Manages communication between different timescale processes."""
    
    def __init__(self, processes: Dict[str, TemporalProcess]):
        self.processes = processes
        self.communication_channels = {}
        self.message_buffer = defaultdict(list)
    def _spike_pair_filter(self, spike_data):
        """Filter spike pairs for STDP-like communication."""
        if not isinstance(spike_data, dict):
            return spike_data
        return {
            "pre_spikes": spike_data.get("pre_spikes", []),
            "post_spikes": spike_data.get("post_spikes", []),
            "timing_window": 20.0,
            "plasticity_threshold": 0.1
        }
        self.message_buffer =         self._spike_pair_filter = self._create_spike_pair_filter()
        defaultdict(list)
        
    async def setup_communication_channels(self) -> None:
        """Set up communication channels between processes."""
        
        # Fast synaptic -> Calcium plasticity
        self.communication_channels["fast_to_calcium"] = {
            "source": "fast_synaptic",
            "target": "calcium_plasticity", 
            "message_type": "spike_pair",
            "filter": self._spike_pair_filter
        }
        
        # Calcium plasticity -> Protein synthesis
        self.communication_channels["calcium_to_protein"] = {
            "source": "calcium_plasticity",
            "target": "protein_synthesis",
            "message_type": "consolidation_request",
            "filter": self._plasticity_filter
        }
        
    async def transfer_information(self, current_time: float) -> None:
        """Transfer information between timescale processes."""
        
        # Fast synaptic -> Calcium plasticity (spike pairs)
        await self._transfer_spike_pairs(current_time)
        
        # Calcium plasticity -> Protein synthesis (plasticity events)
        await self._transfer_plasticity_events(current_time)
        
        # Protein synthesis -> Systems consolidation (consolidated memories)
        await self._transfer_consolidated_memories(current_time)
        
    async def _transfer_spike_pairs(self, current_time: float) -> None:
        """Transfer spike pair information from fast synaptic to calcium plasticity."""
        
        if ("fast_synaptic" not in self.processes or 
            "calcium_plasticity" not in self.processes):
            return
            
        fast_process = self.processes["fast_synaptic"]
        calcium_process = self.processes["calcium_plasticity"]
        
        # Get recent spike trains from fast synaptic
        spike_trains = fast_process.internal_state.get("spike_trains", {})
        
        # Find spike pairs within STDP window
        spike_pairs = []
        neuron_ids = list(spike_trains.keys())
        
        for i, pre_neuron in enumerate(neuron_ids):
            for j, post_neuron in enumerate(neuron_ids[i+1:], i+1):
                
                pre_spikes = [t for t in spike_trains[pre_neuron] if current_time - t < 0.1]
                post_spikes = [t for t in spike_trains[post_neuron] if current_time - t < 0.1]
                
                # Find coincident spikes
                for pre_time in pre_spikes:
                    for post_time in post_spikes:
                        timing_diff = post_time - pre_time
                        if abs(timing_diff) < 0.1:  # 100ms STDP window
                            
                            spike_pair_event = TemporalEvent(
                                event_type="spike_pair",
                                timestamp=current_time,
                                source_process="fast_synaptic",
                                data={
                                    "pre_neuron": pre_neuron,
                                    "post_neuron": post_neuron,
                                    "timing_difference": timing_diff,
                                    "pre_time": pre_time,
                                    "post_time": post_time
                                }
                            )
                            
                            await calcium_process.inject_event(spike_pair_event)
                            
    async def _transfer_plasticity_events(self, current_time: float) -> None:
        """Transfer plasticity events from calcium to protein synthesis."""
        
        if ("calcium_plasticity" not in self.processes or 
            "protein_synthesis" not in self.processes):
            return
            
        calcium_process = self.processes["calcium_plasticity"]
        protein_process = self.processes["protein_synthesis"]
        
        # Get recent plasticity events
        plasticity_events = calcium_process.internal_state.get("plasticity_events", [])
        recent_events = [e for e in plasticity_events if current_time - e["timestamp"] < 1.0]
        
        # Group events by synapse and create consolidation requests
        synapse_groups = defaultdict(list)
        for event in recent_events:
            synapse_groups[event["synapse"]].append(event)
            
        for synapse, events in synapse_groups.items():
            if len(events) >= 3:  # Threshold for consolidation request
                
                avg_strength = np.mean([e["strength"] for e in events])
                predominant_type = max(set(e["type"] for e in events), 
                                     key=lambda x: sum(1 for e in events if e["type"] == x))
                
                consolidation_request = TemporalEvent(
                    event_type="consolidation_request",
                    timestamp=current_time,
                    source_process="calcium_plasticity",
                    data={
                        "memory_data": {
                            "synapse": synapse,
                            "plasticity_type": predominant_type,
                            "strength": avg_strength,
                            "trace_id": f"synapse_{synapse[0]}_{synapse[1]}_{current_time}",
                            "salience": min(1.0, avg_strength * 1.2)
                        },
                        "priority": 2 if predominant_type == "LTP" else 1
                    }
                )
                
                await protein_process.inject_event(consolidation_request)
                
    async def _transfer_consolidated_memories(self, current_time: float) -> None:
        """Transfer consolidated memories to systems consolidation."""
        
        if ("protein_synthesis" not in self.processes or 
            "systems_consolidation" not in self.processes):
            return
            
        # This would transfer long-term consolidated memories
        # Implementation would depend on systems consolidation process
        pass

# Main Dynamics Engine
    def _spike_pair_filter(self, spike_data):
        """Filter spike pairs for STDP-like communication."""
        if not isinstance(spike_data, dict):
            return spike_data
        return {
            "pre_spikes": spike_data.get("pre_spikes", []),
            "post_spikes": spike_data.get("post_spikes", []),
            "timing_window": 20.0,
            "plasticity_threshold": 0.1
        }

    def _plasticity_filter(self, plasticity_data):
        """Filter plasticity events for calcium-protein communication."""
        if not isinstance(plasticity_data, dict):
            return plasticity_data
        return {
            "calcium_level": plasticity_data.get("calcium_level", 0.0),
            "protein_threshold": 0.5,
            "consolidation_strength": plasticity_data.get("strength", 1.0),
            "duration_ms": plasticity_data.get("duration", 100.0)
        }

    def _consolidation_filter(self, consolidation_data):
        """Filter consolidation events for protein-homeostatic communication."""
        if not isinstance(consolidation_data, dict):
            return consolidation_data
        return {
            "consolidation_strength": consolidation_data.get("strength", 1.0),
            "memory_traces": consolidation_data.get("traces", []),
            "synaptic_tags": consolidation_data.get("tags", []),
            "permanence_factor": 0.8
        }

    def _homeostatic_filter(self, homeostatic_data):
        """Filter homeostatic signals for system-wide regulation."""
        if not isinstance(homeostatic_data, dict):
            return homeostatic_data
        return {
            "regulation_strength": homeostatic_data.get("strength", 1.0),
            "target_processes": homeostatic_data.get("targets", []),
            "adjustment_factor": 0.1,
            "stability_threshold": 0.05
        }


class MultiTimescaleDynamicsEngine:
    """Main multi-timescale dynamics engine for NDML."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.current_time = 0.0
        self.delta_time = 0.001  # 1ms default time step
        
        # Initialize temporal processes
        self.processes = self._initialize_processes()
        
        # Cross-scale communication
        self.communicator = CrossScaleCommunicator(self.processes)
        
        # Performance monitoring
        self.performance_monitor = TemporalPerformanceMonitor()
        
        # System state
        self.is_running = False
        self.step_count = 0
        
        # Integration interfaces (will be set by NDML components)
        self.btsp_bridge = None
        self.dmn_bridges = {}
        self.memory_gateway_bridge = None
        
    def _initialize_processes(self) -> Dict[str, TemporalProcess]:
        """Initialize all temporal processes."""
        
        processes = {}
        
        # Get configurations for each process
        fast_config = self.config.get("fast_synaptic", TIMESCALE_CONFIGS["fast_synaptic"])
        calcium_config = self.config.get("calcium_plasticity", TIMESCALE_CONFIGS["calcium_plasticity"])
        protein_config = self.config.get("protein_synthesis", TIMESCALE_CONFIGS["protein_synthesis"])
        
        # Initialize processes
        processes["fast_synaptic"] = FastSynapticProcess(fast_config)
        processes["calcium_plasticity"] = CalciumPlasticityProcess(calcium_config)
        processes["protein_synthesis"] = ProteinSynthesisProcess(protein_config)
        
        # TODO: Add homeostatic scaling and systems consolidation processes
        
        return processes
        
    async def start(self) -> None:
        """Start the dynamics engine."""
        
        self.is_running = True
        self.current_time = time.time()
        
        # Activate all processes
        for process in self.processes.values():
            await process.activate()
            
        # Setup communication channels
        await self.communicator.setup_communication_channels()
        
        logging.info("Multi-timescale dynamics engine started")
        
    async def stop(self) -> None:
        """Stop the dynamics engine."""
        
        self.is_running = False
        
        # Deactivate all processes
        for process in self.processes.values():
            await process.deactivate()
            
        logging.info("Multi-timescale dynamics engine stopped")
        
    async def step(self) -> TemporalState:
        """Execute one simulation step."""
        
        if not self.is_running:
            return TemporalState()
            
        prev_time = self.current_time
        self.current_time += self.delta_time
        
        # Execute all temporal processes
        process_outputs = {}
        
        for name, process in self.processes.items():
            if process.status == ProcessStatus.ACTIVE:
                output = await process.process_step(self.current_time, self.delta_time)
                process_outputs[name] = output
                
        # Cross-scale communication
        await self.communicator.transfer_information(self.current_time)
        
        # Update performance monitoring
        await self.performance_monitor.update(self.current_time, self.processes)
        
        # Create temporal state
        temporal_state = TemporalState(
            fast_synaptic=process_outputs.get("fast_synaptic", {}),
            calcium_plasticity=process_outputs.get("calcium_plasticity", {}),
            protein_synthesis=process_outputs.get("protein_synthesis", {}),
            homeostatic_scaling=process_outputs.get("homeostatic_scaling", {}),
            systems_consolidation=process_outputs.get("systems_consolidation", {}),
            coherence_metric=self._compute_temporal_coherence(),
            stability_metric=self._compute_system_stability(),
            timestamp=self.current_time
        )
        
        self.step_count += 1
        
        return temporal_state
    async def process_update_async(self, content: torch.Tensor, salience: float, calcium_level: float) -> bool:
    
        try:
            # Inject update event into temporal dynamics
            if salience > 0.7:  # High salience updates
                success = await self.inject_event(
                    "memory_update",
                    "fast_synaptic",
                    {
                        "content_norm": float(torch.norm(content)),
                        "salience": float(salience),
                        "calcium_level": float(calcium_level)
                    }
                )
                return success
            return True
        except Exception as e:
            logger.error(f"Error processing memory update: {e}")
            return False

    async def process_retrieval_async(self, query: torch.Tensor, retrieved_traces: List[Any]) -> bool:
        """Process memory retrieval through temporal dynamics."""
        try:
            if retrieved_traces:
                # Inject retrieval event
                success = await self.inject_event(
                    "memory_retrieval",
                    "calcium_plasticity",
                    {
                        "query_norm": float(torch.norm(query)),
                        "num_retrieved": len(retrieved_traces),
                        "avg_salience": float(np.mean([getattr(trace, 'salience', 0.5) for trace in retrieved_traces]))
                    }
                )
                return success
            return True
        except Exception as e:
            logger.error(f"Error processing memory retrieval: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive temporal dynamics statistics."""
        try:
            # Get performance summary from monitor
            perf_summary = self.performance_monitor.get_performance_summary()
            
            # Get process-level stats
            process_stats = {}
            for name, process in self.processes.items():
                if hasattr(process, 'performance_metrics'):
                    process_stats[name] = process.performance_metrics.copy()
            
            return {
                'system_health': perf_summary.get('system_health', 0.8),
                'step_count': getattr(self, 'step_count', 0),
                'current_time': getattr(self, 'current_time', 0.0),
                'is_running': getattr(self, 'is_running', False),
                'temporal_coherence': self._compute_temporal_coherence(),
                'system_stability': self._compute_system_stability(),
                'process_stats': process_stats,
                'performance_summary': perf_summary
            }
        except Exception as e:
            logger.error(f"Error getting dynamics stats: {e}")
            return {
                'system_health': 0.8,
                'step_count': 0,
                'current_time': 0.0,
                'is_running': False,
                'error': str(e)
            }

    async def initiate_consolidation(self, trace: Any, current_time: float) -> None:
        """Initiate consolidation for a memory trace."""
        try:
            if hasattr(trace, 'temporal_metadata'):
                trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATING
                
            # Inject consolidation event
            await self.inject_event(
                "consolidation_initiation",
                "protein_synthesis",
                {
                    "trace_id": getattr(trace, 'trace_id', 'unknown'),
                    "salience": getattr(trace, 'salience', 0.5),
                    "current_time": current_time
                }
            )
        except Exception as e:
            logger.error(f"Error initiating consolidation: {e}")

    async def update_consolidation_progress(self, trace: Any, current_time: float, progress_rate: float) -> None:
        """Update consolidation progress for a memory trace."""
        try:
            if hasattr(trace, 'temporal_metadata'):
                current_progress = getattr(trace.temporal_metadata, 'consolidation_strength', 0.0)
                new_progress = min(1.0, current_progress + progress_rate * 0.01)
                trace.temporal_metadata.consolidation_strength = new_progress
                
                # Check if consolidation is complete
                if new_progress >= 0.95:
                    trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATED
                    
        except Exception as e:
            logger.error(f"Error updating consolidation progress: {e}")
        
    def _compute_temporal_coherence(self) -> float:
        """Compute temporal coherence across all timescales."""
        
        performance_values = []
        for process in self.processes.values():
            if process.status == ProcessStatus.ACTIVE:
                performance_values.append(process.compute_performance_metric())
                
        if not performance_values:
            return 1.0
            
        # Coherence is inverse of performance variance
        variance = np.var(performance_values)
        mean_performance = np.mean(performance_values)
        
        coherence = 1.0 / (1.0 + variance / (mean_performance + 1e-6))
        return float(coherence)
        
    def _compute_system_stability(self) -> float:
        """Compute overall system stability."""
        
        stability_values = []
        for process in self.processes.values():
            if process.status == ProcessStatus.ACTIVE:
                stability_values.append(process.performance_metrics.get("stability", 1.0))
                
        if not stability_values:
            return 1.0
            
        return float(np.mean(stability_values))
        
    async def inject_event(self, event_type: str, target_process: str, data: Dict[str, Any]) -> bool:
        """Inject an event into a specific temporal process."""
        
        if target_process not in self.processes:
            logging.warning(f"Target process {target_process} not found")
            return False
            
        event = TemporalEvent(
            event_type=event_type,
            timestamp=self.current_time,
            source_process="external",
            data=data
        )
        
        await self.processes[target_process].inject_event(event)
        return True
        
    def get_temporal_state(self) -> TemporalState:
        """Get current temporal state."""
        
        return TemporalState(
            fast_synaptic=self.processes.get("fast_synaptic", {}).internal_state if "fast_synaptic" in self.processes else {},
            calcium_plasticity=self.processes.get("calcium_plasticity", {}).internal_state if "calcium_plasticity" in self.processes else {},
            protein_synthesis=self.processes.get("protein_synthesis", {}).internal_state if "protein_synthesis" in self.processes else {},
            coherence_metric=self._compute_temporal_coherence(),
            stability_metric=self._compute_system_stability(),
            timestamp=self.current_time
        )
        
    # Integration methods (for NDML components)
    
    def integrate_with_btsp(self, btsp_mechanism) -> None:
        """Integrate with BTSP mechanism."""
        self.btsp_bridge = btsp_mechanism
        logging.info("Integrated with BTSP mechanism")
        
    def integrate_with_dmn(self, dmn_id: str, dmn_instance) -> None:
        """Integrate with a Distributed Memory Node."""
        self.dmn_bridges[dmn_id] = dmn_instance
        logging.info(f"Integrated with DMN: {dmn_id}")
        
    def integrate_with_memory_gateway(self, memory_gateway) -> None:
        """Integrate with memory gateway."""
        self.memory_gateway_bridge = memory_gateway
        logging.info("Integrated with memory gateway")
        
    
    async def process_update_async(self, content: torch.Tensor, salience: float, calcium_level: float) -> bool:
        
        try:
            # Inject update event into fast synaptic process
            if salience > 0.5:  # Only process significant updates
                await self.inject_event(
                    "memory_update",
                    "fast_synaptic",
                    {
                        "content": content.detach().cpu().numpy().tolist() if hasattr(content, 'numpy') else content,
                        "salience": salience,
                        "calcium_level": calcium_level,
                        "timestamp": self.current_time
                    }
                )
            
            # Also inject into calcium plasticity if calcium level is high
            if calcium_level > 0.7:
                await self.inject_event(
                    "high_calcium_update",
                    "calcium_plasticity",
                    {
                        "calcium_level": calcium_level,
                        "salience": salience,
                        "timestamp": self.current_time
                    }
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing update in dynamics engine: {e}")
            return False
        
    async def process_retrieval_async(self, query: torch.Tensor, retrieved_traces: List[Any]) -> bool:
    
        try:
            if not retrieved_traces:
                return True
                
            # Calculate retrieval pattern statistics
            num_traces = len(retrieved_traces)
            avg_salience = np.mean([trace.salience for trace in retrieved_traces])
            
            # Inject retrieval event
            await self.inject_event(
                "retrieval_pattern",
                "fast_synaptic",
                {
                    "query_norm": float(torch.norm(query)) if hasattr(query, 'norm') else 1.0,
                    "num_retrieved": num_traces,
                    "average_salience": avg_salience,
                    "timestamp": self.current_time
                }
            )
            
            # Process spike pairs if multiple traces retrieved
            if num_traces > 1:
                for i in range(min(5, num_traces-1)):  # Process up to 5 pairs
                    await self.inject_event(
                        "spike_pair",
                        "calcium_plasticity",
                        {
                            "pre_neuron": i,
                            "post_neuron": i+1,
                            "timing_difference": 0.01 * (i+1),  # Simulated timing
                            "timestamp": self.current_time
                        }
                    )
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing retrieval in dynamics engine: {e}")
            return False

    async def initiate_consolidation(self, trace: Any, current_time: float) -> bool:
        
        try:
            await self.inject_event(
                "consolidation_request",
                "protein_synthesis",
                {
                    "memory_data": {
                        "trace_id": getattr(trace, 'trace_id', f'trace_{current_time}'),
                        "salience": getattr(trace, 'salience', 0.5),
                        "content": getattr(trace, 'content', None),
                        "consolidation_strength": 0.0
                    },
                    "priority": 1,
                    "timestamp": current_time
                }
            )
            return True
        except Exception as e:
            logging.error(f"Error initiating consolidation: {e}")
            return False

    async def update_consolidation_progress(self, trace: Any, current_time: float, progress: float) -> bool:
        
        try:
            # This is a placeholder - actual implementation would update internal state
            return True
        except Exception as e:
            logging.error(f"Error updating consolidation progress: {e}")
            return False
        
        

class TemporalPerformanceMonitor:
    """Monitors performance of the temporal dynamics system."""
    
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.system_health = 1.0
        
    async def update(self, current_time: float, processes: Dict[str, TemporalProcess]) -> None:
        """Update performance metrics."""
        
        self.metrics["timestamp"].append(current_time)
        
        # Process-level metrics
        for name, process in processes.items():
            if process.status == ProcessStatus.ACTIVE:
                perf_metric = process.compute_performance_metric()
                self.metrics[f"{name}_performance"].append(perf_metric)
                
        # System-level health
        self._update_system_health(processes)
        
        # Keep only recent metrics (last 1000 entries)
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = deque(list(self.metrics[key])[-1000:], maxlen=1000)
                
    def _update_system_health(self, processes: Dict[str, TemporalProcess]) -> None:
        """Update overall system health metric."""
        
        active_processes = [p for p in processes.values() if p.status == ProcessStatus.ACTIVE]
        
        if not active_processes:
            self.system_health = 0.0
            return
            
        health_components = []
        
        for process in active_processes:
            performance = process.compute_performance_metric()
            stability = process.performance_metrics.get("stability", 1.0)
            
            process_health = (performance + stability) / 2.0
            health_components.append(process_health)
            
        self.system_health = np.mean(health_components)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        
        summary = {
            "system_health": self.system_health,
            "process_performance": {},
            "recent_metrics": {}
        }
        
        # Process performance
        for key in self.metrics:
            if key.endswith("_performance") and self.metrics[key]:
                process_name = key.replace("_performance", "")
                recent_values = list(self.metrics[key])[-10:]
                summary["process_performance"][process_name] = {
                    "current": recent_values[-1] if recent_values else 0.0,
                    "average": float(np.mean(recent_values)),
                    "trend": float(np.polyfit(range(len(recent_values)), recent_values, 1)[0]) if len(recent_values) > 1 else 0.0
                }
                
        return summary

# Example usage and testing
async def main():
    """Example usage of the multi-timescale dynamics engine."""
    
    logging.info("Initializing Multi-Timescale Dynamics Engine for NDML...")
    
    # Create dynamics engine
    engine = MultiTimescaleDynamicsEngine()
    
    # Start the engine
    await engine.start()
    
    # Run simulation
    for step in range(100):
        
        # Inject some test events
        if step % 10 == 0:
            await engine.inject_event(
                "external_stimulus",
                "fast_synaptic", 
                {"neuron_ids": [0, 1, 2], "strength": 0.03}
            )
            
        if step % 25 == 0:
            await engine.inject_event(
                "memory_trace_activation",
                "fast_synaptic",
                {
                    "trace_data": {"content": np.random.randn(10)},
                    "salience": 0.8
                }
            )
            
        # Execute step
        temporal_state = await engine.step()
        
        # Log progress
        if step % 20 == 0:
            performance_summary = engine.performance_monitor.get_performance_summary()
            logging.info(f"Step {step}: System Health = {performance_summary['system_health']:.3f}")
            
            for process_name, metrics in performance_summary["process_performance"].items():
                logging.info(f"  {process_name}: Performance = {metrics['current']:.3f}")
                
    # Stop the engine
    await engine.stop()
    
    logging.info("Multi-timescale dynamics engine demo completed")

if __name__ == "__main__":
    asyncio.run(main())
