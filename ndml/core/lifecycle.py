# core/lifecycle.py - Enhanced Production Version
import asyncio
import torch
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryLifecycleState(Enum):
    ACTIVE = "active"
    AGING = "aging"
    CONSOLIDATING = "consolidating"
    ARCHIVED = "archived"
    EVICTION_CANDIDATE = "eviction_candidate"

class ConsolidationState(Enum):
    INITIAL = "initial"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    STABLE = "stable"

@dataclass
class LifecycleConfig:
    """Configuration parameters for lifecycle management."""
    eviction_batch_size: int = 50
    consolidation_interval: float = 3600.0
    maintenance_interval: float = 1800.0
    age_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'aging_threshold': 3600.0,  # 1 hour
        'eviction_threshold': 86400.0,  # 24 hours
        'archive_threshold': 604800.0,  # 1 week
    })
    decay_params: Dict[str, float] = field(default_factory=lambda: {
        'age_scale': 86400.0,
        'recency_scale': 3600.0,
        'base_decay_rate': 0.0001,
        'salience_weight': 0.5,
        'consolidation_protection': 2.0,
        'association_boost': 0.1,
    })
    consolidation_params: Dict[str, float] = field(default_factory=lambda: {
        'readiness_age': 60.0,  # 1 minute
        'progress_rate': 0.01,
        'success_threshold': 0.95,
        'stability_time': 7200.0,  # 2 hours
    })

class MemoryLifecycleManager:
    """
    Enhanced memory lifecycle manager with production-quality features.
    """

    def __init__(self,
                 node_id: str,
                 config: Optional[LifecycleConfig] = None,
                 dynamics_engine: Optional[Any] = None):
        """
        Initialize the MemoryLifecycleManager.

        Args:
            node_id: Unique identifier for the memory node
            config: Lifecycle configuration parameters
            dynamics_engine: Reference to multi-timescale dynamics engine
        """
        self.node_id = node_id
        self.config = config or LifecycleConfig()
        self.dynamics_engine = dynamics_engine

        # State tracking
        self.last_consolidation_time = 0.0
        self.last_maintenance_time = 0.0
        self._lock = asyncio.Lock()

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.operation_timings = defaultdict(list)

        # Initialize statistics
        self.stats = self._initialize_stats()

        # Lifecycle transition tracking
        self.transition_history = deque(maxlen=10000)
        self.consolidation_queue = asyncio.Queue()

        logger.info(f"MemoryLifecycleManager initialized for node {node_id}")

    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize comprehensive statistics tracking."""
        return {
            'total_traces': 0,
            'state_counts': {state.value: 0 for state in MemoryLifecycleState},
            'consolidation_counts': {state.value: 0 for state in ConsolidationState},
            'eviction_stats': {
                'total_evicted': 0,
                'eviction_rate': 0.0,
                'average_eviction_priority': 0.0,
            },
            'consolidation_stats': {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'average_duration': 0.0,
            },
            'performance_metrics': {
                'maintenance_cycle_time': 0.0,
                'eviction_selection_time': 0.0,
                'consolidation_update_time': 0.0,
            },
            'last_maintenance_time': 0.0,
            'last_consolidation_time': 0.0,
        }

    async def evaluate_trace_lifecycle(self,
                                     trace: Any,
                                     current_time: float) -> MemoryLifecycleState:
        """
        Enhanced lifecycle state evaluation with comprehensive criteria.
        """
        try:
            age = current_time - trace.timestamp
            recency = current_time - trace.last_access

            # Check for explicit state assignments
            if hasattr(trace, 'state') and trace.state == MemoryLifecycleState.EVICTION_CANDIDATE:
                return MemoryLifecycleState.EVICTION_CANDIDATE

            # Consolidation-based states
            if hasattr(trace, 'temporal_metadata'):
                consolidation_state = trace.temporal_metadata.consolidation_state
                if consolidation_state == ConsolidationState.CONSOLIDATING:
                    return MemoryLifecycleState.CONSOLIDATING
                elif consolidation_state in [ConsolidationState.CONSOLIDATED, ConsolidationState.STABLE]:
                    # Consolidated traces can still age if not accessed
                    if recency > self.config.age_thresholds['aging_threshold'] * 2:
                        return MemoryLifecycleState.AGING
                    return MemoryLifecycleState.ACTIVE

            # Age-based transitions
            if age > self.config.age_thresholds['archive_threshold']:
                return MemoryLifecycleState.ARCHIVED
            elif age > self.config.age_thresholds['eviction_threshold']:
                return MemoryLifecycleState.EVICTION_CANDIDATE
            elif age > self.config.age_thresholds['aging_threshold']:
                return MemoryLifecycleState.AGING
            else:
                return MemoryLifecycleState.ACTIVE

        except Exception as e:
            logger.error(f"Error evaluating lifecycle for trace {getattr(trace, 'trace_id', 'unknown')}: {e}")
            return MemoryLifecycleState.ACTIVE

    async def compute_eviction_priority(self,
                                      trace: Any,
                                      current_time: float,
                                      system_pressure: float = 0.5) -> float:
        """
        Enhanced eviction priority calculation with system pressure awareness.
        """
        try:
            age = current_time - trace.timestamp
            recency = current_time - trace.last_access

            # Base decay calculation
            age_factor = np.exp(-age / self.config.decay_params['age_scale'])
            recency_penalty = np.exp(recency / self.config.decay_params['recency_scale'])

            # Salience protection
            salience = getattr(trace, 'current_salience', trace.salience)
            salience_factor = 1.0 + salience * self.config.decay_params['salience_weight']

            # Consolidation protection
            consolidation_factor = 1.0
            if hasattr(trace, 'temporal_metadata'):
                consolidation_state = trace.temporal_metadata.consolidation_state
                if consolidation_state == ConsolidationState.CONSOLIDATED:
                    consolidation_factor = self.config.decay_params['consolidation_protection']
                elif consolidation_state == ConsolidationState.STABLE:
                    consolidation_factor = self.config.decay_params['consolidation_protection'] * 1.5
                elif consolidation_state == ConsolidationState.CONSOLIDATING:
                    # Protect traces currently consolidating
                    progress = getattr(trace.temporal_metadata, 'consolidation_strength', 0.0)
                    consolidation_factor = 1.0 + progress * 0.5

            # Association network protection
            association_factor = 1.0
            if hasattr(trace, 'associated_traces'):
                num_associations = len(trace.associated_traces)
                association_factor = 1.0 + num_associations * self.config.decay_params['association_boost']

            # Access pattern analysis
            access_factor = 1.0
            if hasattr(trace, 'access_count'):
                # Recent access frequency
                access_frequency = trace.access_count / max(1, age / 3600.0)  # accesses per hour
                access_factor = 1.0 + np.log1p(access_frequency) * 0.2

            # Specialization domain relevance
            domain_factor = 1.0
            if hasattr(trace, 'context') and 'domain' in trace.context:
                # Could be enhanced with node specialization matching
                domain_factor = 1.0

            # System pressure adjustment
            pressure_adjustment = 1.0 - system_pressure * 0.3

            # Compute final priority
            priority = (age_factor * salience_factor * consolidation_factor *
                       association_factor * access_factor * domain_factor *
                       pressure_adjustment) / recency_penalty

            # Add small random factor to break ties
            priority *= np.random.uniform(0.98, 1.02)

            return max(0.0, priority)

        except Exception as e:
            logger.error(f"Error computing eviction priority for trace {getattr(trace, 'trace_id', 'unknown')}: {e}")
            return 0.5  # Default medium priority

    async def select_eviction_candidates(self,
                                       traces: List[Any],
                                       num_to_evict: int,
                                       system_pressure: float = 0.5) -> List[Any]:
        """
        Enhanced eviction candidate selection with multiple strategies.
        """
        start_time = time.time()

        try:
            if not traces or num_to_evict <= 0:
                return []

            current_time = time.time()

            # Filter candidates based on protection criteria
            candidates = []
            protected_count = 0

            for trace in traces:
                # Protection criteria
                if (hasattr(trace, 'eviction_protection') and trace.eviction_protection):
                    protected_count += 1
                    continue

                # Don't evict traces currently consolidating
                if (hasattr(trace, 'temporal_metadata') and
                    trace.temporal_metadata.consolidation_state == ConsolidationState.CONSOLIDATING):
                    protected_count += 1
                    continue

                # Don't evict very recent high-salience traces
                age = current_time - trace.timestamp
                if age < 60.0 and trace.salience > 0.8:
                    protected_count += 1
                    continue

                candidates.append(trace)

            if not candidates:
                logger.warning(f"No eviction candidates available. Protected: {protected_count}")
                return []

            # Calculate priorities for all candidates
            priorities = []
            for trace in candidates:
                priority = await self.compute_eviction_priority(trace, current_time, system_pressure)
                priorities.append((priority, trace))

            # Sort by priority (lowest = highest eviction priority)
            priorities.sort(key=lambda x: x[0])

            # Select candidates for eviction
            selected = [trace for _, trace in priorities[:num_to_evict]]

            # Update statistics
            if selected:
                avg_priority = np.mean([p for p, _ in priorities[:num_to_evict]])
                self.stats['eviction_stats']['average_eviction_priority'] = avg_priority

            # Record timing
            selection_time = time.time() - start_time
            self.operation_timings['eviction_selection'].append(selection_time)
            self.stats['performance_metrics']['eviction_selection_time'] = selection_time

            logger.debug(f"Selected {len(selected)} eviction candidates from {len(candidates)} in {selection_time:.3f}s")

            return selected

        except Exception as e:
            logger.error(f"Error selecting eviction candidates: {e}")
            return []

    async def manage_memory_aging(self, traces: List[Any]) -> Dict[str, int]:
        """
        Enhanced memory aging with comprehensive state transitions.
        """
        try:
            current_time = time.time()
            transitions = defaultdict(int)

            for trace in traces:
                old_state = await self.evaluate_trace_lifecycle(trace, current_time)

                # Update trace state based on lifecycle evaluation
                age = current_time - trace.timestamp
                recency = current_time - trace.last_access

                new_state = old_state

                # Age-based transitions
                if old_state == MemoryLifecycleState.ACTIVE:
                    if age > self.config.age_thresholds['aging_threshold']:
                        new_state = MemoryLifecycleState.AGING
                elif old_state == MemoryLifecycleState.AGING:
                    if age > self.config.age_thresholds['eviction_threshold']:
                        new_state = MemoryLifecycleState.EVICTION_CANDIDATE
                    elif recency < self.config.age_thresholds['aging_threshold'] / 2:
                        # Reactivate if recently accessed
                        new_state = MemoryLifecycleState.ACTIVE

                # Record transitions
                if old_state != new_state:
                    if hasattr(trace, 'state'):
                        trace.state = new_state

                    transition_key = f"{old_state.value}_to_{new_state.value}"
                    transitions[transition_key] += 1

                    # Record in transition history
                    self.transition_history.append({
                        'trace_id': getattr(trace, 'trace_id', 'unknown'),
                        'from_state': old_state.value,
                        'to_state': new_state.value,
                        'timestamp': current_time,
                        'age': age,
                        'recency': recency,
                    })

                    logger.debug(f"Trace {getattr(trace, 'trace_id', 'unknown')} transitioned: {old_state.value} -> {new_state.value}")

            return dict(transitions)

        except Exception as e:
            logger.error(f"Error managing memory aging: {e}")
            return {}

    async def consolidation_readiness_check(self, trace: Any) -> bool:
        """
        Enhanced consolidation readiness check with multiple criteria.
        """
        try:
            current_time = time.time()
            age = current_time - trace.timestamp

            # Basic age requirement
            if age < self.config.consolidation_params['readiness_age']:
                return False

            # Check current consolidation state
            if hasattr(trace, 'temporal_metadata'):
                consolidation_state = trace.temporal_metadata.consolidation_state
                if consolidation_state != ConsolidationState.INITIAL:
                    return False

            # Salience threshold
            if trace.salience < 0.5:
                return False

            # Access pattern analysis
            if hasattr(trace, 'access_count') and trace.access_count < 2:
                return False

            # Temporal age category check
            if hasattr(trace, 'get_temporal_age_category'):
                age_category = trace.get_temporal_age_category()
                if age_category not in ['protein_synthesis', 'homeostatic_scaling']:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking consolidation readiness: {e}")
            return False

    async def update_consolidation_states(self, traces: List[Any]) -> Dict[str, int]:
        """
        Enhanced consolidation state management with progress tracking.
        """
        start_time = time.time()

        try:
            current_time = time.time()
            state_counts = defaultdict(int)
            consolidation_updates = 0

            for trace in traces:
                if not hasattr(trace, 'temporal_metadata'):
                    continue

                consolidation_state = trace.temporal_metadata.consolidation_state

                if consolidation_state == ConsolidationState.INITIAL:
                    if await self.consolidation_readiness_check(trace):
                        # Initiate consolidation
                        if self.dynamics_engine:
                            await self.dynamics_engine.initiate_consolidation(trace, current_time)
                        else:
                            trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATING

                        self.stats['consolidation_stats']['attempts'] += 1
                        consolidation_updates += 1

                elif consolidation_state == ConsolidationState.CONSOLIDATING:
                    # Update consolidation progress
                    if self.dynamics_engine:
                        await self.dynamics_engine.update_consolidation_progress(trace, current_time, 1.0)
                    else:
                        # Simple progress simulation
                        progress = getattr(trace.temporal_metadata, 'consolidation_strength', 0.0)
                        progress += self.config.consolidation_params['progress_rate']
                        trace.temporal_metadata.consolidation_strength = min(1.0, progress)

                        if progress >= self.config.consolidation_params['success_threshold']:
                            trace.temporal_metadata.consolidation_state = ConsolidationState.CONSOLIDATED
                            self.stats['consolidation_stats']['successes'] += 1

                    consolidation_updates += 1

                elif consolidation_state == ConsolidationState.CONSOLIDATED:
                    # Check for transition to stable
                    age = current_time - trace.timestamp
                    if age > self.config.consolidation_params['stability_time']:
                        trace.temporal_metadata.consolidation_state = ConsolidationState.STABLE
                        consolidation_updates += 1

                state_counts[trace.temporal_metadata.consolidation_state.value] += 1

            # Update success rate
            if self.stats['consolidation_stats']['attempts'] > 0:
                self.stats['consolidation_stats']['success_rate'] = (
                    self.stats['consolidation_stats']['successes'] /
                    self.stats['consolidation_stats']['attempts']
                )

            # Record timing
            update_time = time.time() - start_time
            self.operation_timings['consolidation_update'].append(update_time)
            self.stats['performance_metrics']['consolidation_update_time'] = update_time

            if consolidation_updates > 0:
                logger.debug(f"Updated {consolidation_updates} consolidation states in {update_time:.3f}s")

            return dict(state_counts)

        except Exception as e:
            logger.error(f"Error updating consolidation states: {e}")
            return {}

    async def cleanup_orphaned_references(self, valid_trace_ids: Set[str]) -> int:
        """
        Enhanced cleanup with comprehensive reference tracking.
        """
        try:
            cleaned_count = 0

            # Clean transition history
            original_history_size = len(self.transition_history)
            self.transition_history = deque(
                [event for event in self.transition_history
                 if event['trace_id'] in valid_trace_ids],
                maxlen=10000
            )
            cleaned_count += original_history_size - len(self.transition_history)

            # Clean performance tracking for specific traces
            # (Implementation would depend on specific data structures)

            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} orphaned references")

            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning orphaned references: {e}")
            return 0

    async def perform_maintenance_cycle(self, traces: List[Any]) -> Dict[str, Any]:
        """
        Enhanced comprehensive maintenance cycle with detailed monitoring.
        """
        async with self._lock:
            start_time = time.time()
            current_time = time.time()

            try:
                logger.debug(f"Starting maintenance cycle for node {self.node_id}")

                maintenance_results = {
                    'cycle_start_time': current_time,
                    'traces_processed': len(traces),
                    'transitions': {},
                    'consolidation_updates': {},
                    'cleanup_count': 0,
                    'errors': [],
                }

                # 1. Memory aging management
                try:
                    transitions = await self.manage_memory_aging(traces)
                    maintenance_results['transitions'] = transitions
                except Exception as e:
                    error_msg = f"Error in aging management: {e}"
                    logger.error(error_msg)
                    maintenance_results['errors'].append(error_msg)

                # 2. Consolidation state updates
                try:
                    if current_time - self.last_consolidation_time >= self.config.consolidation_interval:
                        consolidation_updates = await self.update_consolidation_states(traces)
                        maintenance_results['consolidation_updates'] = consolidation_updates
                        self.last_consolidation_time = current_time
                        self.stats['last_consolidation_time'] = current_time
                except Exception as e:
                    error_msg = f"Error in consolidation updates: {e}"
                    logger.error(error_msg)
                    maintenance_results['errors'].append(error_msg)

                # 3. Reference cleanup
                try:
                    valid_trace_ids = {
                        getattr(trace, 'trace_id', f'trace_{i}')
                        for i, trace in enumerate(traces)
                        if not hasattr(trace, 'state') or trace.state != MemoryLifecycleState.ARCHIVED
                    }
                    cleanup_count = await self.cleanup_orphaned_references(valid_trace_ids)
                    maintenance_results['cleanup_count'] = cleanup_count
                except Exception as e:
                    error_msg = f"Error in cleanup: {e}"
                    logger.error(error_msg)
                    maintenance_results['errors'].append(error_msg)

                # 4. Update comprehensive statistics
                try:
                    await self._update_comprehensive_stats(traces)
                except Exception as e:
                    error_msg = f"Error updating stats: {e}"
                    logger.error(error_msg)
                    maintenance_results['errors'].append(error_msg)

                # 5. Performance monitoring
                cycle_time = time.time() - start_time
                self.operation_timings['maintenance_cycle'].append(cycle_time)
                self.stats['performance_metrics']['maintenance_cycle_time'] = cycle_time
                self.stats['last_maintenance_time'] = current_time
                self.last_maintenance_time = current_time

                maintenance_results['cycle_duration'] = cycle_time
                maintenance_results['cycle_end_time'] = time.time()

                # Add to performance history
                self.performance_history.append({
                    'timestamp': current_time,
                    'cycle_time': cycle_time,
                    'traces_count': len(traces),
                    'transitions_count': sum(transitions.values()) if transitions else 0,
                    'errors_count': len(maintenance_results['errors']),
                })

                logger.debug(f"Maintenance cycle completed in {cycle_time:.3f}s")

                return maintenance_results

            except Exception as e:
                logger.error(f"Critical error in maintenance cycle: {e}")
                return {'error': str(e), 'cycle_duration': time.time() - start_time}

    async def _update_comprehensive_stats(self, traces: List[Any]) -> None:
        """Update comprehensive statistics."""
        try:
            self.stats['total_traces'] = len(traces)

            # Reset state counts
            for state in MemoryLifecycleState:
                self.stats['state_counts'][state.value] = 0
            for state in ConsolidationState:
                self.stats['consolidation_counts'][state.value] = 0

            # Count current states
            for trace in traces:
                if hasattr(trace, 'state'):
                    self.stats['state_counts'][trace.state.value] += 1

                if hasattr(trace, 'temporal_metadata'):
                    consolidation_state = trace.temporal_metadata.consolidation_state.value
                    self.stats['consolidation_counts'][consolidation_state] += 1

            # Calculate eviction rate
            if len(self.performance_history) > 1:
                recent_performance = list(self.performance_history)[-10:]
                total_time = recent_performance[-1]['timestamp'] - recent_performance[0]['timestamp']
                if total_time > 0:
                    evicted_traces = self.stats['eviction_stats']['total_evicted']
                    self.stats['eviction_stats']['eviction_rate'] = evicted_traces / total_time

        except Exception as e:
            logger.error(f"Error updating comprehensive stats: {e}")

    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """
        Enhanced statistics with performance metrics and trends.
        """
        try:
            stats = self.stats.copy()

            # Add performance trends
            if self.performance_history:
                recent_performance = list(self.performance_history)[-10:]
                stats['performance_trends'] = {
                    'average_cycle_time': np.mean([p['cycle_time'] for p in recent_performance]),
                    'cycle_time_trend': self._calculate_trend([p['cycle_time'] for p in recent_performance]),
                    'traces_per_second': np.mean([p['traces_count'] / max(0.001, p['cycle_time']) for p in recent_performance]),
                }

            # Add operation timing statistics
            stats['operation_timings'] = {}
            for operation, timings in self.operation_timings.items():
                if timings:
                    recent_timings = timings[-100:]  # Last 100 operations
                    stats['operation_timings'][operation] = {
                        'average': np.mean(recent_timings),
                        'median': np.median(recent_timings),
                        'std': np.std(recent_timings),
                        'count': len(timings),
                    }

            # Add transition statistics
            if self.transition_history:
                recent_transitions = [t for t in self.transition_history
                                    if time.time() - t['timestamp'] < 3600]  # Last hour
                transition_counts = defaultdict(int)
                for transition in recent_transitions:
                    key = f"{transition['from_state']}_to_{transition['to_state']}"
                    transition_counts[key] += 1
                stats['recent_transitions'] = dict(transition_counts)

            return stats

        except Exception as e:
            logger.error(f"Error getting lifecycle statistics: {e}")
            return self.stats.copy()

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1) for a list of values."""
        if len(values) < 2:
            return 0.0

        try:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            return np.clip(z[0] / np.mean(values) if np.mean(values) != 0 else 0, -1, 1)
        except:
            return 0.0

    async def save_state(self, filepath: str) -> bool:
        """Save lifecycle manager state to file."""
        try:
            state_data = {
                'node_id': self.node_id,
                'config': {
                    'eviction_batch_size': self.config.eviction_batch_size,
                    'consolidation_interval': self.config.consolidation_interval,
                    'maintenance_interval': self.config.maintenance_interval,
                    'age_thresholds': self.config.age_thresholds,
                    'decay_params': self.config.decay_params,
                    'consolidation_params': self.config.consolidation_params,
                },
                'stats': self.stats,
                'last_consolidation_time': self.last_consolidation_time,
                'last_maintenance_time': self.last_maintenance_time,
                'performance_history': list(self.performance_history),
                'transition_history': list(self.transition_history),
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info(f"Lifecycle manager state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving lifecycle manager state: {e}")
            return False

    async def load_state(self, filepath: str) -> bool:
        """Load lifecycle manager state from file."""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)

            # Restore state
            self.stats = state_data.get('stats', self._initialize_stats())
            self.last_consolidation_time = state_data.get('last_consolidation_time', 0.0)
            self.last_maintenance_time = state_data.get('last_maintenance_time', 0.0)

            # Restore performance history
            performance_data = state_data.get('performance_history', [])
            self.performance_history = deque(performance_data, maxlen=1000)

            # Restore transition history
            transition_data = state_data.get('transition_history', [])
            self.transition_history = deque(transition_data, maxlen=10000)

            logger.info(f"Lifecycle manager state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading lifecycle manager state: {e}")
            return False
