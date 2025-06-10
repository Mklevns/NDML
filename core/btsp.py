# core/btsp.py - Fixed version with proper device handling
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BTSPUpdateDecision:
    """Decision output from BTSP mechanism."""
    should_update: bool
    calcium_level: float
    learning_rate: float
    novelty_score: float
    importance_score: float
    error_score: float
    confidence: float

class BTSPUpdateMechanism:
    """
    Biological Tag-and-Store Plasticity mechanism for intelligent memory updates.
    
    FIXED: Proper device handling for all neural networks and tensor operations.
    """
    
    def __init__(self,
                 calcium_threshold: float = 0.7,
                 decay_rate: float = 0.95,
                 novelty_weight: float = 0.4,
                 importance_weight: float = 0.3,
                 error_weight: float = 0.3,
                 learning_rate: float = 0.1,
                 dimension: int = 512,
                 device: str = None):  # ADD device parameter
        """
        Initialize BTSP mechanism with proper device handling.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.calcium_threshold = calcium_threshold
        self.decay_rate = decay_rate
        self.novelty_weight = novelty_weight
        self.importance_weight = importance_weight
        self.error_weight = error_weight
        self.base_learning_rate = learning_rate
        self.dimension = dimension
        
        # FIXED: Proper device handling
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"BTSP using device: {self.device}")
        
        # Initialize neural components with proper device placement
        self._init_neural_components()
        
        # State tracking
        self.calcium_levels = {}
        self.update_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_evaluations': 0,
            'updates_triggered': 0,
            'average_calcium': 0.0,
            'average_novelty': 0.0,
            'average_importance': 0.0,
            'average_error': 0.0,
        }
        
        logger.info(f"BTSP Update Mechanism initialized on {self.device}")

    def _init_neural_components(self):
        """Initialize neural network components with proper device placement."""
        
        # Novelty detection network
        self.novelty_network = nn.Sequential(
            nn.Linear(self.dimension, self.dimension // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dimension // 2, self.dimension // 4),
            nn.ReLU(),
            nn.Linear(self.dimension // 4, 1),
            nn.Sigmoid()
        ).to(self.device)  # FIXED: Move to device
        
        # Importance estimation network
        self.importance_network = nn.Sequential(
            nn.Linear(self.dimension + 64, self.dimension // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dimension // 2, self.dimension // 4),
            nn.ReLU(),
            nn.Linear(self.dimension // 4, 1),
            nn.Sigmoid()
        ).to(self.device)  # FIXED: Move to device
        
        # Error prediction network
        self.error_network = nn.Sequential(
            nn.Linear(self.dimension * 2, self.dimension),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dimension, self.dimension // 2),
            nn.ReLU(),
            nn.Linear(self.dimension // 2, 1),
            nn.Sigmoid()
        ).to(self.device)  # FIXED: Move to device
        
        # Calcium dynamics simulator
        self.calcium_dynamics = CalciumDynamicsSimulator(
            decay_rate=self.decay_rate,
            threshold=self.calcium_threshold
        )
        
        logger.info(f"BTSP neural networks initialized on {self.device}")

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Utility method to ensure tensor is on the correct device."""
        if tensor.device.type != self.device:
            return tensor.to(self.device)
        return tensor

    async def should_update_async(self,
                                  input_state: torch.Tensor,
                                  existing_traces: List[Any],
                                  context: Dict[str, Any],
                                  user_feedback: Optional[Dict[str, Any]] = None) -> BTSPUpdateDecision:
        """
        Determine if memory should be updated based on BTSP mechanism.
        FIXED: Proper device handling throughout.
        """
        try:
            current_time = time.time()
            
            # FIXED: Ensure input_state is on correct device
            input_state = self._ensure_device(input_state)
            
            # Compute novelty score
            novelty_score = await self._compute_novelty(input_state, existing_traces)
            
            # Compute importance score
            importance_score = await self._compute_importance(input_state, context, user_feedback)
            
            # Compute error score
            error_score = await self._compute_error(input_state, existing_traces, context)
            
            # Update calcium dynamics
            calcium_level = await self._update_calcium_dynamics(
                novelty_score, importance_score, error_score, current_time
            )
            
            # Make update decision
            should_update = calcium_level >= self.calcium_threshold
            
            # Compute adaptive learning rate
            learning_rate = self._compute_adaptive_learning_rate(
                calcium_level, novelty_score, importance_score, error_score
            )
            
            # Compute confidence in decision
            confidence = self._compute_decision_confidence(
                calcium_level, novelty_score, importance_score, error_score
            )
            
            # Create decision object
            decision = BTSPUpdateDecision(
                should_update=should_update,
                calcium_level=calcium_level,
                learning_rate=learning_rate,
                novelty_score=novelty_score,
                importance_score=importance_score,
                error_score=error_score,
                confidence=confidence
            )
            
            # Update statistics
            await self._update_performance_metrics(decision)
            
            # Record in history
            self.update_history.append({
                'timestamp': current_time,
                'decision': decision,
                'context': context.copy() if context else {}
            })
            
            logger.debug(f"BTSP decision: update={should_update}, "
                        f"calcium={calcium_level:.3f}, "
                        f"novelty={novelty_score:.3f}, "
                        f"importance={importance_score:.3f}, "
                        f"error={error_score:.3f}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in BTSP update decision: {e}")
            # Return conservative decision on error
            return BTSPUpdateDecision(
                should_update=False,
                calcium_level=0.0,
                learning_rate=self.base_learning_rate,
                novelty_score=0.0,
                importance_score=0.0,
                error_score=0.0,
                confidence=0.0
            )

    async def _compute_novelty(self, 
                               input_state: torch.Tensor, 
                               existing_traces: List[Any]) -> float:
        """Compute novelty score with proper device handling."""
        try:
            if not existing_traces:
                return 1.0  # Completely novel if no existing traces
            
            # FIXED: Ensure input is on correct device and normalized
            input_state = self._ensure_device(input_state)
            input_normalized = F.normalize(input_state, dim=-1)
            
            # Compare with existing traces
            similarities = []
            for trace in existing_traces[-50:]:  # Check last 50 traces for efficiency
                if hasattr(trace, 'content'):
                    # FIXED: Ensure trace content is on same device
                    trace_content = self._ensure_device(trace.content)
                    trace_content = F.normalize(trace_content, dim=-1)
                    
                    similarity = F.cosine_similarity(
                        input_normalized.unsqueeze(0), 
                        trace_content.unsqueeze(0)
                    ).item()
                    similarities.append(similarity)
            
            if not similarities:
                return 1.0
            
            # Novelty is inverse of maximum similarity
            max_similarity = max(similarities)
            novelty = 1.0 - max_similarity
            
            # Apply novelty network for refinement
            with torch.no_grad():
                # FIXED: Ensure input is on correct device for network
                input_for_network = input_normalized.unsqueeze(0)
                network_novelty = self.novelty_network(input_for_network).item()
                
                # Combine rule-based and network-based novelty
                combined_novelty = 0.7 * novelty + 0.3 * network_novelty
            
            return float(np.clip(combined_novelty, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing novelty: {e}")
            return 0.5

    async def _compute_importance(self, 
                              input_state: torch.Tensor,
                              context: Dict[str, Any],
                              user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Compute importance with proper device handling."""
        try:
            # FIXED: Ensure input is on correct device
            input_state = self._ensure_device(input_state)
            
            # Extract context features on the same device
            context_features = self._extract_context_features(
                context, user_feedback, device=self.device
            )
            
            # Normalize input
            input_normalized = F.normalize(input_state, dim=-1)
            
            # FIXED: Ensure both tensors are on same device before concatenation
            context_features = self._ensure_device(context_features)
            combined_input = torch.cat([
                input_normalized.unsqueeze(0),
                context_features.unsqueeze(0)
            ], dim=1)
            
            # Apply importance network
            with torch.no_grad():
                importance = self.importance_network(combined_input).item()
            
            # Apply user feedback multiplier
            if user_feedback:
                feedback_multiplier = self._get_feedback_multiplier(user_feedback)
                importance *= feedback_multiplier
            
            return float(np.clip(importance, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing importance: {e}")
            return 0.5

    async def _compute_error(self, 
                         input_state: torch.Tensor,
                         existing_traces: List[Any],
                         context: Dict[str, Any]) -> float:
        """Compute error with proper device handling."""
        try:
            if not existing_traces:
                return 0.5
            
            # FIXED: Ensure input is on correct device
            input_state = self._ensure_device(input_state)
            input_normalized = F.normalize(input_state, dim=-1)
            
            # Find most similar existing trace as reference
            best_match = None
            best_similarity = -1.0
            
            for trace in existing_traces[-20:]:
                if hasattr(trace, 'content'):
                    # FIXED: Ensure trace content is on same device as input
                    trace_content = self._ensure_device(trace.content)
                    trace_content = F.normalize(trace_content, dim=-1)
                    
                    similarity = F.cosine_similarity(
                        input_normalized.unsqueeze(0),
                        trace_content.unsqueeze(0)
                    ).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = trace
            
            if best_match is None:
                return 0.5
            
            # Compute error using network
            reference_content = self._ensure_device(best_match.content)
            reference_content = F.normalize(reference_content, dim=-1)
            
            # FIXED: Ensure both tensors are on same device before concatenation
            combined_input = torch.cat([
                input_normalized.unsqueeze(0),
                reference_content.unsqueeze(0)
            ], dim=1)
            
            with torch.no_grad():
                error = self.error_network(combined_input).item()
            
            # Modulate based on context similarity
            context_similarity = self._compute_context_similarity(
                context, getattr(best_match, 'context', {})
            )
            error_adjustment = 1.0 - context_similarity * 0.3
            error *= error_adjustment
            
            return float(np.clip(error, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing error: {e}")
            return 0.5

    def _extract_context_features(self, 
                              context: Dict[str, Any],
                              user_feedback: Optional[Dict[str, Any]] = None,
                              device: str = 'cpu') -> torch.Tensor:
        """Extract context features with explicit device placement."""
        
        # FIXED: Create tensor on specified device
        features = torch.zeros(64, device=device)
        
        try:
            # Task type encoding
            task_types = ['learning', 'correction', 'exploration', 'recall']
            task_type = context.get('task_type', 'exploration')
            if task_type in task_types:
                features[task_types.index(task_type)] = 1.0
            
            # Domain encoding
            domains = ['general', 'code', 'factual', 'conversational']
            domain = context.get('domain', 'general')
            if domain in domains:
                features[4 + domains.index(domain)] = 1.0
            
            # Temporal features
            if 'timestamp' in context:
                hour = (context['timestamp'] % 86400) / 86400
                features[8] = hour
            
            # User feedback features
            if user_feedback:
                if user_feedback.get('positive', False):
                    features[10] = 1.0
                if user_feedback.get('negative', False):
                    features[11] = 1.0
                if 'confidence' in user_feedback:
                    features[12] = float(user_feedback['confidence'])
            
            # Salience and importance hints
            if 'importance' in context:
                features[20] = float(context['importance'])
            if 'urgency' in context:
                features[21] = float(context['urgency'])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return features

    def _get_feedback_multiplier(self, user_feedback: Dict[str, Any]) -> float:
        """Get feedback multiplier for importance adjustment."""
        multiplier = 1.0
        
        if user_feedback.get('positive', False):
            multiplier *= 1.5
        elif user_feedback.get('negative', False):
            multiplier *= 1.2  # Negative feedback is also important
        
        if 'importance' in user_feedback:
            multiplier *= (0.5 + float(user_feedback['importance']))
        
        return np.clip(multiplier, 0.1, 3.0)

    def _compute_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute similarity between two contexts."""
        try:
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0
            
            similarities = []
            for key in common_keys:
                if key in ['domain', 'task_type']:
                    similarities.append(1.0 if context1[key] == context2[key] else 0.0)
                elif key in ['importance', 'urgency', 'confidence']:
                    val1, val2 = float(context1[key]), float(context2[key])
                    similarities.append(1.0 - abs(val1 - val2))
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error computing context similarity: {e}")
            return 0.0

    async def _update_calcium_dynamics(self, 
                                       novelty: float,
                                       importance: float,
                                       error: float,
                                       current_time: float) -> float:
        """Update calcium dynamics based on input signals."""
        try:
            # Weighted combination of signals
            signal_strength = (
                self.novelty_weight * novelty +
                self.importance_weight * importance +
                self.error_weight * error
            )
            
            # Update calcium level
            calcium_level = await self.calcium_dynamics.update(signal_strength, current_time)
            
            return calcium_level
            
        except Exception as e:
            logger.error(f"Error updating calcium dynamics: {e}")
            return 0.0

    def _compute_adaptive_learning_rate(self, 
                                        calcium_level: float,
                                        novelty: float,
                                        importance: float,
                                        error: float) -> float:
        """Compute adaptive learning rate based on signals."""
        try:
            # Base learning rate modulated by calcium
            calcium_modulation = calcium_level / self.calcium_threshold
            
            # Novelty and error boost learning rate
            novelty_boost = novelty * 0.5
            error_boost = error * 0.3
            
            # Importance provides stable scaling
            importance_scale = 0.5 + importance * 0.5
            
            adaptive_lr = (
                self.base_learning_rate * 
                calcium_modulation * 
                importance_scale *
                (1.0 + novelty_boost + error_boost)
            )
            
            return float(np.clip(adaptive_lr, 0.001, 0.5))
            
        except Exception as e:
            logger.error(f"Error computing adaptive learning rate: {e}")
            return self.base_learning_rate

    def _compute_decision_confidence(self, 
                                     calcium_level: float,
                                     novelty: float,
                                     importance: float,
                                     error: float) -> float:
        """Compute confidence in the update decision."""
        try:
            # Confidence based on signal consistency
            signal_variance = np.var([novelty, importance, error])
            consistency_score = 1.0 / (1.0 + signal_variance)
            
            # Calcium level provides base confidence
            calcium_confidence = calcium_level if calcium_level >= self.calcium_threshold else (1.0 - calcium_level)
            
            # Overall confidence
            confidence = 0.6 * consistency_score + 0.4 * calcium_confidence
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing decision confidence: {e}")
            return 0.5

    async def _update_performance_metrics(self, decision: BTSPUpdateDecision):
        """Update performance metrics based on decision."""
        try:
            self.performance_metrics['total_evaluations'] += 1
            
            if decision.should_update:
                self.performance_metrics['updates_triggered'] += 1
            
            # Update rolling averages
            n = self.performance_metrics['total_evaluations']
            alpha = 1.0 / n if n <= 100 else 0.01  # Exponential moving average
            
            self.performance_metrics['average_calcium'] = (
                (1 - alpha) * self.performance_metrics['average_calcium'] +
                alpha * decision.calcium_level
            )
            
            self.performance_metrics['average_novelty'] = (
                (1 - alpha) * self.performance_metrics['average_novelty'] +
                alpha * decision.novelty_score
            )
            
            self.performance_metrics['average_importance'] = (
                (1 - alpha) * self.performance_metrics['average_importance'] +
                alpha * decision.importance_score
            )
            
            self.performance_metrics['average_error'] = (
                (1 - alpha) * self.performance_metrics['average_error'] +
                alpha * decision.error_score
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive BTSP statistics."""
        stats = self.performance_metrics.copy()
        
        # Add derived metrics
        if stats['total_evaluations'] > 0:
            stats['update_rate'] = stats['updates_triggered'] / stats['total_evaluations']
        else:
            stats['update_rate'] = 0.0
        
        # Add device info
        stats['device'] = self.device
        
        # Add recent history statistics
        if self.update_history:
            recent_decisions = list(self.update_history)[-100:]  # Last 100 decisions
            
            recent_calcium = [d['decision'].calcium_level for d in recent_decisions]
            recent_updates = [d['decision'].should_update for d in recent_decisions]
            
            stats['recent_stats'] = {
                'average_calcium': np.mean(recent_calcium),
                'calcium_std': np.std(recent_calcium),
                'recent_update_rate': np.mean(recent_updates),
                'decisions_count': len(recent_decisions)
            }
        
        return stats

    def save_checkpoint(self, filepath: str):
        """Save BTSP mechanism state."""
        try:
            checkpoint = {
                'config': {
                    'calcium_threshold': self.calcium_threshold,
                    'decay_rate': self.decay_rate,
                    'novelty_weight': self.novelty_weight,
                    'importance_weight': self.importance_weight,
                    'error_weight': self.error_weight,
                    'base_learning_rate': self.base_learning_rate,
                    'device': self.device,  # FIXED: Save device info
                },
                'performance_metrics': self.performance_metrics,
                'calcium_levels': self.calcium_levels,
                'update_history': list(self.update_history)[-1000:],  # Keep last 1000
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"BTSP checkpoint saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving BTSP checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """Load BTSP mechanism state."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)  # FIXED: Load to correct device
            
            # Restore performance metrics
            self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
            self.calcium_levels = checkpoint.get('calcium_levels', {})
            
            # Restore update history
            history_data = checkpoint.get('update_history', [])
            self.update_history = deque(history_data, maxlen=1000)
            
            logger.info(f"BTSP checkpoint loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading BTSP checkpoint: {e}")


class CalciumDynamicsSimulator:
    """Simulates calcium dynamics for BTSP mechanism."""
    
    def __init__(self, decay_rate: float = 0.95, threshold: float = 0.7):
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.calcium_level = 0.0
        self.last_update_time = 0.0
    
    async def update(self, signal_strength: float, current_time: float) -> float:
        """Update calcium level based on input signal."""
        try:
            # Time-based decay
            if self.last_update_time > 0:
                dt = current_time - self.last_update_time
                decay_factor = np.exp(-dt / 10.0)  # 10 second time constant
                self.calcium_level *= decay_factor
            
            # Signal-driven increase
            self.calcium_level += signal_strength * 0.5
            
            # Saturation
            self.calcium_level = np.clip(self.calcium_level, 0.0, 2.0)
            
            self.last_update_time = current_time
            
            return self.calcium_level
            
        except Exception as e:
            logger.error(f"Error updating calcium dynamics: {e}")
            return self.calcium_level