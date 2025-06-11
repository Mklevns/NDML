# utils/__init__.py - Utility functions package
"""
NDML Utility Functions

Common utilities for data processing, monitoring, and system management.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def normalize_embeddings(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize embeddings to unit length."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=dim)

def compute_similarity_matrix(embeddings1: torch.Tensor,
                            embeddings2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of embeddings."""
    norm1 = normalize_embeddings(embeddings1)
    norm2 = normalize_embeddings(embeddings2)
    return torch.mm(norm1, norm2.t())

def create_attention_mask(lengths: List[int], max_length: Optional[int] = None) -> torch.Tensor:
    """Create attention mask from sequence lengths."""
    if max_length is None:
        max_length = max(lengths)

    mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return mask

def batch_embeddings(embeddings: List[torch.Tensor],
                    padding_value: float = 0.0) -> torch.Tensor:
    """Batch variable-length embeddings with padding."""
    max_length = max(emb.shape[0] for emb in embeddings)
    embed_dim = embeddings[0].shape[-1]

    batched = torch.full((len(embeddings), max_length, embed_dim),
                        padding_value, dtype=embeddings[0].dtype)

    for i, emb in enumerate(embeddings):
        batched[i, :emb.shape[0]] = emb

    return batched

def exponential_moving_average(new_value: float,
                             current_avg: float,
                             alpha: float = 0.1) -> float:
    """Compute exponential moving average."""
    return alpha * new_value + (1 - alpha) * current_avg

def create_decay_schedule(initial_value: float,
                         decay_rate: float,
                         num_steps: int) -> np.ndarray:
    """Create exponential decay schedule."""
    steps = np.arange(num_steps)
    return initial_value * np.exp(-decay_rate * steps)

class PerformanceMonitor:
    """Simple performance monitoring utility."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.timestamps = {}

    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.timestamps[metric_name] = []

        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(time.time())

        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]
            self.timestamps[metric_name] = self.timestamps[metric_name][-self.window_size:]

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = self.metrics[metric_name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values),
            'latest': values[-1],
        }

def save_embeddings(embeddings: torch.Tensor,
                   filepath: str,
                   metadata: Optional[Dict[str, Any]] = None):
    """Save embeddings to file with optional metadata."""
    data = {'embeddings': embeddings.cpu()}
    if metadata:
        data['metadata'] = metadata

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, filepath)
    logger.info(f"Saved embeddings to {filepath}")

def load_embeddings(filepath: str) -> Dict[str, Any]:
    """Load embeddings from file."""
    data = torch.load(filepath, map_location='cpu')
    logger.info(f"Loaded embeddings from {filepath}")
    return data

__all__ = [
    'normalize_embeddings',
    'compute_similarity_matrix',
    'create_attention_mask',
    'batch_embeddings',
    'exponential_moving_average',
    'create_decay_schedule',
    'PerformanceMonitor',
    'save_embeddings',
    'load_embeddings',
]
