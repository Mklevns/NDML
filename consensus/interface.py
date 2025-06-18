"""Consensus interface for NDML."""
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any


class ConsensusInterface(ABC):
    """Abstract base class for consensus implementations."""

    @abstractmethod
    async def propose_memory_update(
        self, trace_id: str, content_vector: torch.Tensor, metadata: Dict[str, Any]
    ) -> bool:
        """Propose a memory update to peers and return True on success."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return implementation specific statistics."""
