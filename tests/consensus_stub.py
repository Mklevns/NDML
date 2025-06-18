"""Test-only consensus stub implementing ConsensusInterface."""
from typing import Dict, Any
import torch
from ndml.consensus.interface import ConsensusInterface


class ConsensusStub(ConsensusInterface):
    async def propose_memory_update(
        self, trace_id: str, content_vector: torch.Tensor, metadata: Dict[str, Any]
    ) -> bool:
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {"stub_consensus": True}
