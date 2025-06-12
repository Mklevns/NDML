# core/hybrid_memory.py

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import asyncio
from collections import deque
import time
import math
import uuid

# Import a unified memory trace to avoid circular dependencies
from .memory_trace import UnifiedMemoryTrace


@dataclass
class ImportanceSignals:
    """Placeholder for importance signals."""
    calcium: float = 0.0
    novelty: float = 0.0
    error: float = 0.0

class MemoryReplayBuffer:
    """A simple replay buffer for memory consolidation."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, trace: 'UnifiedMemoryTrace', priority: float):
        self.buffer.append((priority, trace))

    async def replay_consolidation(self, relational_encoder: Optional[nn.Module]):
        # This is a placeholder for the actual replay logic
        pass

# ============================================
# INFORMATION-THEORETIC CONTRACTION ENGINE
# ============================================

class InformationContractor:
    """Manages information-theoretic memory contraction"""

    def __init__(self, target_ratio: float = 4.61):
        self.target_ratio = target_ratio
        self.total_information_destroyed = 0.0
        self.total_information_preserved = 0.0

    def measure_information_content(self, embedding: torch.Tensor) -> float:
        """Estimate information content of an embedding"""
        # Use SVD to estimate intrinsic dimensionality
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        _, s, _ = torch.svd(embedding)

        # Effective rank (information content)
        normalized_singular_values = s / s.sum()
        entropy = -torch.sum(
            normalized_singular_values * torch.log2(normalized_singular_values + 1e-10)
        )

        return entropy.item() * embedding.shape[1]

    def contract_memory(self, trace: 'UnifiedMemoryTrace') -> Tuple[torch.Tensor, float]:
        """Apply information-theoretic contraction"""
        original_info = self.measure_information_content(trace.embedding)

        # Determine contraction strength based on importance
        importance = trace.compute_total_importance()
        contraction_strength = 1.0 - importance # High importance = low contraction

        # Apply PCA-like compression
        embedding_2d = trace.embedding.view(-1, trace.embedding.shape[-1])
        U, S, V = torch.svd(embedding_2d)

        # Keep only top components based on contraction strength
        k = max(1, int((1 - contraction_strength) * len(S)))
        S_contracted = S.clone()
        S_contracted[k:] = 0

        # Reconstruct with information loss
        contracted_embedding = torch.mm(torch.mm(U, torch.diag(S_contracted)), V.t())
        contracted_embedding = contracted_embedding.view_as(trace.embedding)

        # Measure information after contraction
        new_info = self.measure_information_content(contracted_embedding)
        info_destroyed = original_info - new_info

        # Update metrics
        self.total_information_destroyed += info_destroyed
        self.total_information_preserved += new_info

        # Update trace
        trace.compression_level += 1
        trace.information_content = new_info
        trace.contraction_history.append(info_destroyed)

        return contracted_embedding, info_destroyed

    def get_destruction_ratio(self) -> float:
        """Calculate current information destruction ratio"""
        if self.total_information_preserved == 0:
            return 0.0
        return self.total_information_destroyed / self.total_information_preserved


# ============================================
# UNIFIED CONSOLIDATION ORCHESTRATOR
# ============================================

class UnifiedConsolidationOrchestrator:
    """Orchestrates both information-theoretic and biological consolidation"""

    def __init__(self,
                 contraction_ratio: float = 4.61,
                 relational_encoder: Optional[nn.Module] = None):
        self.contractor = InformationContractor(contraction_ratio)
        self.relational_encoder = relational_encoder

        # Consolidation queues by priority
        self.queues = {
            'immediate': asyncio.PriorityQueue(), # High calcium spikes
            'standard': asyncio.PriorityQueue(), # Normal consolidation
            'background': asyncio.PriorityQueue() # Low-priority compression
        }

        # Schema formation tracking
        self.emerging_schemas = {}
        self.schema_threshold = 5 # Memories needed to form schema

    async def evaluate_memory(self, trace: 'UnifiedMemoryTrace') -> str:
        """Determine consolidation pathway for memory"""
        importance = trace.compute_total_importance()
        info_content = trace.information_content

        # High importance + high information = immediate relational encoding
        if importance > 0.8 and info_content > 100:
            return 'immediate'

        # Medium importance or information = standard processing
        elif importance > 0.4 or info_content > 50:
            return 'standard'

        # Everything else = background compression
        else:
            return 'background'

    async def consolidate_immediate(self, traces: List['UnifiedMemoryTrace']):
        """Fast-track consolidation for critical memories"""
        for trace in traces:
            # Skip compression, go straight to relational encoding
            if self.relational_encoder and trace.encoding_type == "content":
                # Find related memories for triplet encoding
                related = self._find_related_memories(trace)

                if len(related) >= 2:
                    # Encode with triplet loss
                    anchor = trace.embedding
                    positive = related[0].embedding
                    negative = related[-1].embedding

                    relational_embedding = self.relational_encoder(
                        torch.cat([anchor, positive, negative]),
                        mode='relational'
                    )

                    trace.embedding = relational_embedding
                    trace.encoding_type = "relational"

            # Check for schema formation
            await self._check_schema_formation(trace)

    async def consolidate_standard(self, traces: List['UnifiedMemoryTrace']):
        """Standard consolidation with balanced compression and encoding"""
        for trace in traces:
            # First apply information contraction
            if trace.compression_level < 2: # Don't over-compress
                contracted_embedding, info_lost = self.contractor.contract_memory(trace)

                # Only update if compression was beneficial
                if info_lost < trace.information_content * 0.5:
                    trace.embedding = contracted_embedding

            # Then consider relational encoding if important enough
            if trace.compute_total_importance() > 0.6 and trace.encoding_type == "content":
                # Upgrade to relational encoding
                await self._upgrade_to_relational(trace)

    async def consolidate_background(self, traces: List['UnifiedMemoryTrace']):
        """Aggressive compression for low-importance memories"""
        for trace in traces:
            # Apply strong contraction
            if trace.compression_level < 3:
                # Use higher contraction for background memories
                original_embedding = trace.embedding.clone()
                contracted_embedding, _ = self.contractor.contract_memory(trace)

                # Extra compression step
                if trace.compression_level < 3:
                    contracted_embedding, _ = self.contractor.contract_memory(trace)

                trace.embedding = contracted_embedding

    async def _check_schema_formation(self, trace: 'UnifiedMemoryTrace'):
        """Check if memory should form or join a schema"""
        # Find memories with high semantic overlap
        similar_memories = self._find_similar_memories(trace, threshold=0.8)

        if len(similar_memories) >= self.schema_threshold:
            # Create or update schema
            schema_id = f"schema_{len(self.emerging_schemas)}"

            if trace.schema_membership is None:
                trace.schema_membership = schema_id

                # Create schematic representation
                schema_embedding = torch.mean(
                    torch.stack([m.embedding for m in similar_memories]),
                    dim=0
                )

                self.emerging_schemas[schema_id] = {
                    'embedding': schema_embedding,
                    'members': {m.id for m in similar_memories},
                    'created': time.time()
                }

                # Update all members
                for mem in similar_memories:
                    mem.schema_membership = schema_id
                    mem.encoding_type = "schematic"

    def _find_related_memories(self, trace: 'UnifiedMemoryTrace') -> List['UnifiedMemoryTrace']:
        """Find memories related through associations or similarity"""
        # This would interface with your memory bank
        # Placeholder implementation
        return []

    def _find_similar_memories(self, trace: 'UnifiedMemoryTrace', threshold: float = 0.7) -> List['UnifiedMemoryTrace']:
        """Find semantically similar memories"""
        # This would interface with your FAISS index
        # Placeholder implementation
        return []

    async def _upgrade_to_relational(self, trace: 'UnifiedMemoryTrace'):
        # Placeholder for upgrading a memory to a relational encoding
        pass

# ============================================
# INTEGRATED HYBRID-NDML MEMORY SYSTEM
# ============================================

class IntegratedHybridNDMLSystem:
    """Complete system combining NDML and biological approaches"""

    def __init__(self,
                 embedding_dim: int = 768,
                 memory_capacity: int = 10000,
                 contraction_ratio: float = 4.61):

        # Memory storage
        self.memory_bank: Dict[str, 'UnifiedMemoryTrace'] = {}
        self.memory_capacity = memory_capacity

        # Information-theoretic components
        self.contractor = InformationContractor(contraction_ratio)
        self.information_budget = memory_capacity * 100.0 # Bits budget

        # Biological components
        self.consolidation_orchestrator = UnifiedConsolidationOrchestrator(
            contraction_ratio=contraction_ratio
        )
        self.replay_buffer = MemoryReplayBuffer(capacity=memory_capacity // 2)

        # Metrics tracking
        self.metrics = {
            'total_information': 0.0,
            'destruction_ratio': 0.0,
            'consolidation_events': 0,
            'schema_formations': 0,
            'memory_distribution': {
                'content': 0,
                'relational': 0,
                'schematic': 0
            }
        }

    async def add_memory(self,
                         content: str,
                         embedding: torch.Tensor,
                         importance_signals: Optional[Dict] = None) -> str:
        """Add new memory with unified processing"""

        # Create unified trace
        trace = UnifiedMemoryTrace(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            information_content=self.contractor.measure_information_content(embedding)
        )

        # Set importance signals if provided
        if importance_signals:
            trace.btsp_calcium = importance_signals.get('calcium', 0.0)
            trace.importance_signals = importance_signals

        # Check if we need to make room (information budget)
        current_info = sum(t.information_content for t in self.memory_bank.values())

        if current_info + trace.information_content > self.information_budget:
            await self._enforce_information_budget()

        # Add to memory bank
        self.memory_bank[trace.id] = trace

        # Queue for consolidation
        pathway = await self.consolidation_orchestrator.evaluate_memory(trace)
        priority = -trace.compute_total_importance() # Negative for priority queue

        await self.consolidation_orchestrator.queues[pathway].put((priority, trace))

        # Add to replay buffer
        self.replay_buffer.add_experience(trace, trace.compute_total_importance())

        # Update metrics
        self._update_metrics()

        return trace.id

    async def _enforce_information_budget(self):
        """Maintain information budget through selective forgetting"""
        # Sort memories by importance
        memories_by_importance = sorted(
            self.memory_bank.values(),
            key=lambda t: t.compute_total_importance()
        )

        # Contract or remove low-importance memories
        for trace in memories_by_importance[:len(memories_by_importance)//4]:
            if trace.compression_level >= 3:
                # Fully compressed, remove it
                del self.memory_bank[trace.id]
                self.metrics['total_information'] -= trace.information_content
            else:
                # Apply contraction
                old_info = trace.information_content
                new_embedding, _ = self.contractor.contract_memory(trace)
                trace.embedding = new_embedding

                # Update information tracking
                info_saved = old_info - trace.information_content
                self.metrics['total_information'] -= info_saved

        # Check if we've achieved target destruction ratio
        ratio = self.contractor.get_destruction_ratio()
        if ratio < self.contractor.target_ratio:
            # Need more aggressive contraction
            await self._aggressive_contraction_cycle()

    async def _aggressive_contraction_cycle(self):
        """Perform aggressive contraction to meet target ratio"""
        # Target memories with medium importance
        targets = [
            t for t in self.memory_bank.values()
            if 0.3 < t.compute_total_importance() < 0.7
        ]

        for trace in targets[:len(targets)//3]:
            if trace.compression_level < 2:
                self.contractor.contract_memory(trace)

    async def retrieve_memories(self,
                                query_embedding: torch.Tensor,
                                k: int = 10,
                                include_schemas: bool = True) -> List['UnifiedMemoryTrace']:
        """Retrieve memories using hybrid similarity metrics"""
        results = []

        # Calculate similarities considering both information content and embedding similarity
        for trace in self.memory_bank.values():
            # Skip schemas if not requested
            if not include_schemas and trace.encoding_type == "schematic":
                continue

            # Compute similarity
            embedding_sim = torch.cosine_similarity(
                query_embedding.flatten(),
                trace.embedding.flatten(),
                dim=0
            )

            # Weight by information content and encoding type
            if trace.encoding_type == "relational":
                similarity = embedding_sim * 1.5 # Boost relational memories
            elif trace.encoding_type == "schematic":
                similarity = embedding_sim * 2.0 # Strong boost for schemas
            else:
                similarity = embedding_sim

            # Information-weighted similarity
            info_weight = math.log1p(trace.information_content) / 10.0
            final_similarity = similarity * (1 + info_weight)

            results.append((final_similarity, trace))

        # Sort and return top k
        results.sort(key=lambda x: x[0], reverse=True)

        # Update access patterns
        for _, trace in results[:k]:
            trace.access_count += 1
            trace.last_access_time = time.time()

        return [trace for _, trace in results[:k]]

    async def run_maintenance_cycle(self):
        """Run all maintenance processes"""
        # Process consolidation queues
        for queue_name, queue in self.consolidation_orchestrator.queues.items():
            if not queue.empty():
                batch = []
                for _ in range(min(32, queue.qsize())): # Process in batches
                    try:
                        _, trace = await asyncio.wait_for(queue.get(), timeout=0.1)
                        batch.append(trace)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    if queue_name == 'immediate':
                        await self.consolidation_orchestrator.consolidate_immediate(batch)
                    elif queue_name == 'standard':
                        await self.consolidation_orchestrator.consolidate_standard(batch)
                    else:
                        await self.consolidation_orchestrator.consolidate_background(batch)

        # Perform replay consolidation
        await self.replay_buffer.replay_consolidation(
            self.consolidation_orchestrator.relational_encoder
        )

        # Update metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update system metrics"""
        self.metrics['total_information'] = sum(
            t.information_content for t in self.memory_bank.values()
        )
        self.metrics['destruction_ratio'] = self.contractor.get_destruction_ratio()

        # Count encoding types
        for encoding_type in ['content', 'relational', 'schematic']:
            self.metrics['memory_distribution'][encoding_type] = sum(
                1 for t in self.memory_bank.values()
                if t.encoding_type == encoding_type
            )

    def get_system_health(self) -> Dict:
        """Get comprehensive system health metrics"""
        return {
            'total_memories': len(self.memory_bank),
            'total_information_bits': self.metrics['total_information'],
            'information_efficiency': self.metrics['total_information'] / max(1, len(self.memory_bank)),
            'destruction_ratio': self.metrics['destruction_ratio'],
            'target_ratio': self.contractor.target_ratio,
            'memory_distribution': self.metrics['memory_distribution'],
            'schema_count': len(self.consolidation_orchestrator.emerging_schemas),
            'compression_distribution': {
                f'level_{i}': sum(1 for t in self.memory_bank.values() if t.compression_level == i)
                for i in range(4)
            }
        }
