# integration/memory_gateway.py
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import time
from collections import defaultdict

from ..core import EnhancedDistributedMemoryNode # Corrected relative import

logger = logging.getLogger(__name__)


class MemoryGateway:
    """Gateway for LLM-memory system interactions"""

    def __init__(self,
                 dimension: int,
                 num_clusters: int = 4,
                 nodes_per_cluster: int = 8,
                 node_capacity: int = 5000,
                 enable_consensus: bool = True,
                 config: Optional[Dict[str, Any]] = None):

        self.dimension = dimension
        self.num_clusters = num_clusters
        self.nodes_per_cluster = nodes_per_cluster
        self.node_capacity = node_capacity
        self.enable_consensus = enable_consensus
        self.config = config or self._default_config()

        # Initialize memory clusters
        self.clusters = self._init_clusters()

        # Initialize consensus layer if enabled (stub for now)
        self.consensus_layer = None
        if enable_consensus:
            # Create a simple consensus stub
            self.consensus_layer = self._create_consensus_stub()

        # Routing and load balancing
        self.cluster_router = self._init_cluster_router()
        self.load_balancer = LoadBalancer(self.clusters)

        # FIXED: Proper statistics initialization with explicit types
        self.stats = {
            'total_queries': 0,           # Explicitly initialize as int
            'total_updates': 0,           # Explicitly initialize as int
            'cache_hits': 0,              # Explicitly initialize as int
            'routing_decisions': defaultdict(int),  # defaultdict of ints
            'cluster_utilization': defaultdict(float),  # defaultdict of floats
            'consensus_operations': 0     # Explicitly initialize as int
        }

        logger.info(f"MemoryGateway initialized: {num_clusters} clusters, {nodes_per_cluster} nodes/cluster")
        logger.info(f"Stats initialized: {self.stats}")  # Debug log to verify stats

    def _create_consensus_stub(self):
        """Create a stub consensus layer for testing."""
        class ConsensusStub:
            def __init__(self):
                pass

            async def propose_memory_update(self, trace_id, content_vector, metadata):
                # Always return success for testing
                return True

            def get_stats(self):
                return {"stub_consensus": True}

        return ConsensusStub()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "routing": {
                "strategy": "round_robin",
                "max_clusters_per_query": 2,
                "cluster_selection_threshold": 0.3,
            },
            "retrieval": {
                "default_k": 10,
                "similarity_threshold": 0.3,
                "diversity_weight": 0.2,
                "context_weight": 0.3,
            },
            "btsp": {
                "calcium_threshold": 0.7,
                "decay_rate": 0.95,
                "novelty_weight": 0.4,
                "importance_weight": 0.3,
                "error_weight": 0.3,
                "learning_rate": 0.1
            }
        }

    def _init_clusters(self) -> List['MemoryCluster']:
        """Initialize memory clusters with specialized nodes"""
        clusters = []

        specializations = ['general', 'code', 'factual', 'conversational']

        for i in range(self.num_clusters):
            cluster = MemoryCluster(
                cluster_id=f"cluster_{i}",
                specialization=specializations[i % len(specializations)],
                dimension=self.dimension,
                num_nodes=self.nodes_per_cluster,
                node_capacity=self.node_capacity,
                config=self.config
            )
            clusters.append(cluster)

        return clusters

    def _init_cluster_router(self) -> 'ClusterRouter':
        """Initialize routing system for directing queries to clusters"""
        return ClusterRouter(
            input_dimension=self.dimension,
            num_clusters=self.num_clusters,
            config=self.config['routing']
        )

    def _safe_increment_stat(self, stat_name: str, increment: int = 1):
        """Safely increment a statistic, ensuring it's always an integer."""
        try:
            if stat_name not in self.stats:
                self.stats[stat_name] = 0

            # Ensure the current value is an integer
            if not isinstance(self.stats[stat_name], int):
                logger.warning(f"Stat '{stat_name}' was not an int (was {type(self.stats[stat_name])}), resetting to 0")
                self.stats[stat_name] = 0

            self.stats[stat_name] += increment

        except Exception as e:
            logger.error(f"Error incrementing stat '{stat_name}': {e}")
            self.stats[stat_name] = increment  # Reset to the increment value

    def _ensure_stats_integrity(self):
        """Ensure all stats are the correct type."""
        expected_int_stats = ['total_queries', 'total_updates', 'cache_hits', 'consensus_operations']

        for stat_name in expected_int_stats:
            if stat_name not in self.stats:
                self.stats[stat_name] = 0
            elif not isinstance(self.stats[stat_name], int):
                logger.warning(f"Fixing corrupted stat '{stat_name}': was {type(self.stats[stat_name])}, setting to 0")
                self.stats[stat_name] = 0

        # Ensure defaultdicts are properly initialized
        if not isinstance(self.stats.get('routing_decisions'), defaultdict):
            self.stats['routing_decisions'] = defaultdict(int)

        if not isinstance(self.stats.get('cluster_utilization'), defaultdict):
            self.stats['cluster_utilization'] = defaultdict(float)

    async def retrieve_memories_async(self,
                                  query: torch.Tensor,
                                  context: Dict[str, Any],
                                  k: int = None,
                                  diversity_weight: float = None) -> List[Tuple[Any, float]]:
        """Retrieve memories from distributed clusters with proper stats tracking."""

        if k is None:
            k = self.config['retrieval']['default_k']

        if diversity_weight is None:
            diversity_weight = self.config['retrieval'].get('diversity_weight', 0.2)

        # FIXED: Use safe increment for stats
        self._safe_increment_stat('total_queries')

        # Ensure stats integrity
        self._ensure_stats_integrity()

        # DEVICE FIX: Ensure consistent device handling
        original_device = query.device

        # Route query to relevant clusters
        cluster_scores = await self.cluster_router.route_query_async(query, context)

        max_clusters = self.config['routing']['max_clusters_per_query']
        selected_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:max_clusters]

        all_results = []

        for cluster_idx, cluster_score in selected_clusters:
            cluster = self.clusters[cluster_idx]

            cluster_results = await cluster.retrieve_memories_async(
                query=query,  # Pass original query, let cluster handle device management
                context=context,
                k=k,
                similarity_threshold=self.config['retrieval']['similarity_threshold']
            )

            weighted_results = [
                (memory, similarity * cluster_score)
                for memory, similarity in cluster_results
            ]

            all_results.extend(weighted_results)

            # FIXED: Use safe increment for routing decisions
            if 'routing_decisions' not in self.stats:
                self.stats['routing_decisions'] = defaultdict(int)
            self.stats['routing_decisions'][cluster_idx] += 1

        # Apply diversity-aware re-ranking
        if diversity_weight > 0:
            all_results = self._diversify_results(all_results, diversity_weight)

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

    async def add_memory_async(self,
                           content: torch.Tensor,
                           context: Dict[str, Any],
                           salience: float,
                           user_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """Add memory to the appropriate cluster with proper routing."""

        # FIXED: Use safe increment instead of direct increment
        self._safe_increment_stat('total_updates')

        # Ensure stats integrity before proceeding
        self._ensure_stats_integrity()

        # DEVICE FIX: Keep original device reference
        original_device = content.device

        # Route to appropriate cluster
        cluster_scores = await self.cluster_router.route_query_async(content, context)

        # FIXED: Better cluster selection logic
        if self.cluster_router.config['strategy'] == 'round_robin':
            # For round-robin, select the cluster with highest score
            best_cluster_idx = max(cluster_scores.items(), key=lambda x: x[1])[0]
        else:
            # For other strategies, use weighted random selection for better distribution
            import random

            # Convert scores to probabilities
            total_score = sum(cluster_scores.values())
            if total_score > 0:
                probabilities = {k: v/total_score for k, v in cluster_scores.items()}

                # Weighted random selection
                rand_val = random.random()
                cumulative = 0.0
                best_cluster_idx = 0

                for cluster_idx, prob in probabilities.items():
                    cumulative += prob
                    if rand_val <= cumulative:
                        best_cluster_idx = cluster_idx
                        break
            else:
                # Fallback to random selection
                best_cluster_idx = random.randint(0, len(self.clusters) - 1)

        best_cluster = self.clusters[best_cluster_idx]

        logger.debug(f"🎯 Selected cluster {best_cluster_idx} for memory storage")

        success = False

        if self.consensus_layer:
            self._safe_increment_stat('consensus_operations')
            success = await self.consensus_layer.propose_memory_update(
                trace_id=f"trace_{int(time.time() * 1000)}_{best_cluster_idx}",
                content_vector=content,
                metadata={
                    'salience': salience,
                    'cluster_id': best_cluster_idx,
                    'context': context,
                    'user_feedback': user_feedback
                }
            )

        if success or not self.consensus_layer:
            node_success = await best_cluster.add_memory_async(
                content=content,
                context=context,
                salience=salience,
                user_feedback=user_feedback
            )

            if node_success:
                await self.load_balancer.record_update(best_cluster_idx)
                utilization = best_cluster.get_utilization()
                self.stats['cluster_utilization'][best_cluster_idx] = utilization
                logger.debug(f"✅ Memory successfully stored in cluster {best_cluster_idx}")
                return True
            else:
                logger.warning(f"❌ Failed to store memory in cluster {best_cluster_idx}")

        return False

    def _diversify_results(self, results: List[Tuple[Any, float]], diversity_weight: float) -> List[Tuple[Any, float]]:
        """Apply diversity-aware re-ranking to avoid redundant memories"""

        if len(results) <= 1:
            return results

        # Extract embeddings for similarity computation
        embeddings = []
        for memory, score in results:
            embeddings.append(memory.content)

        embeddings = torch.stack(embeddings)

        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())

        # Diversity re-ranking
        diversified_results = []
        remaining_indices = list(range(len(results)))

        # Always include the top result
        best_idx = 0
        diversified_results.append(results[best_idx])
        remaining_indices.remove(best_idx)

        # Select subsequent results based on relevance and diversity
        while remaining_indices and len(diversified_results) < len(results):
            best_score = -float('inf')
            best_remaining_idx = None

            for idx in remaining_indices:
                # Original relevance score
                relevance_score = results[idx][1]

                # Compute diversity penalty (max similarity to selected items)
                diversity_penalty = 0.0
                for selected_memory, _ in diversified_results:
                    sim = torch.cosine_similarity(
                        results[idx][0].content.unsqueeze(0),
                        selected_memory.content.unsqueeze(0)
                    ).item()
                    diversity_penalty = max(diversity_penalty, sim)

                # Combined score
                combined_score = relevance_score - diversity_weight * diversity_penalty

                if combined_score > best_score:
                    best_score = combined_score
                    best_remaining_idx = idx

            if best_remaining_idx is not None:
                diversified_results.append((results[best_remaining_idx][0], best_score))
                remaining_indices.remove(best_remaining_idx)

        return diversified_results

    def has_memories(self) -> bool:
        """Check if any cluster has memories"""
        return any(cluster.has_memories() for cluster in self.clusters)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""

        # Collect stats from all clusters
        cluster_stats = []
        for i, cluster in enumerate(self.clusters):
            stats = cluster.get_comprehensive_stats()
            stats['cluster_id'] = i
            cluster_stats.append(stats)

        # Collect routing stats
        routing_stats = self.cluster_router.get_stats()

        # Collect load balancer stats
        load_balancer_stats = self.load_balancer.get_stats()

        # Collect consensus stats if available
        consensus_stats = {}
        if self.consensus_layer:
            consensus_stats = self.consensus_layer.get_stats()

        return {
            'gateway_stats': self.stats,
            'cluster_stats': cluster_stats,
            'routing_stats': routing_stats,
            'load_balancer_stats': load_balancer_stats,
            'consensus_stats': consensus_stats,
            'system_summary': {
                'total_clusters': len(self.clusters),
                'total_nodes': sum(cluster.num_nodes for cluster in self.clusters),
                'total_memories': sum(cluster.get_memory_count() for cluster in self.clusters),
                'average_utilization': np.mean(list(self.stats['cluster_utilization'].values())) if self.stats[
                    'cluster_utilization'] else 0.0
            }
        }

    def save_checkpoint(self, filepath: str):
        """Save complete gateway state"""
        checkpoint_data = {
            'config': self.config,
            'stats': self.stats,
            'cluster_checkpoints': []
        }

        # Save each cluster
        for i, cluster in enumerate(self.clusters):
            cluster_checkpoint_path = f"{filepath}_cluster_{i}.pt"
            cluster.save_checkpoint(cluster_checkpoint_path)
            checkpoint_data['cluster_checkpoints'].append(cluster_checkpoint_path)

        # Save router state
        router_checkpoint_path = f"{filepath}_router.pt"
        self.cluster_router.save_checkpoint(router_checkpoint_path)
        checkpoint_data['router_checkpoint'] = router_checkpoint_path

        # Save consensus layer if present
        if self.consensus_layer:
            consensus_checkpoint_path = f"{filepath}_consensus.pt"
            # Save consensus state (would need implementation)
            checkpoint_data['consensus_checkpoint'] = consensus_checkpoint_path

        torch.save(checkpoint_data, filepath)
        logger.info(f"MemoryGateway: Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load complete gateway state"""
        checkpoint_data = torch.load(filepath)

        # Restore stats
        self.stats = checkpoint_data['stats']

        # Load each cluster
        for i, cluster_checkpoint_path in enumerate(checkpoint_data['cluster_checkpoints']):
            if i < len(self.clusters):
                self.clusters[i].load_checkpoint(cluster_checkpoint_path)

        # Load router state
        if 'router_checkpoint' in checkpoint_data:
            self.cluster_router.load_checkpoint(checkpoint_data['router_checkpoint'])

        # Load consensus layer if present
        if self.consensus_layer and 'consensus_checkpoint' in checkpoint_data:
            # Load consensus state (would need implementation)
            pass

        logger.info(f"MemoryGateway: Loaded checkpoint from {filepath}")

    async def periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            # Trigger consolidation in all clusters
            consolidation_tasks = []
            for cluster in self.clusters:
                consolidation_tasks.append(cluster.consolidate_memories_async())

            await asyncio.gather(*consolidation_tasks)

            # Rebalance load if needed
            await self.load_balancer.rebalance_if_needed()

            # Update routing model if using learned routing
            if self.config['routing']['strategy'] == 'learned':
                await self.cluster_router.update_routing_model()

            logger.info("MemoryGateway: Completed periodic maintenance")

        except Exception as e:
            logger.error(f"MemoryGateway: Maintenance error: {e}")

    def debug_cluster_distribution(self) -> Dict[str, Any]:
        """Debug function to check memory distribution across clusters."""

        distribution = {}
        total_memories = 0

        for i, cluster in enumerate(self.clusters):
            cluster_memory_count = cluster.get_memory_count()
            total_memories += cluster_memory_count

            distribution[f"cluster_{i}"] = {
                "memory_count": cluster_memory_count,
                "utilization": cluster.get_utilization(),
                "specialization": cluster.specialization
            }

            # Check individual nodes in cluster
            node_distribution = {}
            for j, node in enumerate(cluster.nodes):
                node_memory_count = len(node.memory_traces)
                node_index_count = node.index.ntotal if hasattr(node, 'index') else 0
                node_distribution[f"node_{j}"] = {
                    "memories": node_memory_count,
                    "indexed": node_index_count,
                    "node_id": node.node_id
                }

            distribution[f"cluster_{i}"]["nodes"] = node_distribution

        distribution["total_memories"] = total_memories
        distribution["routing_stats"] = self.cluster_router.get_stats()

        return distribution


# Supporting classes for the gateway

class MemoryCluster:
    """Manages a cluster of memory nodes with specialization"""

    def __init__(self, cluster_id: str, specialization: str, dimension: int,
                 num_nodes: int, node_capacity: int, config: Dict[str, Any]):

        self.cluster_id = cluster_id
        self.specialization = specialization
        self.dimension = dimension
        self.num_nodes = num_nodes
        self.config = config

        # Initialize nodes
        self.nodes = []
        for i in range(num_nodes):
            node = EnhancedDistributedMemoryNode(
                node_id=f"{cluster_id}_node_{i}",
                dimension=dimension,
                capacity=node_capacity,
                specialization=specialization,
                config=config
            )
            self.nodes.append(node)

        # Node selection strategy
        self.node_selector = NodeSelector(self.nodes, config)

    async def retrieve_memories_async(self, query: torch.Tensor, context: Dict[str, Any],
                                      k: int, similarity_threshold: float) -> List[Tuple[Any, float]]:
        """Retrieve memories from cluster nodes"""

        # Select relevant nodes
        relevant_nodes = await self.node_selector.select_nodes_for_query(query, context)

        # Retrieve from selected nodes
        all_results = []

        for node in relevant_nodes:
            node_results = await node.retrieve_memories_async(
                query=query,
                k=k,
                context_filter=self._create_context_filter(context),
                similarity_threshold=similarity_threshold
            )
            all_results.extend(node_results)

        # Sort and return top results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

    async def add_memory_async(self, content: torch.Tensor, context: Dict[str, Any],
                               salience: float, user_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """Add memory to appropriate node in cluster"""

        # Select best node for this memory
        target_node = await self.node_selector.select_node_for_update(content, context)

        # Add to selected node
        return await target_node.add_memory_trace_async(
            content=content,
            context=context,
            salience=salience,
            user_feedback=user_feedback
        )

    def _create_context_filter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create context filter for node queries"""
        # Filter based on cluster specialization
        context_filter = {}

        if self.specialization != 'general':
            context_filter['domain'] = self.specialization

        # Add other context-based filters
        if 'user_id' in context:
            context_filter['user_id'] = context['user_id']

        if 'task_type' in context:
            context_filter['task_type'] = context['task_type']

        return context_filter

    def has_memories(self) -> bool:
        """Check if cluster has any memories"""
        return any(len(node.memory_traces) > 0 for node in self.nodes)

    def get_memory_count(self) -> int:
        """Get total memory count across all nodes"""
        return sum(len(node.memory_traces) for node in self.nodes)

    def get_utilization(self) -> float:
        """Get cluster utilization (0.0 to 1.0)"""
        total_capacity = sum(node.capacity for node in self.nodes)
        total_used = sum(len(node.memory_traces) for node in self.nodes)
        return total_used / total_capacity if total_capacity > 0 else 0.0

    async def consolidate_memories_async(self):
        """Trigger consolidation across all nodes"""
        consolidation_tasks = []
        for node in self.nodes:
            consolidation_tasks.append(node.consolidate_memories_async())

        await asyncio.gather(*consolidation_tasks)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        node_stats = [node.get_comprehensive_stats() for node in self.nodes]

        return {
            'cluster_id': self.cluster_id,
            'specialization': self.specialization,
            'num_nodes': self.num_nodes,
            'total_memories': self.get_memory_count(),
            'utilization': self.get_utilization(),
            'node_stats': node_stats
        }

    def save_checkpoint(self, filepath: str):
        """Save cluster state"""
        checkpoint_data = {
            'cluster_id': self.cluster_id,
            'specialization': self.specialization,
            'node_checkpoints': []
        }

        for i, node in enumerate(self.nodes):
            node_checkpoint_path = f"{filepath}_node_{i}.pt"
            node.save_checkpoint(node_checkpoint_path)
            checkpoint_data['node_checkpoints'].append(node_checkpoint_path)

        torch.save(checkpoint_data, filepath)

    def load_checkpoint(self, filepath: str):
        """Load cluster state"""
        checkpoint_data = torch.load(filepath)

        for i, node_checkpoint_path in enumerate(checkpoint_data['node_checkpoints']):
            if i < len(self.nodes):
                self.nodes[i].load_checkpoint(node_checkpoint_path)


class ClusterRouter(nn.Module):
    """Learned routing system for directing queries to clusters"""

    def __init__(self, input_dimension: int, num_clusters: int, config: Dict[str, Any]):
        super().__init__()

        self.input_dimension = input_dimension
        self.num_clusters = num_clusters
        self.config = config

        # FIXED: Add round-robin counter for proper load balancing
        self.round_robin_counter = 0

        # Routing network
        self.router_network = nn.Sequential(
            nn.Linear(input_dimension, input_dimension * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dimension * 2, input_dimension),
            nn.ReLU(),
            nn.Linear(input_dimension, num_clusters),
            nn.Softmax(dim=-1)
        )

        # Training state
        self.routing_history = []
        self.performance_feedback = []

    async def route_query_async(self, query: torch.Tensor, context: Dict[str, Any]) -> Dict[int, float]:
        """Route query to clusters with proper round-robin and load balancing."""

        with torch.no_grad():
            # Normalize query - ensure it's on CPU for consistent processing
            query_cpu = query.detach().cpu() if query.device.type != 'cpu' else query
            query_norm = F.normalize(query_cpu, dim=-1)

            # Get routing scores based on strategy
            if self.config['strategy'] == 'learned':
                scores = self.router_network(query_norm)
                cluster_scores = {i: float(scores[i]) for i in range(self.num_clusters)}

            elif self.config['strategy'] == 'round_robin':
                # FIXED: Proper round-robin implementation
                cluster_scores = {}

                # All clusters get base score
                base_score = 1.0 / self.num_clusters
                for i in range(self.num_clusters):
                    cluster_scores[i] = base_score

                # Boost the current round-robin target
                target_cluster = self.round_robin_counter % self.num_clusters
                cluster_scores[target_cluster] = 1.0  # Give highest score to target

                # Increment counter for next time
                self.round_robin_counter += 1

                logger.debug(f"🔄 Round-robin: targeting cluster {target_cluster}, counter={self.round_robin_counter}")

            elif self.config['strategy'] == 'load_based':
                # FIXED: Implement basic load-based routing
                cluster_scores = {}

                # For now, use round-robin as fallback for load-based
                # In a full implementation, you'd check actual cluster loads
                base_score = 1.0 / self.num_clusters
                for i in range(self.num_clusters):
                    cluster_scores[i] = base_score

                # Simple load balancing: alternate clusters
                target_cluster = self.round_robin_counter % self.num_clusters
                cluster_scores[target_cluster] = 1.0
                self.round_robin_counter += 1

            else:
                # Default: random distribution with slight bias
                import random
                cluster_scores = {}
                target_cluster = random.randint(0, self.num_clusters - 1)

                for i in range(self.num_clusters):
                    if i == target_cluster:
                        cluster_scores[i] = 0.7  # Bias toward target
                    else:
                        cluster_scores[i] = 0.3 / (self.num_clusters - 1)

        # Record routing decision for learning
        self.routing_history.append({
            'query': query_norm.cpu(),
            'context': context,
            'scores': cluster_scores,
            'timestamp': time.time(),
            'strategy': self.config['strategy']
        })

        # DEBUG: Log routing decisions
        logger.debug(f"🔍 DEBUG: Routing scores: {cluster_scores}")

        return cluster_scores

    async def update_routing_model(self):
        """Update routing model based on performance feedback"""
        # This would implement learning from routing performance
        # For now, it's a placeholder for future implementation
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'total_routings': len(self.routing_history),
            'strategy': self.config['strategy'],
            'routing_distribution': self._compute_routing_distribution()
        }

    def _compute_routing_distribution(self) -> Dict[int, float]:
        """Compute distribution of routing decisions"""
        if not self.routing_history:
            return {}

        distribution = defaultdict(float)
        for routing in self.routing_history[-1000:]:  # Last 1000 routings
            for cluster_id, score in routing['scores'].items():
                distribution[cluster_id] += score

        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        return dict(distribution)

    def save_checkpoint(self, filepath: str):
        """Save router state"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'routing_history': self.routing_history[-1000:],  # Keep last 1000
            'performance_feedback': self.performance_feedback[-1000:],
            'round_robin_counter': self.round_robin_counter
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load router state"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.routing_history = checkpoint.get('routing_history', [])
        self.performance_feedback = checkpoint.get('performance_feedback', [])
        self.round_robin_counter = checkpoint.get('round_robin_counter', 0)


class NodeSelector:
    """Selects appropriate nodes within a cluster for queries and updates"""

    def __init__(self, nodes: List[EnhancedDistributedMemoryNode], config: Dict[str, Any]):
        self.nodes = nodes
        self.config = config
        self.selection_history = defaultdict(int)

    async def select_nodes_for_query(self, query: torch.Tensor, context: Dict[str, Any]) -> List[EnhancedDistributedMemoryNode]:
        """Select nodes for query retrieval."""

        # For retrieval, query ALL nodes to ensure we don't miss memories
        # This is more thorough than the original 50% selection

        # For basic testing, select all nodes to ensure comprehensive search
        selected_nodes = self.nodes.copy()

        # Update selection history for all nodes
        for node in selected_nodes:
            self.selection_history[id(node)] += 1

        logger.debug(f"🔍 DEBUG: Selected {len(selected_nodes)} nodes for query")

        return selected_nodes

    async def select_node_for_update(self, content: torch.Tensor,
                                     context: Dict[str, Any]) -> EnhancedDistributedMemoryNode:
        """Select single node for memory update"""

        # For updates, we want to select the best single node
        # Selection criteria:
        # 1. Available capacity
        # 2. Specialization match
        # 3. Load balancing

        best_node = None
        best_score = -1

        for node in self.nodes:
            score = 0

            # Capacity factor (prefer nodes with more available space)
            utilization = len(node.memory_traces) / node.capacity
            capacity_score = 1.0 - utilization
            score += capacity_score * 0.4

            # Specialization factor
            context_domain = context.get('domain', 'general')
            if node.specialization == context_domain or node.specialization == 'general':
                score += 0.3

            # Load balancing factor (prefer less recently used nodes)
            recent_selections = self.selection_history[id(node)]
            load_score = 1.0 / (1.0 + recent_selections)
            score += load_score * 0.3

            if score > best_score:
                best_score = score
                best_node = node

        if best_node:
            self.selection_history[id(best_node)] += 1

        return best_node or self.nodes[0]  # Fallback to first node


class LoadBalancer:
    """Manages load balancing across clusters and nodes"""

    def __init__(self, clusters: List[MemoryCluster]):
        self.clusters = clusters
        self.update_counts = defaultdict(int)
        self.query_counts = defaultdict(int)
        self.last_rebalance = time.time()

    async def record_update(self, cluster_idx: int):
        """Record an update to a cluster"""
        self.update_counts[cluster_idx] += 1

    async def record_query(self, cluster_idx: int):
        """Record a query to a cluster"""
        self.query_counts[cluster_idx] += 1

    async def rebalance_if_needed(self):
        """Rebalance load if thresholds are exceeded"""
        current_time = time.time()

        # Check if rebalance interval has elapsed
        if current_time - self.last_rebalance < 300:  # 5 minutes
            return

        # Compute load imbalance
        update_counts = list(self.update_counts.values())
        if not update_counts:
            return

        max_updates = max(update_counts)
        min_updates = min(update_counts)

        # If imbalance is significant, trigger rebalancing
        if max_updates > 0 and (max_updates - min_updates) / max_updates > 0.3:
            await self._perform_rebalancing()
            self.last_rebalance = current_time

    async def _perform_rebalancing(self):
        """Perform actual load rebalancing"""
        # This would implement memory migration between clusters
        # For now, it's a placeholder
        logger.info("LoadBalancer: Performing load rebalancing")

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            'update_counts': dict(self.update_counts),
            'query_counts': dict(self.query_counts),
            'last_rebalance': self.last_rebalance
        }