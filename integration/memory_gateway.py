# integration/memory_gateway.py
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

# Import from local files
from core.dmn import EnhancedDistributedMemoryNode

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

        # Statistics and monitoring
        self.stats = {
            'total_queries': 0,
            'total_updates': 0,
            'cache_hits': 0,
            'routing_decisions': defaultdict(int),
            'cluster_utilization': defaultdict(float),
            'consensus_operations': 0
        }

        logger.info(f"MemoryGateway initialized: {num_clusters} clusters, {nodes_per_cluster} nodes/cluster")

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
            "routing": {"strategy": "round_robin"},
            "retrieval": {"default_k": 10},
            "btsp": {"calcium_threshold": 0.7, "decay_rate": 0.95, "novelty_weight": 0.4, "importance_weight": 0.3, "error_weight": 0.3, "learning_rate": 0.1}
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

    async def retrieve_memories_async(self,
                                      query: torch.Tensor,
                                      context: Dict[str, Any],
                                      k: int = None,
                                      diversity_weight: float = None) -> List[Tuple[Any, float]]:
        """Retrieve memories across all relevant clusters"""

        if k is None:
            k = self.config['retrieval']['default_k']

        if diversity_weight is None:
            diversity_weight = self.config['retrieval']['diversity_weight']

        self.stats['total_queries'] += 1

        # Route query to relevant clusters
        cluster_scores = await self.cluster_router.route_query_async(query, context)

        # Select top clusters based on routing scores
        max_clusters = self.config['routing']['max_clusters_per_query']
        selected_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:max_clusters]

        # Retrieve from selected clusters
        all_results = []

        for cluster_idx, cluster_score in selected_clusters:
            cluster = self.clusters[cluster_idx]

            cluster_results = await cluster.retrieve_memories_async(
                query=query,
                context=context,
                k=k,
                similarity_threshold=self.config['retrieval']['similarity_threshold']
            )

            # Weight results by cluster routing score
            weighted_results = [
                (memory, similarity * cluster_score)
                for memory, similarity in cluster_results
            ]

            all_results.extend(weighted_results)

            # Update routing statistics
            self.stats['routing_decisions'][cluster_idx] += 1

        # Diversity-aware re-ranking
        if diversity_weight > 0:
            all_results = self._diversify_results(all_results, diversity_weight)

        # Sort by final score and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

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

    async def add_memory_async(self,
                               content: torch.Tensor,
                               context: Dict[str, Any],
                               salience: float,
                               user_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """Add new memory to appropriate cluster/node"""

        self.stats['total_updates'] += 1

        # Route to appropriate cluster
        cluster_scores = await self.cluster_router.route_query_async(content, context)
        best_cluster_idx = max(cluster_scores.items(), key=lambda x: x[1])[0]
        best_cluster = self.clusters[best_cluster_idx]

        # Use consensus layer if enabled
        success = False

        if self.consensus_layer:
            # Coordinate update through consensus layer
            self.stats['consensus_operations'] += 1

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
            # Add to selected cluster
            node_success = await best_cluster.add_memory_async(
                content=content,
                context=context,
                salience=salience,
                user_feedback=user_feedback
            )

            if node_success:
                # Update load balancer
                await self.load_balancer.record_update(best_cluster_idx)

                # Update cluster utilization stats
                utilization = best_cluster.get_utilization()
                self.stats['cluster_utilization'][best_cluster_idx] = utilization

                return True

        return False

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
        """Route query to clusters and return scores"""

        with torch.no_grad():
            # Normalize query
            query_norm = F.normalize(query, dim=-1)

            # Get routing scores
            if self.config['strategy'] == 'learned':
                scores = self.router_network(query_norm)
                cluster_scores = {i: float(scores[i]) for i in range(self.num_clusters)}

            elif self.config['strategy'] == 'round_robin':
                # Simple round-robin routing
                cluster_scores = {i: 1.0 / self.num_clusters for i in range(self.num_clusters)}

            elif self.config['strategy'] == 'load_based':
                # Route based on cluster load (would need load information)
                cluster_scores = {i: 1.0 / self.num_clusters for i in range(self.num_clusters)}

            else:
                # Default to uniform distribution
                cluster_scores = {i: 1.0 / self.num_clusters for i in range(self.num_clusters)}

        # Record routing decision for learning
        self.routing_history.append({
            'query': query_norm.cpu(),
            'context': context,
            'scores': cluster_scores,
            'timestamp': time.time()
        })

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
            'performance_feedback': self.performance_feedback[-1000:]
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load router state"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.routing_history = checkpoint.get('routing_history', [])
        self.performance_feedback = checkpoint.get('performance_feedback', [])


class NodeSelector:
    """Selects appropriate nodes within a cluster for queries and updates"""

    def __init__(self, nodes: List[EnhancedDistributedMemoryNode], config: Dict[str, Any]):
        self.nodes = nodes
        self.config = config
        self.selection_history = defaultdict(int)

    async def select_nodes_for_query(self, query: torch.Tensor, context: Dict[str, Any]) -> List[
        EnhancedDistributedMemoryNode]:
        """Select nodes for memory retrieval"""

        # For retrieval, we typically want to query multiple nodes
        # to get diverse results

        # Simple strategy: select top 50% of nodes by some criteria
        num_to_select = max(1, len(self.nodes) // 2)

        # For now, just rotate through nodes to balance load
        # In a more sophisticated implementation, this would consider:
        # - Node specialization
        # - Current load
        # - Historical performance

        selected_nodes = []
        for i in range(num_to_select):
            node_idx = (sum(self.selection_history.values()) + i) % len(self.nodes)
            selected_nodes.append(self.nodes[node_idx])
            self.selection_history[node_idx] += 1

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