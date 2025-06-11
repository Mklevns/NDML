# deployment/consensus_node.py
import asyncio
import aiohttp
import json
import logging
import os
import signal
import time
from typing import Dict, List, Optional, Any, Set
from aiohttp import web, ClientSession
import yaml

from ..consensus.neuromorphic import NeuromorphicConsensusLayer
from ..core.memory_trace import MemoryTrace

logger = logging.getLogger(__name__)


class ConsensusNode:
    """Distributed consensus node for NDML deployment"""

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['consensus']

        self.node_id = self.config['node_id']
        self.cluster_name = self.config['cluster_name']

        # Initialize consensus layer
        self.consensus_layer = NeuromorphicConsensusLayer(
            node_id=self.node_id,
            dimension=512,  # Could be configurable
            **self.config.get('vo2', {}),
            **self.config.get('bloom', {})
        )

        # Network components
        self.app = web.Application()
        self.client_session: Optional[ClientSession] = None
        self.peer_nodes: Set[str] = set()

        # Setup routes
        self._setup_routes()

        # Runtime state
        self.running = False
        self.heartbeat_task = None
        self.sync_task = None

        logger.info(f"ConsensusNode {self.node_id} initialized")

    def _setup_routes(self):
        """Setup HTTP routes for consensus operations"""

        # Health check
        self.app.router.add_get('/health', self.health_check)

        # Peer management
        self.app.router.add_post('/peers/register', self.register_peer)
        self.app.router.add_get('/peers', self.list_peers)

        # Memory operations
        self.app.router.add_post('/memory/propose', self.propose_memory_update)
        self.app.router.add_get('/memory/stats', self.get_memory_stats)

        # CRDT synchronization
        self.app.router.add_post('/sync/crdt', self.sync_crdt_data)
        self.app.router.add_get('/sync/crdt', self.get_crdt_data)

        # Conflict resolution
        self.app.router.add_post('/conflicts/resolve', self.resolve_conflicts)

        # Administrative
        self.app.router.add_get('/admin/config', self.get_config)
        self.app.router.add_post('/admin/config', self.update_config)

    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'node_id': self.node_id,
            'cluster': self.cluster_name,
            'uptime': time.time() - self.start_time,
            'peers': len(self.peer_nodes),
            'stats': self.consensus_layer.get_stats()
        })

    async def register_peer(self, request):
        """Register a new peer node"""
        data = await request.json()
        peer_id = data.get('node_id')
        peer_address = data.get('address')

        if not peer_id or not peer_address:
            return web.json_response({'error': 'Missing node_id or address'}, status=400)

        # Register with consensus layer
        await self.consensus_layer.register_peer(peer_id)

        # Store peer information
        self.peer_nodes.add(f"{peer_id}@{peer_address}")

        logger.info(f"Registered peer: {peer_id}@{peer_address}")

        return web.json_response({
            'status': 'registered',
            'peer_id': peer_id,
            'peer_count': len(self.peer_nodes)
        })

    async def list_peers(self, request):
        """List all registered peers"""
        return web.json_response({
            'peers': list(self.peer_nodes),
            'count': len(self.peer_nodes)
        })

    async def propose_memory_update(self, request):
        """Propose a memory update through consensus"""
        data = await request.json()

        try:
            # Extract update data
            trace_id = data.get('trace_id')
            content_vector = torch.tensor(data.get('content'), dtype=torch.float32)
            metadata = data.get('metadata', {})

            # Propose update through consensus layer
            success = await self.consensus_layer.propose_memory_update(
                trace_id=trace_id,
                content_vector=content_vector,
                metadata=metadata
            )

            return web.json_response({
                'success': success,
                'trace_id': trace_id,
                'node_id': self.node_id
            })

        except Exception as e:
            logger.error(f"Error proposing memory update: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_memory_stats(self, request):
        """Get memory statistics"""
        stats = self.consensus_layer.get_stats()
        return web.json_response(stats)

    async def sync_crdt_data(self, request):
        """Synchronize CRDT data with peer"""
        data = await request.json()
        peer_id = data.get('peer_id')
        peer_data = data.get('crdt_data', {})

        try:
            # Sync with consensus layer
            await self.consensus_layer.sync_with_peer(peer_id, peer_data)

            return web.json_response({
                'status': 'synced',
                'peer_id': peer_id,
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"Error syncing CRDT data: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_crdt_data(self, request):
        """Get current CRDT data for synchronization"""

        # Prepare CRDT data for sharing
        crdt_data = {
            'access_counters': {},
            'salience_registers': {}
        }

        # Export access counters
        for trace_id, counter in self.consensus_layer.access_counters.items():
            crdt_data['access_counters'][trace_id] = counter.to_dict()

        # Export salience registers
        for trace_id, register in self.consensus_layer.salience_registers.items():
            crdt_data['salience_registers'][trace_id] = register.to_dict()

        return web.json_response({
            'node_id': self.node_id,
            'timestamp': time.time(),
            'crdt_data': crdt_data
        })

    async def resolve_conflicts(self, request):
        """Handle conflict resolution requests"""
        data = await request.json()

        try:
            # Extract conflict data
            conflicts = data.get('conflicts', [])

            # Process through update protocol
            resolution = await self.consensus_layer.update_protocol.resolve_conflicts(conflicts)

            return web.json_response({
                'resolution': {
                    'conflict_id': resolution.conflict_id,
                    'winning_update': resolution.winning_update,
                    'losing_updates': resolution.losing_updates,
                    'method': resolution.resolution_method,
                    'confidence': resolution.confidence
                } if resolution else None
            })

        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_config(self, request):
        """Get current configuration"""
        return web.json_response(self.config)

    async def update_config(self, request):
        """Update configuration (limited subset)"""
        data = await request.json()

        # Only allow certain config updates
        allowed_updates = ['vo2', 'crdt', 'bloom', 'performance']

        updated = {}
        for key in allowed_updates:
            if key in data:
                self.config[key].update(data[key])
                updated[key] = self.config[key]

        return web.json_response({
            'status': 'updated',
            'updated_config': updated
        })

    async def start(self):
        """Start the consensus node"""
        self.running = True
        self.start_time = time.time()

        # Initialize HTTP client session
        self.client_session = ClientSession()

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.sync_task = asyncio.create_task(self._sync_loop())

        # Discover and register with existing peers
        await self._discover_peers()

        # Start HTTP server
        port = self.config['network']['port']
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()

        logger.info(f"ConsensusNode {self.node_id} started on port {port}")

    async def stop(self):
        """Stop the consensus node"""
        self.running = False

        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.sync_task:
            self.sync_task.cancel()

        # Close HTTP client session
        if self.client_session:
            await self.client_session.close()

        # Cleanup consensus layer
        self.consensus_layer.cleanup()

        logger.info(f"ConsensusNode {self.node_id} stopped")

    async def _discover_peers(self):
        """Discover existing peers in the cluster"""
        try:
            # Use Kubernetes service discovery or similar
            cluster_service = f"ndml-consensus-service.{os.getenv('NAMESPACE', 'default')}.svc.cluster.local"

            # This is a simplified example - real implementation would use
            # proper service discovery mechanisms

            logger.info(f"Discovering peers via {cluster_service}")

        except Exception as e:
            logger.warning(f"Peer discovery failed: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to peers"""
        interval = self.config['network']['heartbeat_interval']

        while self.running:
            try:
                await self._send_heartbeats()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(interval)

    async def _send_heartbeats(self):
        """Send heartbeat to all peers"""
        if not self.client_session:
            return

        heartbeat_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'status': 'alive',
            'stats': self.consensus_layer.get_stats()
        }

        for peer in list(self.peer_nodes):
            try:
                peer_address = peer.split('@')[1]
                url = f"http://{peer_address}/heartbeat"

                async with self.client_session.post(url, json=heartbeat_data, timeout=5) as resp:
                    if resp.status != 200:
                        logger.warning(f"Heartbeat failed for peer {peer}: {resp.status}")

            except Exception as e:
                logger.warning(f"Heartbeat error for peer {peer}: {e}")

    async def _sync_loop(self):
        """Periodic CRDT synchronization with peers"""
        interval = self.config['crdt']['sync_interval']

        while self.running:
            try:
                await self._sync_with_peers()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(interval)

    async def _sync_with_peers(self):
        """Synchronize CRDT data with all peers"""
        if not self.client_session or not self.peer_nodes:
            return

        # Get our current CRDT data
        response = await self.get_crdt_data(None)
        our_data = json.loads(response.text)

        # Sync with each peer
        for peer in list(self.peer_nodes):
            try:
                peer_id, peer_address = peer.split('@')

                # Get peer's CRDT data
                url = f"http://{peer_address}/sync/crdt"
                async with self.client_session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        peer_data = await resp.json()

                        # Sync with our consensus layer
                        await self.consensus_layer.sync_with_peer(
                            peer_id, peer_data.get('crdt_data', {})
                        )

                        # Send our data to peer
                        sync_url = f"http://{peer_address}/sync/crdt"
                        sync_data = {
                            'peer_id': self.node_id,
                            'crdt_data': our_data['crdt_data']
                        }

                        async with self.client_session.post(sync_url, json=sync_data, timeout=10) as sync_resp:
                            if sync_resp.status != 200:
                                logger.warning(f"CRDT sync failed with peer {peer_id}")

            except Exception as e:
                logger.warning(f"CRDT sync error with peer {peer}: {e}")


def create_consensus_node():
    """Factory function to create consensus node"""
    config_path = os.getenv('NDML_CONFIG_PATH', '/etc/ndml/consensus.yaml')
    return ConsensusNode(config_path)


async def main():
    """Main entry point for consensus node"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start node
    node = create_consensus_node()

    # Setup signal handlers
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(node.stop())

    loop = asyncio.get_event_loop()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, signal_handler)

    try:
        await node.start()

        # Keep running until stopped
        while node.running:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Node error: {e}")
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())