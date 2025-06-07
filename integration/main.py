# main.py - NDML System Entry Point and Testing Framework
import asyncio
import logging
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ndml_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Import NDML components
try:
    from core.dmn import EnhancedDistributedMemoryNode
    from core.lifecycle import MemoryLifecycleManager
    from core.dynamics import MultiTimescaleDynamicsEngine
    from core.btsp import BTSPUpdateMechanism
    from core.memory_trace import MemoryTrace
    from integration.memory_gateway import MemoryGateway
    from integration.llm_wrapper import NDMLIntegratedLLM
    from integration.fusion_network import MemoryFusionNetwork
    from integration.temporal_bridge import TemporalLLMBridge
except ImportError as e:
    logger.error(f"Failed to import NDML components: {e}")
    logger.error("Please ensure all NDML modules are properly installed")
    exit(1)


class NDMLSystemManager:
    """
    Main system manager for the NDML (Neuromorphic Distributed Memory Layer).
    
    Coordinates initialization, testing, and operation of all NDML components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize NDML System Manager."""
        self.config = self._load_config(config_path)
        self.components = {}
        self.is_running = False
        self.test_results = {}
        
        logger.info("NDML System Manager initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration."""
        default_config = {
            'system': {
                'dimension': 512,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'num_clusters': 4,
                'nodes_per_cluster': 8,
                'node_capacity': 5000,
            },
            'temporal': {
                'enabled': True,
                'fast_synaptic_duration': 0.005,
                'calcium_duration': 0.5,
                'protein_duration': 60.0,
                'homeostatic_duration': 3600.0,
                'systems_duration': 86400.0,
            },
            'btsp': {
                'calcium_threshold': 0.7,
                'decay_rate': 0.95,
                'novelty_weight': 0.4,
                'importance_weight': 0.3,
                'error_weight': 0.3,
                'learning_rate': 0.1,
            },
            'fusion': {
                'num_attention_heads': 8,
                'fusion_layers': 2,
                'dropout_rate': 0.1,
                'fusion_strategy': 'attention',
            },
            'testing': {
                'num_test_traces': 1000,
                'test_duration': 300,  # 5 minutes
                'batch_size': 32,
                'similarity_threshold': 0.5,
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Deep merge configurations
                self._deep_merge(default_config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    async def initialize_system(self) -> bool:
        """Initialize all NDML system components."""
        try:
            logger.info("Initializing NDML system components...")
            
            # 1. Initialize Temporal Dynamics Engine
            if self.config['temporal']['enabled']:
                logger.info("Initializing Multi-Timescale Dynamics Engine...")
                self.components['temporal_engine'] = MultiTimescaleDynamicsEngine(
                    config=self.config['temporal']
                )
                await self.components['temporal_engine'].start()
            
            # 2. Initialize Memory Gateway
            logger.info("Initializing Memory Gateway...")
            self.components['memory_gateway'] = MemoryGateway(
                dimension=self.config['system']['dimension'],
                num_clusters=self.config['system']['num_clusters'],
                nodes_per_cluster=self.config['system']['nodes_per_cluster'],
                node_capacity=self.config['system']['node_capacity'],
                enable_consensus=False,  # Disable for basic testing
                config={
                    'routing': {'strategy': 'round_robin'},
                    'retrieval': {'default_k': 10}
                }
            )
            
            # 3. Initialize Fusion Network
            logger.info("Initializing Memory Fusion Network...")
            self.components['fusion_network'] = MemoryFusionNetwork(
                model_dimension=768,  # Common LLM dimension
                memory_dimension=self.config['system']['dimension'],
                **self.config['fusion']
            )
            
            # 4. Initialize Temporal Bridge (if temporal engine is enabled)
            if 'temporal_engine' in self.components:
                logger.info("Initializing Temporal-LLM Bridge...")
                self.components['temporal_bridge'] = TemporalLLMBridge(
                    self.components['temporal_engine']
                )
            
            # 5. Start maintenance tasks
            asyncio.create_task(self._maintenance_loop())
            
            self.is_running = True
            logger.info("NDML system initialization completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NDML system: {e}")
            await self.shutdown()
            return False

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests."""
        logger.info("Starting comprehensive NDML system tests...")
        
        test_results = {
            'timestamp': time.time(),
            'system_config': self.config,
            'tests': {}
        }
        
        # 1. Basic Memory Operations Test
        logger.info("Running basic memory operations test...")
        test_results['tests']['basic_memory'] = await self._test_basic_memory_operations()
        
        # 2. Memory Retrieval Test
        logger.info("Running memory retrieval test...")
        test_results['tests']['memory_retrieval'] = await self._test_memory_retrieval()
        
        # 3. Memory Fusion Test
        logger.info("Running memory fusion test...")
        test_results['tests']['memory_fusion'] = await self._test_memory_fusion()
        
        # 4. Temporal Dynamics Test (if enabled)
        if 'temporal_engine' in self.components:
            logger.info("Running temporal dynamics test...")
            test_results['tests']['temporal_dynamics'] = await self._test_temporal_dynamics()
        
        # 5. Lifecycle Management Test
        logger.info("Running lifecycle management test...")
        test_results['tests']['lifecycle_management'] = await self._test_lifecycle_management()
        
        # 6. Performance Stress Test
        logger.info("Running performance stress test...")
        test_results['tests']['performance_stress'] = await self._test_performance_stress()
        
        # 7. System Integration Test
        logger.info("Running system integration test...")
        test_results['tests']['system_integration'] = await self._test_system_integration()
        
        # Calculate overall results
        test_results['summary'] = self._calculate_test_summary(test_results['tests'])
        
        self.test_results = test_results
        
        logger.info("Comprehensive tests completed!")
        logger.info(f"Overall Success Rate: {test_results['summary']['overall_success_rate']:.1%}")
        
        return test_results

    async def _test_basic_memory_operations(self) -> Dict[str, Any]:
        """Test basic memory storage and retrieval operations."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            # Test memory trace creation
            test_content = torch.randn(self.config['system']['dimension'])
            test_context = {'domain': 'test', 'task_type': 'storage_test'}
            
            trace = MemoryTrace(
                content=test_content,
                context=test_context,
                timestamp=time.time(),
                salience=0.8
            )
            
            results['details']['trace_creation'] = True
            
            # Test memory gateway storage
            success = await self.components['memory_gateway'].add_memory_async(
                content=test_content,
                context=test_context,
                salience=0.8
            )
            
            results['details']['gateway_storage'] = success
            
            if not success:
                results['success'] = False
                results['errors'].append("Failed to store memory in gateway")
            
            # Test basic retrieval
            retrieved = await self.components['memory_gateway'].retrieve_memories_async(
                query=test_content,
                context=test_context,
                k=5
            )
            
            results['details']['basic_retrieval'] = len(retrieved) > 0
            results['details']['retrieval_count'] = len(retrieved)
            
            if len(retrieved) == 0:
                results['success'] = False
                results['errors'].append("Failed to retrieve stored memory")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Basic memory operations test failed: {str(e)}"]
            }

    async def _test_memory_retrieval(self) -> Dict[str, Any]:
        """Test memory retrieval with various query patterns."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            # Store test memories with different patterns
            test_memories = []
            for i in range(50):
                content = torch.randn(self.config['system']['dimension'])
                context = {
                    'domain': f'domain_{i % 5}',
                    'category': f'cat_{i % 3}',
                    'importance': np.random.random()
                }
                
                await self.components['memory_gateway'].add_memory_async(
                    content=content,
                    context=context,
                    salience=np.random.random()
                )
                test_memories.append((content, context))
            
            results['details']['test_memories_stored'] = len(test_memories)
            
            # Test similarity-based retrieval
            query_content = test_memories[0][0] + torch.randn(self.config['system']['dimension']) * 0.1
            retrieved = await self.components['memory_gateway'].retrieve_memories_async(
                query=query_content,
                context={},
                k=10
            )
            
            results['details']['similarity_retrieval_count'] = len(retrieved)
            results['details']['similarity_retrieval_success'] = len(retrieved) > 0
            
            # Test context-filtered retrieval
            context_filter = {'domain': 'domain_0'}
            filtered_retrieved = await self.components['memory_gateway'].retrieve_memories_async(
                query=torch.randn(self.config['system']['dimension']),
                context=context_filter,
                k=10
            )
            
            results['details']['context_filtered_count'] = len(filtered_retrieved)
            results['details']['context_filtering_success'] = True
            
            # Test diversity in results
            if len(retrieved) > 1:
                similarities = []
                for i in range(len(retrieved) - 1):
                    sim = torch.cosine_similarity(
                        retrieved[i][0].content,
                        retrieved[i+1][0].content,
                        dim=0
                    ).item()
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                results['details']['average_result_similarity'] = avg_similarity
                results['details']['diversity_good'] = avg_similarity < 0.9
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Memory retrieval test failed: {str(e)}"]
            }

    async def _test_memory_fusion(self) -> Dict[str, Any]:
        """Test memory fusion network functionality."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            batch_size = 4
            num_memories = 8
            model_dim = 768
            memory_dim = self.config['system']['dimension']
            
            # Create test data
            query_states = torch.randn(batch_size, model_dim)
            memory_embeddings = torch.randn(batch_size, num_memories, memory_dim)
            
            # Test fusion network forward pass
            fusion_output = self.components['fusion_network'](
                query_states=query_states,
                memory_embeddings=memory_embeddings
            )
            
            # Validate output structure
            expected_keys = ['fused_states', 'attention_weights', 'confidence', 'fusion_metadata']
            has_all_keys = all(key in fusion_output for key in expected_keys)
            results['details']['output_structure_valid'] = has_all_keys
            
            if not has_all_keys:
                results['success'] = False
                results['errors'].append("Missing keys in fusion output")
            
            # Validate output shapes
            fused_states = fusion_output['fused_states']
            results['details']['output_shape_correct'] = (
                fused_states.shape == (batch_size, model_dim)
            )
            
            # Test attention weights
            if fusion_output['attention_weights'] is not None:
                attention_weights = fusion_output['attention_weights']
                attention_shape_correct = attention_weights.shape == (batch_size, num_memories)
                results['details']['attention_shape_correct'] = attention_shape_correct
                
                # Check if attention weights sum to 1
                attention_sums = attention_weights.sum(dim=1)
                attention_normalized = torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5)
                results['details']['attention_normalized'] = attention_normalized
            
            # Test confidence scores
            confidence = fusion_output['confidence']
            confidence_valid = (
                confidence.shape == (batch_size, 1) and
                torch.all(confidence >= 0) and
                torch.all(confidence <= 1)
            )
            results['details']['confidence_valid'] = confidence_valid
            
            # Test fusion statistics
            fusion_stats = self.components['fusion_network'].get_fusion_stats()
            results['details']['fusion_stats_available'] = len(fusion_stats) > 0
            results['details']['fusion_stats'] = fusion_stats
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Memory fusion test failed: {str(e)}"]
            }

    async def _test_temporal_dynamics(self) -> Dict[str, Any]:
        """Test temporal dynamics engine functionality."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            temporal_engine = self.components['temporal_engine']
            
            # Test temporal engine state
            temporal_state = temporal_engine.get_temporal_state()
            results['details']['temporal_state_available'] = temporal_state is not None
            
            # Test event injection
            test_event_success = await temporal_engine.inject_event(
                "external_stimulus",
                "fast_synaptic",
                {"neuron_ids": [0, 1, 2], "strength": 0.03}
            )
            results['details']['event_injection_success'] = test_event_success
            
            # Run several temporal steps
            step_results = []
            for i in range(10):
                step_result = await temporal_engine.step()
                step_results.append(step_result)
                
                # Inject some events periodically
                if i % 3 == 0:
                    await temporal_engine.inject_event(
                        "memory_trace_activation",
                        "fast_synaptic",
                        {
                            "trace_data": {"content": np.random.randn(10)},
                            "salience": 0.8
                        }
                    )
            
            results['details']['temporal_steps_completed'] = len(step_results)
            results['details']['temporal_coherence_maintained'] = all(
                hasattr(step, 'coherence_metric') and step.coherence_metric > 0
                for step in step_results
            )
            
            # Test performance monitoring
            performance_summary = temporal_engine.performance_monitor.get_performance_summary()
            results['details']['performance_monitoring_available'] = len(performance_summary) > 0
            results['details']['system_health'] = performance_summary.get('system_health', 0.0)
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Temporal dynamics test failed: {str(e)}"]
            }

    async def _test_lifecycle_management(self) -> Dict[str, Any]:
        """Test memory lifecycle management functionality."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            # Create a test lifecycle manager
            from core.lifecycle import LifecycleConfig
            lifecycle_config = LifecycleConfig(
                eviction_batch_size=10,
                consolidation_interval=60.0,  # 1 minute for testing
                maintenance_interval=30.0     # 30 seconds for testing
            )
            
            lifecycle_manager = MemoryLifecycleManager(
                node_id="test_node",
                config=lifecycle_config
            )
            
            # Create test memory traces
            test_traces = []
            current_time = time.time()
            
            for i in range(50):
                content = torch.randn(self.config['system']['dimension'])
                trace = MemoryTrace(
                    content=content,
                    context={'domain': f'test_{i%5}', 'priority': np.random.random()},
                    timestamp=current_time - np.random.random() * 3600,  # Random age up to 1 hour
                    salience=np.random.random()
                )
                # Simulate some access patterns
                trace.last_access = current_time - np.random.random() * 1800  # Random recency up to 30 min
                trace.access_count = np.random.randint(0, 20)
                test_traces.append(trace)
            
            results['details']['test_traces_created'] = len(test_traces)
            
            # Test lifecycle state evaluation
            lifecycle_states = []
            for trace in test_traces[:10]:  # Test first 10
                state = await lifecycle_manager.evaluate_trace_lifecycle(trace, current_time)
                lifecycle_states.append(state)
            
            results['details']['lifecycle_evaluation_success'] = len(lifecycle_states) == 10
            results['details']['lifecycle_states'] = [state.value for state in lifecycle_states]
            
            # Test eviction candidate selection
            eviction_candidates = await lifecycle_manager.select_eviction_candidates(
                test_traces, num_to_evict=5
            )
            results['details']['eviction_selection_success'] = len(eviction_candidates) <= 5
            results['details']['eviction_candidates_count'] = len(eviction_candidates)
            
            # Test maintenance cycle
            maintenance_results = await lifecycle_manager.perform_maintenance_cycle(test_traces)
            results['details']['maintenance_cycle_success'] = 'error' not in maintenance_results
            results['details']['maintenance_results'] = maintenance_results
            
            # Test statistics
            lifecycle_stats = lifecycle_manager.get_lifecycle_statistics()
            results['details']['statistics_available'] = len(lifecycle_stats) > 0
            results['details']['lifecycle_statistics'] = lifecycle_stats
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Lifecycle management test failed: {str(e)}"]
            }

    async def _test_performance_stress(self) -> Dict[str, Any]:
        """Test system performance under stress conditions."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            # Configuration for stress test
            num_operations = self.config['testing']['num_test_traces']
            batch_size = self.config['testing']['batch_size']
            
            # Measure memory storage performance
            storage_times = []
            storage_start = time.time()
            
            for i in range(0, num_operations, batch_size):
                batch_start = time.time()
                
                # Create batch of memories
                batch_tasks = []
                for j in range(min(batch_size, num_operations - i)):
                    content = torch.randn(self.config['system']['dimension'])
                    context = {
                        'domain': f'stress_test_{j%10}',
                        'batch': i // batch_size,
                        'index': j
                    }
                    
                    task = self.components['memory_gateway'].add_memory_async(
                        content=content,
                        context=context,
                        salience=np.random.random()
                    )
                    batch_tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                batch_time = time.time() - batch_start
                storage_times.append(batch_time)
                
                # Check for failures
                failures = sum(1 for result in batch_results if isinstance(result, Exception))
                if failures > 0:
                    results['errors'].append(f"Batch {i//batch_size}: {failures} storage failures")
            
            storage_total_time = time.time() - storage_start
            
            results['details']['storage_performance'] = {
                'total_time': storage_total_time,
                'average_batch_time': np.mean(storage_times),
                'std_batch_time': np.std(storage_times),
                'operations_per_second': num_operations / storage_total_time,
                'batches_completed': len(storage_times)
            }
            
            # Measure retrieval performance
            retrieval_times = []
            retrieval_start = time.time()
            
            for i in range(100):  # 100 retrieval operations
                query_start = time.time()
                query = torch.randn(self.config['system']['dimension'])
                
                retrieved = await self.components['memory_gateway'].retrieve_memories_async(
                    query=query,
                    context={},
                    k=10
                )
                
                query_time = time.time() - query_start
                retrieval_times.append(query_time)
            
            retrieval_total_time = time.time() - retrieval_start
            
            results['details']['retrieval_performance'] = {
                'total_time': retrieval_total_time,
                'average_query_time': np.mean(retrieval_times),
                'std_query_time': np.std(retrieval_times),
                'queries_per_second': 100 / retrieval_total_time,
                'queries_completed': len(retrieval_times)
            }
            
            # Memory usage analysis
            gateway_stats = self.components['memory_gateway'].get_comprehensive_stats()
            results['details']['memory_usage'] = {
                'total_memories': gateway_stats['system_summary']['total_memories'],
                'average_utilization': gateway_stats['system_summary']['average_utilization'],
                'cluster_count': gateway_stats['system_summary']['total_clusters'],
                'node_count': gateway_stats['system_summary']['total_nodes']
            }
            
            # Performance thresholds
            storage_ops_per_sec = results['details']['storage_performance']['operations_per_second']
            retrieval_ops_per_sec = results['details']['retrieval_performance']['queries_per_second']
            
            results['details']['performance_acceptable'] = (
                storage_ops_per_sec > 50 and  # At least 50 storage ops/sec
                retrieval_ops_per_sec > 100   # At least 100 retrieval ops/sec
            )
            
            if not results['details']['performance_acceptable']:
                results['success'] = False
                results['errors'].append(
                    f"Performance below threshold: storage={storage_ops_per_sec:.1f}/s, "
                    f"retrieval={retrieval_ops_per_sec:.1f}/s"
                )
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"Performance stress test failed: {str(e)}"]
            }

    async def _test_system_integration(self) -> Dict[str, Any]:
        """Test integration between all system components."""
        try:
            results = {'success': True, 'details': {}, 'errors': []}
            
            # Test end-to-end memory flow
            test_input = torch.randn(self.config['system']['dimension'])
            test_context = {
                'domain': 'integration_test',
                'task_type': 'end_to_end',
                'importance': 0.9
            }
            
            # 1. Store memory through gateway
            storage_success = await self.components['memory_gateway'].add_memory_async(
                content=test_input,
                context=test_context,
                salience=0.9
            )
            results['details']['end_to_end_storage'] = storage_success
            
            # 2. Retrieve memory
            retrieved_memories = await self.components['memory_gateway'].retrieve_memories_async(
                query=test_input + torch.randn(self.config['system']['dimension']) * 0.1,
                context=test_context,
                k=5
            )
            results['details']['end_to_end_retrieval'] = len(retrieved_memories) > 0
            
            # 3. Test fusion with retrieved memories
            if retrieved_memories:
                query_states = torch.randn(2, 768)  # Batch of 2, model_dim=768
                memory_embeddings = torch.stack([mem.content for mem, _ in retrieved_memories[:3]])
                memory_embeddings = memory_embeddings.unsqueeze(0).repeat(2, 1, 1)  # Batch dimension
                
                fusion_result = self.components['fusion_network'](
                    query_states=query_states,
                    memory_embeddings=memory_embeddings
                )
                results['details']['end_to_end_fusion'] = 'fused_states' in fusion_result
            
            # 4. Test temporal integration (if available)
            if 'temporal_engine' in self.components and 'temporal_bridge' in self.components:
                # Simulate LLM token processing
                token_embeddings = [torch.randn(self.config['system']['dimension']) for _ in range(10)]
                
                temporal_output = await self.components['temporal_bridge'].process_llm_token_sequence(
                    token_embeddings=token_embeddings,
                    context_length=20
                )
                results['details']['temporal_integration'] = 'temporal_state' in temporal_output
            
            # 5. Test component statistics integration
            all_stats = {}
            
            # Gateway stats
            gateway_stats = self.components['memory_gateway'].get_comprehensive_stats()
            all_stats['memory_gateway'] = gateway_stats
            
            # Fusion stats
            fusion_stats = self.components['fusion_network'].get_fusion_stats()
            all_stats['fusion_network'] = fusion_stats
            
            # Temporal stats (if available)
            if 'temporal_engine' in self.components:
                temporal_stats = self.components['temporal_engine'].performance_monitor.get_performance_summary()
                all_stats['temporal_engine'] = temporal_stats
            
            results['details']['statistics_integration'] = len(all_stats) >= 2
            results['details']['component_statistics'] = all_stats
            
            # 6. Test system health monitoring
            system_health_indicators = {
                'memory_utilization': gateway_stats['system_summary']['average_utilization'],
                'fusion_performance': fusion_stats.get('fusion_time', 0.0),
                'component_count': len(self.components),
                'error_count': len(results['errors'])
            }
            
            overall_health = (
                system_health_indicators['memory_utilization'] < 0.9 and
                system_health_indicators['fusion_performance'] < 1.0 and
                system_health_indicators['error_count'] == 0
            )
            
            results['details']['system_health'] = overall_health
            results['details']['health_indicators'] = system_health_indicators
            
            if not overall_health:
                results['success'] = False
                results['errors'].append("System health indicators below acceptable thresholds")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'errors': [f"System integration test failed: {str(e)}"]
            }

    def _calculate_test_summary(self, test_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall test summary statistics."""
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
        
        all_errors = []
        for test_name, result in test_results.items():
            if 'errors' in result:
                all_errors.extend([f"{test_name}: {error}" for error in result['errors']])
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'overall_success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'total_errors': len(all_errors),
            'error_summary': all_errors[:10],  # First 10 errors
            'all_tests_passed': successful_tests == total_tests
        }

    async def _maintenance_loop(self):
        """Background maintenance loop for system health."""
        while self.is_running:
            try:
                # Perform periodic maintenance on memory gateway
                if 'memory_gateway' in self.components:
                    await self.components['memory_gateway'].periodic_maintenance()
                
                # Log system health
                if hasattr(self, 'test_results') and self.test_results:
                    logger.debug("System maintenance cycle completed")
                
                # Wait for next maintenance cycle (1 hour)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(60)  # Shorter wait on error

    async def save_test_results(self, filepath: str):
        """Save test results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            logger.info(f"Test results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    async def shutdown(self):
        """Shutdown NDML system gracefully."""
        logger.info("Shutting down NDML system...")
        
        self.is_running = False
        
        # Shutdown temporal engine if running
        if 'temporal_engine' in self.components:
            try:
                await self.components['temporal_engine'].stop()
            except Exception as e:
                logger.error(f"Error stopping temporal engine: {e}")
        
        # Clean up other components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {component_name}: {e}")
        
        logger.info("NDML system shutdown completed")


async def main():
    """Main entry point for NDML system."""
    parser = argparse.ArgumentParser(description="NDML System - Neuromorphic Distributed Memory Layer")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--test', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--output', type=str, default='ndml_test_results.json', 
                       help='Output file for test results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system manager
    system_manager = NDMLSystemManager(config_path=args.config)
    
    try:
        # Initialize the system
        logger.info("Starting NDML System initialization...")
        init_success = await system_manager.initialize_system()
        
        if not init_success:
            logger.error("System initialization failed")
            return 1
        
        if args.test:
            # Run comprehensive tests
            logger.info("Running comprehensive NDML tests...")
            test_results = await system_manager.run_comprehensive_tests()
            
            # Save results
            await system_manager.save_test_results(args.output)
            
            # Print summary
            summary = test_results['summary']
            print(f"\n{'='*60}")
            print("NDML SYSTEM TEST RESULTS")
            print(f"{'='*60}")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Successful: {summary['successful_tests']}")
            print(f"Failed: {summary['failed_tests']}")
            print(f"Success Rate: {summary['overall_success_rate']:.1%}")
            print(f"{'='*60}")
            
            if summary['total_errors'] > 0:
                print(f"\nErrors encountered ({summary['total_errors']} total):")
                for error in summary['error_summary']:
                    print(f"  - {error}")
            
            if summary['all_tests_passed']:
                print("\n✅ All tests passed! NDML system is ready for deployment.")
                return_code = 0
            else:
                print("\n❌ Some tests failed. Please review the results and fix issues.")
                return_code = 1
        else:
            # Interactive mode
            logger.info("NDML System initialized successfully!")
            logger.info("System is running in interactive mode.")
            logger.info("Press Ctrl+C to shutdown.")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                return_code = 0
        
    except Exception as e:
        logger.error(f"System error: {e}")
        return_code = 1
    
    finally:
        # Shutdown system
        await system_manager.shutdown()
    
    return return_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
