#!/usr/bin/env python3
"""Comprehensive testing script for NDML system"""

import asyncio
import time
import torch
import numpy as np
import argparse
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd

from ndml.integration.llm_wrapper import NDMLIntegratedLLM
from ndml.core.memory_trace import MemoryTrace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NDMLTester:
    """Test suite for NDML system"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }

        # Initialize model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize NDML model for testing"""
        logger.info("Initializing NDML model for testing...")

        model = NDMLIntegratedLLM(
            model_name_or_path=self.model_path,
            memory_dimension=self.config.get('memory_dimension', 512),
            memory_config=self.config.get('memory', {})
        )

        return model

    async def test_memory_operations(self):
        """Test basic memory operations"""
        logger.info("Testing memory operations...")

        test_results = {
            'add_memory': {},
            'retrieve_memory': {},
            'update_memory': {}
        }

        # Test adding memories
        start_time = time.time()
        success_count = 0

        for i in range(100):
            content = f"Test memory content {i}"
            inputs = self.model.tokenizer(content, return_tensors="pt")

            with torch.no_grad():
                hidden_states = self.model.adapter.get_hidden_states(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                embedding = self.model.adapter.query_projection(hidden_states)

            success = await self.model.memory_gateway.add_memory_async(
                content=embedding,
                context={'test_id': i, 'type': 'test'},
                salience=0.5 + (i % 5) * 0.1
            )

            if success:
                success_count += 1

        add_time = time.time() - start_time
        test_results['add_memory'] = {
            'total_attempts': 100,
            'successful': success_count,
            'total_time': add_time,
            'avg_time_per_add': add_time / 100
        }

        # Test retrieving memories
        query_times = []
        query_results = []

        for i in range(20):
            query = f"Test query {i % 5}"
            inputs = self.model.tokenizer(query, return_tensors="pt")

            with torch.no_grad():
                hidden_states = self.model.adapter.get_hidden_states(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                query_embedding = self.model.adapter.query_projection(hidden_states)

            start_time = time.time()
            results = await self.model.memory_gateway.retrieve_memories_async(
                query=query_embedding,
                context={'type': 'test'},
                k=5
            )
            query_time = time.time() - start_time

            query_times.append(query_time)
            query_results.append(len(results))

        test_results['retrieve_memory'] = {
            'num_queries': 20,
            'avg_query_time': np.mean(query_times),
            'min_query_time': np.min(query_times),
            'max_query_time': np.max(query_times),
            'avg_results_returned': np.mean(query_results)
        }

        self.results['tests']['memory_operations'] = test_results
        return test_results

    async def test_scalability(self):
        """Test system scalability"""
        logger.info("Testing scalability...")

        memory_counts = [100, 500, 1000, 5000, 10000]
        results = {
            'memory_count': [],
            'add_time': [],
            'query_time': [],
            'memory_usage': []
        }

        for count in memory_counts:
            logger.info(f"Testing with {count} memories...")

            # Clear existing memories (if method exists)
            # self.model.memory_gateway.clear_memories()

            # Add memories
            start_time = time.time()
            for i in range(count):
                content = f"Scalability test memory {i}"
                inputs = self.model.tokenizer(content, return_tensors="pt")

                with torch.no_grad():
                    hidden_states = self.model.adapter.get_hidden_states(
                        inputs['input_ids'],
                        inputs['attention_mask']
                    )
                    embedding = self.model.adapter.query_projection(hidden_states)

                await self.model.memory_gateway.add_memory_async(
                    content=embedding,
                    context={'test': 'scalability', 'id': i},
                    salience=0.5
                )

            add_time = time.time() - start_time

            # Test query performance
            query_times = []
            for _ in range(10):
                query = "Scalability test query"
                inputs = self.model.tokenizer(query, return_tensors="pt")

                with torch.no_grad():
                    hidden_states = self.model.adapter.get_hidden_states(
                        inputs['input_ids'],
                        inputs['attention_mask']
                    )
                    query_embedding = self.model.adapter.query_projection(hidden_states)

                start_time = time.time()
                await self.model.memory_gateway.retrieve_memories_async(
                    query=query_embedding,
                    context={},
                    k=10
                )
                query_times.append(time.time() - start_time)

            # Record results
            results['memory_count'].append(count)
            results['add_time'].append(add_time)
            results['query_time'].append(np.mean(query_times))

            # Get memory usage
            stats = self.model.get_memory_stats()
            results['memory_usage'].append(
                stats['system_summary']['total_memories']
            )

        self.results['tests']['scalability'] = results
        return results

    async def test_btsp_dynamics(self):
        """Test BTSP update dynamics"""
        logger.info("Testing BTSP dynamics...")

        # Test repeated exposures to same content
        content = "BTSP test content"
        inputs = self.model.tokenizer(content, return_tensors="pt")

        with torch.no_grad():
            hidden_states = self.model.adapter.get_hidden_states(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            embedding = self.model.adapter.query_projection(hidden_states)

        update_results = []

        for i in range(20):
            # Vary salience to test threshold behavior
            salience = 0.5 + 0.05 * i

            success = await self.model.memory_gateway.add_memory_async(
                content=embedding,
                context={'iteration': i, 'test': 'btsp'},
                salience=salience
            )

            update_results.append({
                'iteration': i,
                'salience': salience,
                'success': success
            })

            # Small delay to simulate time passing
            await asyncio.sleep(0.1)

        # Analyze update pattern
        success_rate = sum(r['success'] for r in update_results) / len(update_results)

        self.results['tests']['btsp_dynamics'] = {
            'total_attempts': len(update_results),
            'success_rate': success_rate,
            'update_pattern': update_results
        }

        return update_results

    async def test_memory_consolidation(self):
        """Test memory consolidation process"""
        logger.info("Testing memory consolidation...")

        # Add memories with varying salience and access patterns
        for i in range(50):
            content = f"Consolidation test memory {i}"
            inputs = self.model.tokenizer(content, return_tensors="pt")

            with torch.no_grad():
                hidden_states = self.model.adapter.get_hidden_states(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                embedding = self.model.adapter.query_projection(hidden_states)

            # Vary salience
            salience = 0.3 + (i % 10) * 0.07

            await self.model.memory_gateway.add_memory_async(
                content=embedding,
                context={'test': 'consolidation', 'id': i},
                salience=salience
            )

        # Get pre-consolidation stats
        pre_stats = self.model.get_memory_stats()

        # Trigger consolidation
        await self.model.memory_gateway.periodic_maintenance()

        # Get post-consolidation stats
        post_stats = self.model.get_memory_stats()

        self.results['tests']['consolidation'] = {
            'pre_consolidation': pre_stats,
            'post_consolidation': post_stats
        }

        return {
            'pre': pre_stats,
            'post': post_stats
        }

    async def test_concurrent_access(self):
        """Test concurrent memory access"""
        logger.info("Testing concurrent access...")

        async def concurrent_task(task_id: int, operation: str):
            """Execute concurrent operations"""
            results = []

            for i in range(10):
                if operation == 'add':
                    content = f"Concurrent task {task_id} memory {i}"
                    inputs = self.model.tokenizer(content, return_tensors="pt")

                    with torch.no_grad():
                        hidden_states = self.model.adapter.get_hidden_states(
                            inputs['input_ids'],
                            inputs['attention_mask']
                        )
                        embedding = self.model.adapter.query_projection(hidden_states)

                    start_time = time.time()
                    success = await self.model.memory_gateway.add_memory_async(
                        content=embedding,
                        context={'task_id': task_id, 'op': 'add'},
                        salience=0.5
                    )
                    duration = time.time() - start_time

                    results.append({
                        'operation': 'add',
                        'success': success,
                        'duration': duration
                    })

                elif operation == 'query':
                    query = f"Concurrent query {i}"
                    inputs = self.model.tokenizer(query, return_tensors="pt")

                    with torch.no_grad():
                        hidden_states = self.model.adapter.get_hidden_states(
                            inputs['input_ids'],
                            inputs['attention_mask']
                        )
                        query_embedding = self.model.adapter.query_projection(hidden_states)

                    start_time = time.time()
                    memories = await self.model.memory_gateway.retrieve_memories_async(
                        query=query_embedding,
                        context={},
                        k=5
                    )
                    duration = time.time() - start_time

                    results.append({
                        'operation': 'query',
                        'num_results': len(memories),
                        'duration': duration
                    })

            return results

        # Run concurrent tasks
        tasks = []
        for i in range(5):
            tasks.append(concurrent_task(i, 'add'))
            tasks.append(concurrent_task(i, 'query'))

        start_time = time.time()
        all_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Analyze results
        flattened_results = [r for task_results in all_results for r in task_results]

        self.results['tests']['concurrent_access'] = {
            'total_operations': len(flattened_results),
            'total_time': total_time,
            'operations_per_second': len(flattened_results) / total_time,
            'avg_operation_time': np.mean([r['duration'] for r in flattened_results])
        }

        return self.results['tests']['concurrent_access']

    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")

        # Save raw results
        report_path = f"ndml_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate visualizations
        if 'scalability' in self.results['tests']:
            self._plot_scalability_results()

        if 'btsp_dynamics' in self.results['tests']:
            self._plot_btsp_dynamics()

        logger.info(f"Test report saved to {report_path}")

        # Print summary
        print("\n=== NDML Test Summary ===")
        for test_name, test_results in self.results['tests'].items():
            print(f"\n{test_name.upper()}:")
            if isinstance(test_results, dict):
                for key, value in test_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

    def _plot_scalability_results(self):
        """Plot scalability test results"""
        results = self.results['tests']['scalability']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Add time vs memory count
        ax1.plot(results['memory_count'], results['add_time'], 'bo-', label='Add Time')
        ax1.set_xlabel('Number of Memories')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Memory Addition Time vs Count')
        ax1.grid(True)

        # Query time vs memory count
        ax2.plot(results['memory_count'], results['query_time'], 'ro-', label='Query Time')
        ax2.set_xlabel('Number of Memories')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Query Time vs Memory Count')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('ndml_scalability_results.png')
        plt.close()

    def _plot_btsp_dynamics(self):
        """Plot BTSP dynamics results"""
        results = self.results['tests']['btsp_dynamics']['update_pattern']

        iterations = [r['iteration'] for r in results]
        saliences = [r['salience'] for r in results]
        successes = [1 if r['success'] else 0 for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Salience over iterations
        ax1.plot(iterations, saliences, 'b-', label='Salience')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Salience')
        ax1.set_title('Salience Values Over Iterations')
        ax1.grid(True)

        # Success rate over iterations
        ax2.bar(iterations, successes, label='Update Success')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Success (1) / Failure (0)')
        ax2.set_title('Update Success Pattern')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('ndml_btsp_dynamics.png')
        plt.close()

    async def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting NDML system tests...")

        # Run tests
        await self.test_memory_operations()
        await self.test_scalability()
        await self.test_btsp_dynamics()
        await self.test_memory_consolidation()
        await self.test_concurrent_access()

        # Generate report
        self.generate_report()

        logger.info("All tests completed!")


async def main():
    parser = argparse.ArgumentParser(description='Test NDML System')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--config', help='Test configuration file')
    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Create tester and run tests
    tester = NDMLTester(args.model, config)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())