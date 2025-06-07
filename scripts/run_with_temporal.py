#!/usr/bin/env python3
# scripts/run_with_temporal.py - Run NDML with temporal dynamics

import asyncio
import logging
import yaml
from pathlib import Path

# Import NDML components
from ndml.core.dynamics import MultiTimescaleDynamicsEngine
from ndml.core.dmn import DistributedMemoryNode
from ndml.core.btsp import BTSPMechanism
from ndml.integration.temporal_bridge import TemporalLLMBridge
from ndml.integration.memory_gateway import MemoryGateway

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """Main function to run NDML with temporal dynamics."""
    
    logging.info("Starting NDML with Multi-Timescale Dynamics...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "temporal_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize temporal dynamics engine
    temporal_config = config.get("temporal_dynamics", {})
    temporal_engine = MultiTimescaleDynamicsEngine(temporal_config)
    
    # Initialize NDML components
    dmn_config = config.get("dmn", {})
    dmn = DistributedMemoryNode("dmn_01", dmn_config)
    
    btsp_config = config.get("btsp", {})
    btsp = BTSPMechanism(btsp_config)
    
    # Create LLM bridge
    llm_bridge = TemporalLLMBridge(temporal_engine)
    
    # Integrate components
    await dmn.integrate_temporal_dynamics(temporal_engine)
    await btsp.integrate_temporal_dynamics(temporal_engine)
    
    # Start temporal engine
    await temporal_engine.start()
    
    logging.info("NDML temporal system initialized successfully")
    
    # Example usage
    await run_temporal_example(temporal_engine, dmn, btsp, llm_bridge)
    
    # Cleanup
    await temporal_engine.stop()
    logging.info("NDML temporal system stopped")

async def run_temporal_example(temporal_engine, dmn, btsp, llm_bridge):
    """Run an example with temporal dynamics."""
    
    logging.info("Running temporal dynamics example...")
    
    # Simulate LLM token processing
    import numpy as np
    
    for step in range(100):
        
        # Simulate token embeddings
        token_embeddings = [np.random.randn(64) for _ in range(3)]
        
        # Process through temporal bridge
        temporal_output = await llm_bridge.process_llm_token_sequence(
            token_embeddings, step * 3
        )
        
        # Store some memories in DMN
        if step % 10 == 0:
            from ndml.core.memory_trace import MemoryTrace
            
            trace = MemoryTrace(
                content=np.random.randn(64),
                context={"step": step, "type": "example"},
                salience=0.5 + np.random.rand() * 0.5,
                timestamp=time.time(),
                trace_id=f"example_trace_{step}"
            )
            
            await dmn.store_memory_trace_async(trace)
        
        # Execute temporal step
        temporal_state = await temporal_engine.step()
        
        # Log progress
        if step % 20 == 0:
            performance = temporal_engine.performance_monitor.get_performance_summary()
            logging.info(f"Step {step}: System Health = {performance['system_health']:.3f}")
            
            # Log DMN statistics
            dmn_stats = dmn.get_temporal_statistics()
            logging.info(f"  DMN: {dmn_stats.get('total_traces', 0)} traces, "
                        f"coherence = {dmn_stats.get('average_coherence', 0):.3f}")
            
            # Log BTSP statistics
            btsp_stats = btsp.get_btsp_statistics()
            logging.info(f"  BTSP: {btsp_stats.get('active_traces', 0)} active traces")

if __name__ == "__main__":
    asyncio.run(main())
