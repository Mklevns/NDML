#!/usr/bin/env python3
"""NDML Inference Server"""

import os
import asyncio
import logging
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import time

# Import NDML components
from ndml.integration.llm_wrapper import NDMLIntegratedLLM
from ndml.integration.memory_gateway import MemoryGateway
from ndml.deployment.health_monitor import HealthMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="NDML Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ndml_model = None
health_monitor = None
server_config = None


# Request/Response models
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None
    max_tokens: int = 512
    temperature: float = 0.7
    update_memory: bool = True


class ChatResponse(BaseModel):
    response: str
    memory_stats: Dict[str, Any]
    processing_time: float


class MemoryQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    k: int = 10


class MemoryUpdateRequest(BaseModel):
    content: str
    context: Dict[str, Any]
    salience: float
    user_feedback: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    uptime: float
    memory_stats: Dict[str, Any]
    gpu_available: bool
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize NDML system on startup"""
    global ndml_model, health_monitor, server_config

    logger.info("Starting NDML server...")

    # Load configuration
    config_path = os.environ.get('NDML_CONFIG_PATH', '/opt/ndml/config/server.yaml')
    server_config = load_config(config_path)

    # Initialize NDML model
    ndml_model = NDMLIntegratedLLM(
        model_name_or_path=server_config['model']['name'],
        memory_dimension=server_config['memory']['dimension'],
        memory_config=server_config['memory']
    )

    # Initialize health monitor
    health_monitor = HealthMonitor(ndml_model)

    # Start background tasks
    asyncio.create_task(periodic_maintenance())

    logger.info("NDML server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down NDML server...")

    # Save memory checkpoint
    if ndml_model:
        checkpoint_path = os.path.join(
            os.environ.get('NDML_DATA_PATH', '/opt/ndml/data'),
            'checkpoints',
            f'shutdown_{int(time.time())}.pt'
        )
        ndml_model.save_memory_checkpoint(checkpoint_path)

    logger.info("NDML server shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Server not initialized")

    stats = health_monitor.get_health_stats()

    return HealthResponse(
        status="healthy" if stats['healthy'] else "unhealthy",
        uptime=stats['uptime'],
        memory_stats=stats['memory_stats'],
        gpu_available=stats['gpu_available'],
        gpu_memory_used=stats.get('gpu_memory_used'),
        gpu_memory_total=stats.get('gpu_memory_total')
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Chat completion endpoint"""
    if not ndml_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    start_time = time.time()

    try:
        # Process chat request
        response = await ndml_model.chat_completion(
            messages=request.messages,
            context=request.context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Get memory statistics
        memory_stats = ndml_model.get_memory_stats()

        processing_time = time.time() - start_time

        return ChatResponse(
            response=response,
            memory_stats=memory_stats,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    """Query memory system"""
    if not ndml_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Encode query
        inputs = ndml_model.tokenizer(
            request.query,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Get query embedding
        with torch.no_grad():
            hidden_states = ndml_model.adapter.get_hidden_states(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            query_embedding = ndml_model.adapter.query_projection(hidden_states)

        # Query memory
        results = await ndml_model.memory_gateway.retrieve_memories_async(
            query=query_embedding,
            context=request.context or {},
            k=request.k
        )

        # Format results
        formatted_results = []
        for memory, score in results:
            formatted_results.append({
                'trace_id': memory.trace_id,
                'content': memory.context.get('text', ''),
                'score': float(score),
                'timestamp': memory.timestamp,
                'access_count': memory.access_count
            })

        return {
            'results': formatted_results,
            'total_results': len(formatted_results)
        }

    except Exception as e:
        logger.error(f"Memory query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/update")
async def update_memory(request: MemoryUpdateRequest):
    """Manually update memory"""
    if not ndml_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Encode content
        inputs = ndml_model.tokenizer(
            request.content,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Get content embedding
        with torch.no_grad():
            hidden_states = ndml_model.adapter.get_hidden_states(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            content_embedding = ndml_model.adapter.query_projection(hidden_states)

        # Add to memory
        success = await ndml_model.memory_gateway.add_memory_async(
            content=content_embedding,
            context=request.context,
            salience=request.salience,
            user_feedback=request.user_feedback
        )

        return {
            'success': success,
            'message': 'Memory updated successfully' if success else 'Memory update failed'
        }

    except Exception as e:
        logger.error(f"Memory update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def get_memory_stats():
    """Get comprehensive memory statistics"""
    if not ndml_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return ndml_model.get_memory_stats()


@app.post("/memory/consolidate")
async def trigger_consolidation(background_tasks: BackgroundTasks):
    """Trigger memory consolidation"""
    if not ndml_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    background_tasks.add_task(run_consolidation)

    return {
        'message': 'Consolidation triggered',
        'status': 'running'
    }


# Background tasks
async def periodic_maintenance():
    """Periodic maintenance tasks"""
    while True:
        try:
            # Run every hour
            await asyncio.sleep(3600)

            logger.info("Running periodic maintenance...")

            # Trigger memory consolidation
            await run_consolidation()

            # Save checkpoint
            checkpoint_path = os.path.join(
                os.environ.get('NDML_DATA_PATH', '/opt/ndml/data'),
                'checkpoints',
                f'periodic_{int(time.time())}.pt'
            )
            ndml_model.save_memory_checkpoint(checkpoint_path)

            logger.info("Periodic maintenance complete")

        except Exception as e:
            logger.error(f"Maintenance error: {e}")


async def run_consolidation():
    """Run memory consolidation"""
    if ndml_model and ndml_model.memory_gateway:
        await ndml_model.memory_gateway.periodic_maintenance()


def load_config(config_path):
    """Load server configuration"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='NDML Inference Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()

    # Run server
    uvicorn.run(
        "start_ndml_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()