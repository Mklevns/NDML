# NDML (Neural Distributed Memory Learning) Requirements

This document outlines the requirements for the NDML system, including hardware, software dependencies, and system configuration.

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 32GB (less may impact performance)
- **Storage**: 20GB available space for installation and models
- **GPU**: Optional but recommended for optimal performance

### Recommended Requirements
- **CPU**: 8+ core processor
- **RAM**: 64GB or more for large-scale memory operations
- **Storage**: 100GB+ SSD for faster data access
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (16GB+ recommended for larger models)

## Software Requirements

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows support planned (see tasks.md)

### Core Dependencies
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **PyTorch**: Latest stable version with CUDA support (if GPU available)
- **FAISS**: For efficient similarity search (GPU version if CUDA available)
- **Redis**: For distributed memory operations and caching

### Python Packages
- torch, torchvision, torchaudio
- faiss-gpu (or faiss-cpu for non-GPU systems)
- transformers (for LLM integration)
- numpy
- asyncio (for asynchronous operations)
- logging
- typing
- hashlib
- threading
- concurrent.futures

## System Configuration

### Directory Structure
The system requires the following directory structure:
```
/opt/ndml/
  ├── config/
  ├── data/
  │   ├── memory/
  │   └── checkpoints/
  ├── logs/
  ├── models/
  ├── monitoring/
  └── scripts/
```

### Environment Variables
The following environment variables should be configured:
- NDML_HOME: /opt/ndml
- NDML_DATA_PATH: /opt/ndml/data
- NDML_CONFIG_PATH: /opt/ndml/config
- NDML_LOG_PATH: /opt/ndml/logs
- NDML_MODEL_PATH: /opt/ndml/models
- NDML_LOG_LEVEL: INFO
- PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:512
- OMP_NUM_THREADS: 8
- REDIS_HOST: localhost
- REDIS_PORT: 6379
- PYTHONPATH: $PYTHONPATH:/opt/ndml

## Installation

The installation process is automated through the `install_ndml.sh` script, which:
1. Checks system requirements
2. Creates a Python virtual environment
3. Installs all dependencies
4. Sets up the directory structure
5. Configures environment variables
6. Installs and configures Redis
7. Downloads pre-trained models (if specified)

## System Components

The NDML system consists of several key components:

### Core Components
- **GPUAcceleratedDMN (DMN)**: The central memory node that manages memory traces
- **MemoryTrace**: Represents individual memory items with metadata
- **BTSPUpdateMechanism**: Biologically-inspired plasticity mechanism
- **MemoryLifecycleManager**: Manages memory lifecycle (creation, consolidation, eviction)
- **MultiTimescaleDynamicsEngine**: Handles temporal dynamics of memory

### Integration Components
- **NDMLIntegratedLLM**: Integrates NDML with language models
- **MemoryGateway**: Interface for memory operations
- **MemoryFusionNetwork**: Fuses retrieved memories with model processing

### LLM Support
- Transformer-based models (GPT, Llama, Mistral)
- State-space models (Mamba)

## Performance Considerations

- Memory usage scales with the number of memory traces and their dimension
- GPU acceleration significantly improves performance for both memory operations and LLM inference
- Redis performance affects distributed operations
- Consider adjusting batch sizes and memory limits based on available hardware