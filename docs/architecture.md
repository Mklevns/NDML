# Architecture

This document outlines the high-level architecture of the NDML system. It describes the main components and how they interact with each other.

## Core Components

The NDML system is built around a set of core components that provide the fundamental functionalities for neuromorphic distributed memory.

### Enhanced Distributed Memory Node (DMN)
Individual memory storage units with biological plasticity. These nodes are responsible for storing and managing memory traces. They implement mechanisms for memory encoding, retrieval, and local updates based on biological principles.

### Multi-Timescale Dynamics Engine
Coordinates temporal processing across five distinct timescales:
- Fast Synaptic (milliseconds)
- Calcium Plasticity (sub-seconds)
- Protein Synthesis (minutes)
- Homeostatic Scaling (hours)
- Systems Consolidation (days)
This engine ensures that memory processes occur over realistic biological timeframes.

### BTSP Update Mechanism
The Biological Tag-and-Store Plasticity (BTSP) mechanism governs intelligent memory updates. It determines whether a new piece of information is stored, or an existing memory trace is updated, based on factors like:
- Novelty
- Importance
- Error (prediction error)

### Memory Lifecycle Manager
Manages the lifecycle of memory traces, including:
- Aging of memories over time
- Consolidation of important memories into long-term storage
- Eviction of outdated or less relevant memories to free up resources

## Integration Components

These components facilitate the interaction of the core NDML system with external systems, particularly Large Language Models (LLMs).

### MemoryGateway
The MemoryGateway coordinates distributed memory operations across multiple MemoryClusters. It acts as a central access point for storing and retrieving memories from the distributed network of DMNs. It handles routing of requests and aggregation of results.

### MemoryCluster
A MemoryCluster is a group of Enhanced Distributed Memory Nodes (DMNs). Clustering DMNs allows for better organization, scalability, and fault tolerance. The MemoryGateway interacts with MemoryClusters to manage the distributed memory.

### NDMLIntegratedLLM
This component provides a seamless integration of the NDML system with Large Language Models (LLMs). It allows an LLM to leverage the neuromorphic memory capabilities of NDML for tasks such as:
- Storing information learned during conversations
- Retrieving relevant memories to inform responses
- Contextualizing LLM outputs based on past interactions

This wrapper enables LLMs to have a persistent, dynamic memory, overcoming some limitations of standard LLM architectures.

## Component Interactions

The typical flow of information and interaction between these components is as follows:

1.  An external system (e.g., an application using an LLM) interacts with the **NDMLIntegratedLLM**.
2.  The **NDMLIntegratedLLM** processes requests, which may involve storing new information or retrieving existing memories.
3.  For memory operations, the **NDMLIntegratedLLM** communicates with the **MemoryGateway**.
4.  The **MemoryGateway** routes requests to the appropriate **MemoryCluster(s)**.
5.  Within a **MemoryCluster**, the request is handled by one or more **Enhanced Distributed Memory Nodes (DMNs)**.
6.  The **DMNs** utilize the **Multi-Timescale Dynamics Engine** and **BTSP Update Mechanism** to process memory updates and lifecycle events, managed by the **Memory Lifecycle Manager**.
7.  Retrieved memories are passed back up the chain to the **NDMLIntegratedLLM**, which then integrates them into the LLM's processing.

This architecture allows for a scalable, biologically-inspired memory system that can enhance the capabilities of modern AI models.
