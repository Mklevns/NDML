import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

print("Attempting to import ndml package...")
import ndml
print("Successfully imported ndml")

print("Attempting to import EnhancedDistributedMemoryNode from ndml (top-level)...")
from ndml import EnhancedDistributedMemoryNode # Changed this line
print("Successfully imported EnhancedDistributedMemoryNode from ndml (top-level)")

print("Attempting to import MemoryGateway from ndml.integration...")
from ndml.integration.memory_gateway import MemoryGateway
print("Successfully imported MemoryGateway")

print("Attempting to import ConsensusNode from ndml.deployment...")
from ndml.deployment.consensus_node import ConsensusNode
print("Successfully imported ConsensusNode")

# Assuming consensus/__init__.py might be empty or not export anything specific yet based on its template
# print("Attempting to import ... from ndml.consensus...")
# from ndml.consensus import ...
# print("Successfully imported ... from ndml.consensus")

print("Attempting to import normalize_embeddings from ndml.utils...")
from ndml.utils import normalize_embeddings # This assumes utils/__init__.py exports it
print("Successfully imported normalize_embeddings")

print("Attempting to import get_version directly from ndml...")
from ndml import get_version
print("Successfully imported get_version")

print("Attempting to call ndml.get_version()...")
version = ndml.get_version()
print(f"NDML version: {version}")

print("All import checks passed.")
